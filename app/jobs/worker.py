"""可恢复持久化 Job Worker；具体 Provider 通过 handler 注册。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Event, Thread
from typing import Any

from app.generation.security import redact_sensitive_text
from app.jobs.repository import JobRecord, JobStatus, PersistentJobRepository


class JobCancelled(RuntimeError):
    """Handler 在安全检查点响应 cooperative cancellation。"""


@dataclass(frozen=True, slots=True)
class JobExecutionContext:
    repository: PersistentJobRepository
    job_id: str
    worker_id: str

    def checkpoint(self, value: Mapping[str, Any]) -> JobRecord:
        self.raise_if_cancelled()
        return self.repository.heartbeat(
            self.job_id, worker_id=self.worker_id, checkpoint=value
        )

    def cancellation_requested(self) -> bool:
        return self.repository.cancellation_requested(self.job_id)

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested():
            raise JobCancelled("Job cancellation requested.")


JobHandler = Callable[[JobRecord, JobExecutionContext], str | None]


class PersistentJobWorker:
    def __init__(
        self,
        repository: PersistentJobRepository,
        handlers: Mapping[str, JobHandler],
        *,
        worker_id: str,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval 必须为正数。")
        self.repository = repository
        self.handlers = dict(handlers)
        self.worker_id = worker_id
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    def recover_after_restart(
        self, *, stale_after_seconds: float = 30.0
    ) -> tuple[JobRecord, ...]:
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds 不能为负数。")
        threshold = datetime.now(timezone.utc) - timedelta(
            seconds=stale_after_seconds
        )
        marked = self.repository.mark_interrupted(heartbeat_before=threshold.isoformat())
        recovered = self.repository.recover_interrupted()
        # Return every transition caused by this recovery pass, including a
        # stale CANCEL_REQUESTED -> CANCELLED transition.  The Scholar Run
        # reconciler needs the final Job state to close its paired Run.
        by_id = {job.job_id: job for job in (*marked, *recovered)}
        return tuple(by_id.values())

    def _heartbeat_loop(self, job_id: str, stop: Event) -> None:
        while not stop.wait(self.heartbeat_interval_seconds):
            try:
                checkpoint = self.repository.get(job_id).checkpoint
                self.repository.heartbeat(
                    job_id, worker_id=self.worker_id, checkpoint=checkpoint
                )
            except (sqlite3.OperationalError, OSError):
                # SQLite may briefly reject a concurrent writer while the API,
                # event store, or recovery path is committing. A transient
                # storage error must not permanently kill the heartbeat
                # daemon; the next interval will retry with the same durable
                # checkpoint.
                continue
            except (KeyError, RuntimeError):
                # KeyError means the job disappeared. RuntimeError means the
                # job was cancelled, recovered, or claimed by another Worker.
                # In either case this Worker must stop heartbeating rather than
                # overwrite the new owner or terminal state.
                return

    def _still_owned(self, job_id: str) -> JobRecord | None:
        current = self.repository.get(job_id)
        if current.status != "RUNNING" or current.worker_id != self.worker_id:
            return None
        return current

    def run_once(self) -> JobRecord | None:
        job = self.repository.claim_next(
            self.worker_id, job_types=tuple(sorted(self.handlers))
        )
        if job is None:
            return None
        handler = self.handlers.get(job.job_type)
        if handler is None:
            return self.repository.set_status(
                job.job_id,
                "FAILED",
                error_type="MissingJobHandler",
                error_summary=f"未注册 Job handler：{job.job_type}",
            )
        stop = Event()
        heartbeat = Thread(
            target=self._heartbeat_loop,
            args=(job.job_id, stop),
            name=f"heartbeat-{job.job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            result_reference = handler(
                job,
                JobExecutionContext(self.repository, job.job_id, self.worker_id),
            )
        except JobCancelled as error:
            current = self.repository.get(job.job_id)
            if current.status == "CANCELLED":
                return current
            # request_cancel intentionally changes a live Job to
            # CANCEL_REQUESTED before the handler reaches its cooperative
            # checkpoint. That is still this attempt's valid cancellation
            # transition and must be finalized as CANCELLED.
            if current.worker_id != self.worker_id or current.status not in {
                "RUNNING",
                "CANCEL_REQUESTED",
            }:
                return current
            return self.repository.set_status(
                job.job_id,
                "CANCELLED",
                error_type=type(error).__name__,
                error_summary=str(error),
            )
        except Exception as error:
            summary = redact_sensitive_text(f"{type(error).__name__}: {error}")[:1000]
            current = self._still_owned(job.job_id)
            if current is None:
                # Another Worker/recovery process already owns the lifecycle
                # transition. Do not clobber its state with this old attempt's
                # RETRY_PENDING/FAILED result.
                return self.repository.get(job.job_id)
            status: JobStatus = (
                "RETRY_PENDING"
                if current.attempt < current.max_attempts
                else "FAILED"
            )
            return self.repository.set_status(
                job.job_id,
                status,
                error_type=type(error).__name__,
                error_summary=summary,
            )
        finally:
            stop.set()
            heartbeat.join(timeout=max(0.1, self.heartbeat_interval_seconds * 2))
        if self._still_owned(job.job_id) is None:
            return self.repository.get(job.job_id)
        return self.repository.set_status(
            job.job_id, "SUCCEEDED", result_reference=result_reference
        )

    def run_until_idle(self, *, max_jobs: int = 100) -> tuple[JobRecord, ...]:
        if max_jobs < 1:
            raise ValueError("max_jobs 必须为正数。")
        results: list[JobRecord] = []
        for _ in range(max_jobs):
            result = self.run_once()
            if result is None:
                break
            results.append(result)
        return tuple(results)
