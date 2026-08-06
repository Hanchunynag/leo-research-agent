"""Web 兼容门面：持久化 Job 状态、事件和结果引用。"""

from __future__ import annotations

import json
import secrets
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from app.generation.security import redact_sensitive_text
from app.jobs import PersistentJobRepository
from app.storage import write_json_atomic
from app.web.models import JobCreated, JobEvent, JobSnapshot


JobKind = Literal["answer", "parse"]
JobTask = Callable[[Callable[..., None]], dict[str, Any]]


class JobManager:
    """固定线程执行兼容任务；状态和结果可在进程重启后查询。"""

    def __init__(
        self,
        max_workers: int = 2,
        *,
        project_root: Path | None = None,
        repository: PersistentJobRepository | None = None,
    ) -> None:
        if max_workers < 1 or max_workers > 8:
            raise ValueError("max_workers 必须在 1 到 8 之间。")
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if repository is None:
            if project_root is None:
                self._temporary = tempfile.TemporaryDirectory(
                    prefix="leo-job-manager-"
                )
                project_root = Path(self._temporary.name)
            repository = PersistentJobRepository(project_root)
        self.repository = repository
        self.project_root = repository.path.parents[2]
        self.result_directory = repository.path.parent / "web_results"
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="leo-web",
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        self._recover_orphaned_closures()

    def _recover_orphaned_closures(self) -> None:
        """Web closure 无法跨重启重建时明确失败，不伪装为仍在运行。"""

        from datetime import datetime, timedelta, timezone

        future = datetime.now(timezone.utc) + timedelta(seconds=1)
        self.repository.mark_interrupted(heartbeat_before=future.isoformat())
        for value in self.repository.recover_interrupted():
            if value.status == "RETRY_PENDING" and value.job_type.startswith("web."):
                self.repository.set_status(
                    value.job_id,
                    "FAILED",
                    error_type="ProcessRestartRequiresResubmission",
                    error_summary=(
                        "Web compatibility closure cannot be reconstructed after restart; "
                        "resubmit the operation using its idempotency key."
                    ),
                )

    def submit(
        self,
        kind: JobKind,
        task: JobTask,
        *,
        idempotency_key: str | None = None,
        workspace_id: str = "default",
        scope_version: int = 1,
        payload: Mapping[str, Any] | None = None,
    ) -> JobCreated:
        record, created = self.repository.submit(
            f"web.{kind}",
            workspace_id=workspace_id,
            scope_version=scope_version,
            payload=payload,
            idempotency_key=idempotency_key or f"web.{kind}:{secrets.token_hex(16)}",
            max_attempts=1,
        )
        if created:
            self._emit(record.job_id, "queued", "任务已进入队列。", 0.0)
            future = self._executor.submit(self._run, record.job_id, task)
            with self._lock:
                self._futures[record.job_id] = future
        return JobCreated(job_id=record.job_id)

    def _emit(
        self,
        job_id: str,
        stage: str,
        message: str,
        progress: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.repository.cancellation_requested(job_id):
            raise RuntimeError("JobCancelled")
        safe_details = {
            str(key): value
            for key, value in (details or {}).items()
            if not any(
                marker in str(key).casefold()
                for marker in ("key", "token", "prompt", "secret")
            )
        }
        self.repository.append_event(
            job_id,
            {
                "stage": stage,
                "message": redact_sensitive_text(message),
                "progress": max(0.0, min(1.0, progress)),
                "details": safe_details,
            },
        )

    def _run(self, job_id: str, task: JobTask) -> None:
        try:
            self.repository.claim(job_id, f"web-thread:{job_id}")
        except RuntimeError:
            return
        self._emit(job_id, "running", "任务开始执行。", 0.02)

        def emit(
            stage: str,
            message: str,
            progress: float,
            details: dict[str, Any] | None = None,
        ) -> None:
            self.repository.heartbeat(
                job_id,
                worker_id=f"web-thread:{job_id}",
                checkpoint={"stage": stage, "progress": progress},
            )
            self._emit(job_id, stage, message, progress, details)

        try:
            result = task(emit)
            if self.repository.cancellation_requested(job_id):
                self.repository.set_status(
                    job_id,
                    "CANCELLED",
                    error_type="CancelledByUser",
                    error_summary="Job cancelled before result commit.",
                )
                return
            path = self.result_directory / f"{job_id}.json"
            write_json_atomic(path, result)
            reference = path.relative_to(self.project_root).as_posix()
        except Exception as error:
            if self.repository.cancellation_requested(job_id):
                self.repository.set_status(
                    job_id,
                    "CANCELLED",
                    error_type="CancelledByUser",
                    error_summary="Job cancelled at cooperative checkpoint.",
                )
                return
            safe_error = redact_sensitive_text(
                f"{type(error).__name__}: {error}"
            )[:1000]
            self.repository.set_status(
                job_id,
                "FAILED",
                error_type=type(error).__name__,
                error_summary=safe_error,
            )
            self.repository.append_event(
                job_id,
                {
                    "stage": "failed",
                    "message": safe_error,
                    "progress": 1.0,
                    "details": {},
                },
            )
            return
        self.repository.set_status(
            job_id, "SUCCEEDED", result_reference=reference
        )
        self._emit(job_id, "completed", "任务执行完成。", 1.0)

    def cancel(self, job_id: str) -> JobSnapshot:
        record = self.repository.request_cancel(job_id)
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None and future.cancel():
            self.repository.set_status(
                job_id,
                "CANCELLED",
                error_type="CancelledByUser",
                error_summary="Queued task cancelled before execution.",
            )
        elif record.status == "CANCEL_REQUESTED":
            self.repository.append_event(
                job_id,
                {
                    "stage": "cancelling",
                    "message": "任务将在下一个安全检查点取消。",
                    "progress": 0.0,
                    "details": {},
                },
            )
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> JobSnapshot:
        try:
            record = self.repository.get(job_id)
        except KeyError as error:
            raise KeyError(f"任务不存在：{job_id}") from error
        kind = record.job_type.removeprefix("web.")
        events = [JobEvent(**dict(value)) for value in self.repository.list_events(job_id)]
        result: dict[str, Any] | None = None
        if record.result_reference:
            path = (self.project_root / record.result_reference).resolve()
            if self.result_directory.resolve() not in path.parents:
                raise PermissionError("Web Job result_reference 越界。")
            raw = json.loads(path.read_text(encoding="utf-8"))
            result = dict(raw) if isinstance(raw, dict) else None
        statuses = {
            "QUEUED": "queued",
            "RUNNING": "running",
            "CANCEL_REQUESTED": "running",
            "SUCCEEDED": "succeeded",
            "FAILED": "failed",
            "CANCELLED": "cancelled",
            "INTERRUPTED": "failed",
            "RETRY_PENDING": "queued",
        }
        return JobSnapshot(
            job_id=record.job_id,
            kind=kind,  # type: ignore[arg-type]
            status=statuses[record.status],  # type: ignore[arg-type]
            events=events,
            result=result,
            error=record.error_summary,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
