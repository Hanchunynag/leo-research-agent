"""进程内有界后台任务与 SSE 事件快照。"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from app.generation.security import redact_sensitive_text
from app.web.models import JobCreated, JobEvent, JobSnapshot


JobKind = Literal["answer", "parse"]
JobTask = Callable[[Callable[..., None]], dict[str, Any]]


@dataclass
class _Job:
    job_id: str
    kind: JobKind
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    events: list[JobEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """用固定工作线程数执行长任务，不把 Prompt 或密钥写入事件。"""

    def __init__(self, max_workers: int = 2) -> None:
        if max_workers < 1 or max_workers > 8:
            raise ValueError("max_workers 必须在 1 到 8 之间。")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="leo-web",
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = Lock()

    def submit(self, kind: JobKind, task: JobTask) -> JobCreated:
        """提交任务并立即返回不含时间或路径的稳定句柄。"""

        job_id = f"J_{secrets.token_hex(8)}"
        job = _Job(job_id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
        self._emit(job_id, "queued", "任务已进入队列。", 0.0)
        self._executor.submit(self._run, job_id, task)
        return JobCreated(job_id=job_id)

    def _emit(
        self,
        job_id: str,
        stage: str,
        message: str,
        progress: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_message = redact_sensitive_text(message)
        safe_details = {
            str(key): value
            for key, value in (details or {}).items()
            if "key" not in str(key).casefold()
            and "token" not in str(key).casefold()
            and "prompt" not in str(key).casefold()
        }
        with self._lock:
            job = self._jobs[job_id]
            job.events.append(
                JobEvent(
                    sequence=len(job.events) + 1,
                    stage=stage,
                    message=safe_message,
                    progress=max(0.0, min(1.0, progress)),
                    details=safe_details,
                )
            )

    def _run(self, job_id: str, task: JobTask) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"
        self._emit(job_id, "running", "任务开始执行。", 0.02)

        def emit(
            stage: str,
            message: str,
            progress: float,
            details: dict[str, Any] | None = None,
        ) -> None:
            self._emit(job_id, stage, message, progress, details)

        try:
            result = task(emit)
        except Exception as error:
            safe_error = redact_sensitive_text(
                f"{type(error).__name__}: {error}"
            )
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = safe_error
            self._emit(job_id, "failed", safe_error, 1.0)
            return
        with self._lock:
            job = self._jobs[job_id]
            job.status = "succeeded"
            job.result = result
        self._emit(job_id, "completed", "任务执行完成。", 1.0)

    def snapshot(self, job_id: str) -> JobSnapshot:
        """返回任务的不可变快照。"""

        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as error:
                raise KeyError(f"任务不存在：{job_id}") from error
            return JobSnapshot(
                job_id=job.job_id,
                kind=job.kind,
                status=job.status,
                events=[event.model_copy(deep=True) for event in job.events],
                result=dict(job.result) if job.result is not None else None,
                error=job.error,
            )

    def close(self) -> None:
        """停止接收新任务，并等待已提交任务完成。"""

        self._executor.shutdown(wait=True, cancel_futures=False)
