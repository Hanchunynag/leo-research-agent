"""Session Runtime 的稳定数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SessionStatus = Literal["ACTIVE", "ARCHIVED", "DELETED"]
RunStatus = Literal[
    "PENDING",
    "QUEUED",
    "RUNNING",
    "WAITING_HUMAN_APPROVAL",
    "WAITING_USER",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    title: str
    status: SessionStatus
    created_at: str
    updated_at: str
    relative_path: str
    active_run_id: str | None = None
    project_id: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    session_id: str
    query: str
    status: RunStatus
    thread_id: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    worker_id: str | None = None
    trace_id: str | None = None
    job_id: str | None = None
    project_id: str | None = None
    checkpoint_ref: str | None = None
    failure_message: str | None = None
