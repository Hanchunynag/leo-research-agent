"""独立 SQLite 持久化任务表，不修改现有业务数据库 Schema。"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Mapping


JobStatus = Literal[
    "QUEUED",
    "RUNNING",
    "RETRY_PENDING",
    "INTERRUPTED",
    "SUCCEEDED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
]

_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
_FORBIDDEN_PAYLOAD_KEYS = {
    "api_key",
    "secret",
    "token",
    "prompt",
    "pdf",
    "pdf_bytes",
    "model_output",
    "full_text",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Mapping[str, Any] | None) -> str:
    payload = dict(value or {})
    forbidden = {
        str(key)
        for key in payload
        if str(key).casefold() in _FORBIDDEN_PAYLOAD_KEYS
    }
    if forbidden:
        raise ValueError(f"Job payload 包含禁止持久化字段：{sorted(forbidden)}")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode("utf-8")) > 65_536:
        raise ValueError("Job payload/checkpoint 超过 64 KiB。")
    return encoded


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    job_type: str
    workspace_id: str
    scope_version: int
    document_id: str | None
    generation_id: str | None
    payload: Mapping[str, Any]
    status: JobStatus
    attempt: int
    max_attempts: int
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None
    error_type: str | None
    error_summary: str | None
    checkpoint: Mapping[str, Any]
    result_reference: str | None
    idempotency_key: str
    worker_id: str | None


class PersistentJobRepository:
    def __init__(self, project_root: Path, *, path: Path | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.path = path or root / "data" / "jobs" / "jobs.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    scope_version INTEGER NOT NULL,
                    document_id TEXT,
                    generation_id TEXT,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    finished_at TEXT,
                    error_type TEXT,
                    error_summary TEXT,
                    checkpoint_json TEXT NOT NULL,
                    result_reference TEXT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    worker_id TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)"
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),
            workspace_id=str(row["workspace_id"]),
            scope_version=int(row["scope_version"]),
            document_id=row["document_id"],
            generation_id=row["generation_id"],
            payload=json.loads(row["payload_json"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            attempt=int(row["attempt"]),
            max_attempts=int(row["max_attempts"]),
            created_at=str(row["created_at"]),
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
            finished_at=row["finished_at"],
            error_type=row["error_type"],
            error_summary=row["error_summary"],
            checkpoint=json.loads(row["checkpoint_json"]),
            result_reference=row["result_reference"],
            idempotency_key=str(row["idempotency_key"]),
            worker_id=row["worker_id"],
        )

    def submit(
        self,
        job_type: str,
        *,
        workspace_id: str,
        scope_version: int,
        payload: Mapping[str, Any] | None = None,
        document_id: str | None = None,
        generation_id: str | None = None,
        idempotency_key: str,
        max_attempts: int = 3,
    ) -> tuple[JobRecord, bool]:
        if not job_type or not workspace_id or not idempotency_key:
            raise ValueError("Job type/workspace/idempotency_key 不能为空。")
        if scope_version < 1 or max_attempts < 1:
            raise ValueError("scope_version/max_attempts 必须为正数。")
        payload_json = _json(payload)
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                return self._record(existing), False
            job_id = f"J_{secrets.token_hex(8)}"
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, workspace_id, scope_version,
                    document_id, generation_id, payload_json, status,
                    attempt, max_attempts, created_at, checkpoint_json,
                    idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', 0, ?, ?, '{}', ?)
                """,
                (
                    job_id,
                    job_type,
                    workspace_id,
                    scope_version,
                    document_id,
                    generation_id,
                    payload_json,
                    max_attempts,
                    _now(),
                    idempotency_key,
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            assert row is not None
            return self._record(row), True

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Job 不存在：{job_id}")
        return self._record(row)

    def list(self, *, status: JobStatus | None = None) -> tuple[JobRecord, ...]:
        query = "SELECT * FROM jobs"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at, job_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._record(row) for row in rows)

    def claim_next(self, worker_id: str) -> JobRecord | None:
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('QUEUED', 'RETRY_PENDING')
                  AND attempt < max_attempts
                ORDER BY created_at, job_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status='RUNNING', attempt=attempt+1, worker_id=?,
                    started_at=COALESCE(started_at, ?), heartbeat_at=?,
                    finished_at=NULL, error_type=NULL, error_summary=NULL
                WHERE job_id=? AND status IN ('QUEUED', 'RETRY_PENDING')
                """,
                (worker_id, now, now, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            return self._record(claimed)

    def heartbeat(
        self, job_id: str, *, worker_id: str, checkpoint: Mapping[str, Any] | None = None
    ) -> JobRecord:
        checkpoint_json = _json(checkpoint)
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET heartbeat_at=?, checkpoint_json=?
                WHERE job_id=? AND status='RUNNING' AND worker_id=?
                """,
                (_now(), checkpoint_json, job_id, worker_id),
            ).rowcount
        if changed != 1:
            raise RuntimeError("Job Heartbeat 被拒绝：状态或 worker 不匹配。")
        return self.get(job_id)

    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_type: str | None = None,
        error_summary: str | None = None,
        result_reference: str | None = None,
    ) -> JobRecord:
        finished_at = _now() if status in _TERMINAL else None
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET status=?, finished_at=?, error_type=?,
                    error_summary=?, result_reference=? WHERE job_id=?
                """,
                (
                    status,
                    finished_at,
                    error_type,
                    error_summary,
                    result_reference,
                    job_id,
                ),
            ).rowcount
        if changed != 1:
            raise KeyError(f"Job 不存在：{job_id}")
        return self.get(job_id)

    def mark_interrupted(self, *, heartbeat_before: str) -> tuple[JobRecord, ...]:
        """进程启动时把失联 RUNNING 任务显式标记为 INTERRUPTED。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE status='RUNNING'
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                ORDER BY created_at, job_id
                """,
                (heartbeat_before,),
            ).fetchall()
            ids = tuple(str(row["job_id"]) for row in rows)
            if ids:
                connection.executemany(
                    """
                    UPDATE jobs SET status='INTERRUPTED', worker_id=NULL,
                        error_type='ProcessRestart',
                        error_summary='Worker heartbeat lost after process restart.'
                    WHERE job_id=? AND status='RUNNING'
                    """,
                    ((job_id,) for job_id in ids),
                )
        return tuple(self.get(job_id) for job_id in ids)

    def recover_interrupted(self) -> tuple[JobRecord, ...]:
        """按重试上限把 INTERRUPTED 转入 RETRY_PENDING 或 FAILED。"""

        finished = _now()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id, attempt, max_attempts FROM jobs WHERE status='INTERRUPTED'"
            ).fetchall()
            for row in rows:
                retry = int(row["attempt"]) < int(row["max_attempts"])
                connection.execute(
                    """
                    UPDATE jobs SET status=?, finished_at=?, worker_id=NULL
                    WHERE job_id=? AND status='INTERRUPTED'
                    """,
                    (
                        "RETRY_PENDING" if retry else "FAILED",
                        None if retry else finished,
                        row["job_id"],
                    ),
                )
        return tuple(self.get(str(row["job_id"])) for row in rows)
