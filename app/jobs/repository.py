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
    lease_token: str | None = None
    available_at: str | None = None
    tenant_id: str = "local"
    principal_id: str = "local"


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
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
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
                    worker_id TEXT,
                    lease_token TEXT,
                    available_at TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    principal_id TEXT NOT NULL DEFAULT 'local'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)"
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)")
            }
            for column in ("lease_token", "available_at", "tenant_id", "principal_id"):
                if column not in columns:
                    default = " NOT NULL DEFAULT 'local'" if column in {"tenant_id", "principal_id"} else ""
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {column} TEXT{default}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_actions (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    worker_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
                """
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
            lease_token=row["lease_token"] if "lease_token" in row.keys() else None,
            available_at=row["available_at"] if "available_at" in row.keys() else None,
            tenant_id=str(row["tenant_id"] or "local") if "tenant_id" in row.keys() else "local",
            principal_id=str(row["principal_id"] or "local") if "principal_id" in row.keys() else "local",
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
        tenant_id: str = "local",
        principal_id: str = "local",
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
                    idempotency_key, tenant_id, principal_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', 0, ?, ?, '{}', ?, ?, ?)
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
                    tenant_id,
                    principal_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            assert row is not None
            self._append_action_connection(
                connection,
                job_id,
                status="PLANNED",
                details={"job_type": job_type},
            )
            return self._record(row), True

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Job 不存在：{job_id}")
        return self._record(row)

    def get_by_idempotency(self, idempotency_key: str) -> JobRecord | None:
        """Return an existing durable Job for an idempotency key, if any."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return self._record(row) if row is not None else None

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

    def list_dead_letters(
        self,
        *,
        tenant_id: str | None = None,
        principal_id: str | None = None,
    ) -> tuple[JobRecord, ...]:
        """List terminal Jobs that exhausted their retry budget."""

        query = (
            "SELECT * FROM jobs WHERE status='FAILED' AND attempt >= max_attempts"
        )
        parameters: list[Any] = []
        if tenant_id is not None:
            query += " AND tenant_id=?"
            parameters.append(tenant_id)
        if principal_id is not None:
            query += " AND principal_id=?"
            parameters.append(principal_id)
        query += " ORDER BY finished_at, job_id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._record(row) for row in rows)

    def requeue_failed(
        self,
        job_id: str,
        *,
        tenant_id: str | None = None,
        principal_id: str | None = None,
        reset_attempts: bool = False,
    ) -> JobRecord:
        """Manually replay one dead-letter Job after operator review."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Job 不存在：{job_id}")
            if row["status"] != "FAILED" or int(row["attempt"]) < int(row["max_attempts"]):
                raise ValueError("只有已耗尽重试次数的 FAILED Job 才能进入 DLQ 重放。")
            if tenant_id is not None and str(row["tenant_id"]) != tenant_id:
                raise PermissionError("Job 不属于当前租户。")
            if principal_id is not None and str(row["principal_id"]) != principal_id:
                raise PermissionError("Job 不属于当前主体。")
            attempt = 0 if reset_attempts else int(row["attempt"])
            connection.execute(
                """
                UPDATE jobs SET status='RETRY_PENDING', attempt=?, finished_at=NULL,
                    error_type=NULL, error_summary=NULL, available_at=NULL
                WHERE job_id=? AND status='FAILED'
                """,
                (attempt, job_id),
            )
            self._append_action_connection(
                connection,
                job_id,
                status="REQUEUED",
                details={"reset_attempts": reset_attempts},
            )
        return self.get(job_id)

    def claim_next(
        self,
        worker_id: str,
        *,
        job_types: tuple[str, ...] | None = None,
        max_running_per_scope: int | None = None,
    ) -> JobRecord | None:
        if max_running_per_scope is not None and max_running_per_scope < 1:
            raise ValueError("max_running_per_scope 必须为正数。")
        now = _now()
        lease_token = secrets.token_urlsafe(24)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            type_clause = ""
            parameters: list[Any] = []
            if job_types:
                type_clause = " AND job_type IN (" + ",".join("?" for _ in job_types) + ")"
                parameters.extend(job_types)
            scope_clause = ""
            if max_running_per_scope is not None:
                scope_clause = (
                    " AND (SELECT COUNT(*) FROM jobs running "
                    "WHERE running.status='RUNNING' "
                    "AND running.tenant_id=jobs.tenant_id "
                    "AND running.principal_id=jobs.principal_id) < ?"
                )
                parameters.append(max_running_per_scope)
            parameters.append(now)
            row = connection.execute(
                f"""
                SELECT * FROM jobs
                WHERE status IN ('QUEUED', 'RETRY_PENDING')
                  AND attempt < max_attempts
                  {type_clause}
                  {scope_clause}
                  AND (available_at IS NULL OR available_at <= ?)
                ORDER BY (
                    SELECT COUNT(*) FROM jobs running
                    WHERE running.status='RUNNING'
                      AND running.tenant_id=jobs.tenant_id
                      AND running.principal_id=jobs.principal_id
                ), created_at, job_id LIMIT 1
                """,  # noqa: S608 - placeholders are used for every job_type value.
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status='RUNNING', attempt=attempt+1, worker_id=?,
                    lease_token=?, started_at=COALESCE(started_at, ?), heartbeat_at=?,
                    finished_at=NULL, error_type=NULL, error_summary=NULL,
                    available_at=NULL
                WHERE job_id=? AND status IN ('QUEUED', 'RETRY_PENDING')
                """,
                (worker_id, lease_token, now, now, row["job_id"]),
            )
            self._append_action_connection(
                connection,
                str(row["job_id"]),
                status="STARTED",
                worker_id=worker_id,
                details={"fenced": True},
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            return self._record(claimed)

    def claim(self, job_id: str, worker_id: str) -> JobRecord:
        now = _now()
        lease_token = secrets.token_urlsafe(24)
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE jobs
                SET status='RUNNING', attempt=attempt+1, worker_id=?,
                    lease_token=?, started_at=COALESCE(started_at, ?), heartbeat_at=?,
                    available_at=NULL
                WHERE job_id=? AND status IN ('QUEUED', 'RETRY_PENDING')
                  AND attempt < max_attempts
                  AND (available_at IS NULL OR available_at <= ?)
                """,
                (worker_id, lease_token, now, now, job_id, now),
            ).rowcount
            if changed == 1:
                self._append_action_connection(
                    connection,
                    job_id,
                    status="STARTED",
                    worker_id=worker_id,
                    details={"fenced": True},
                )
        if changed != 1:
            raise RuntimeError("Job 无法由指定 Worker claim。")
        return self.get(job_id)

    def append_event(self, job_id: str, event: Mapping[str, Any]) -> int:
        encoded = _json(event)
        with self._lock, self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone() is None:
                raise KeyError(f"Job 不存在：{job_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS value FROM job_events WHERE job_id=?",
                (job_id,),
            ).fetchone()
            sequence = int(row["value"]) + 1
            connection.execute(
                "INSERT INTO job_events(job_id, sequence, event_json) VALUES (?, ?, ?)",
                (job_id, sequence, encoded),
            )
        return sequence

    @staticmethod
    def _append_action_connection(
        connection: sqlite3.Connection,
        job_id: str,
        *,
        status: str,
        worker_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS value FROM job_actions WHERE job_id=?",
            (job_id,),
        ).fetchone()
        sequence = int(row["value"]) + 1
        connection.execute(
            """
            INSERT INTO job_actions(
                job_id, sequence, action, status, timestamp, worker_id, details_json
            ) VALUES (?, ?, 'job', ?, ?, ?, ?)
            """,
            (
                job_id,
                sequence,
                status,
                _now(),
                worker_id,
                _json(details),
            ),
        )
        return sequence

    def list_actions(self, job_id: str) -> tuple[Mapping[str, Any], ...]:
        """Return the append-only lifecycle journal for one Job."""

        self.get(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, action, status, timestamp, worker_id, details_json
                FROM job_actions WHERE job_id=? ORDER BY sequence
                """,
                (job_id,),
            ).fetchall()
        return tuple(
            {
                "sequence": int(row["sequence"]),
                "action": str(row["action"]),
                "status": str(row["status"]),
                "timestamp": str(row["timestamp"]),
                "worker_id": row["worker_id"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        )

    def list_events(self, job_id: str) -> tuple[Mapping[str, Any], ...]:
        self.get(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_json FROM job_events WHERE job_id=? ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return tuple(
            {"sequence": int(row["sequence"]), **json.loads(row["event_json"])}
            for row in rows
        )

    def heartbeat(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str | None = None,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        checkpoint_json = _json(checkpoint)
        ownership = " AND lease_token=?" if lease_token is not None else ""
        parameters: tuple[Any, ...] = (
            _now(),
            checkpoint_json,
            job_id,
            worker_id,
            *((lease_token,) if lease_token is not None else ()),
        )
        with self._connect() as connection:
            changed = connection.execute(
                f"""
                UPDATE jobs SET heartbeat_at=?, checkpoint_json=?
                WHERE job_id=? AND status='RUNNING' AND worker_id=?{ownership}
                """,
                parameters,
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
        available_at: str | None = None,
        worker_id: str | None = None,
        lease_token: str | None = None,
    ) -> JobRecord:
        finished_at = _now() if status in _TERMINAL else None
        ownership = ""
        parameters: list[Any] = [
            status,
            finished_at,
            error_type,
            error_summary,
            result_reference,
            available_at,
            job_id,
        ]
        if worker_id is not None:
            if lease_token is None:
                raise ValueError("带 worker_id 更新 Job 时必须提供 lease_token。")
            ownership = " AND worker_id=? AND lease_token=?"
            parameters.extend([worker_id, lease_token])
        with self._connect() as connection:
            changed = connection.execute(
                f"""
                UPDATE jobs SET status=?, finished_at=?, error_type=?,
                    error_summary=?, result_reference=?, available_at=?,
                    worker_id=CASE WHEN ? IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
                                   THEN NULL ELSE worker_id END,
                    lease_token=CASE WHEN ? IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
                                     THEN NULL ELSE lease_token END
                WHERE job_id=?{ownership}
                """,
                # The status is repeated for the ownership-clearing CASE
                # expressions before the WHERE parameters.
                [status, finished_at, error_type, error_summary, result_reference, available_at, status, status, *parameters[6:]],
            ).rowcount
            if changed == 1:
                self._append_action_connection(
                    connection,
                    job_id,
                    status=status,
                    worker_id=worker_id,
                    details={"error_type": error_type} if error_type else None,
                )
        if changed != 1:
            raise KeyError(f"Job 不存在：{job_id}")
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> JobRecord:
        current = self.get(job_id)
        if current.status in _TERMINAL:
            return current
        target: JobStatus = (
            "CANCELLED"
            if current.status in {"QUEUED", "RETRY_PENDING", "INTERRUPTED"}
            else "CANCEL_REQUESTED"
        )
        return self.set_status(
            job_id,
            target,
            error_type="CancelledByUser" if target == "CANCELLED" else None,
            error_summary="Job cancellation requested.",
        )

    def cancellation_requested(self, job_id: str) -> bool:
        return self.get(job_id).status in {"CANCEL_REQUESTED", "CANCELLED"}

    def mark_interrupted(
        self,
        *,
        heartbeat_before: str,
        job_type_prefix: str | None = None,
    ) -> tuple[JobRecord, ...]:
        """收束进程启动时已经失联的任务。

        A cancellation request is durable state, so a dead Worker must not
        leave it in ``CANCEL_REQUESTED`` forever.  Only requests whose last
        heartbeat is older than ``heartbeat_before`` are finalized; a live
        Worker still gets the cooperative-cancellation grace period.
        """

        type_filter = ""
        parameters: list[Any] = [heartbeat_before]
        if job_type_prefix is not None:
            type_filter = " AND job_type LIKE ?"
            parameters.append(f"{job_type_prefix}%")
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT job_id, status FROM jobs
                WHERE status IN ('RUNNING', 'CANCEL_REQUESTED')
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                  {type_filter}
                ORDER BY created_at, job_id
                """,
                tuple(parameters),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE jobs SET
                        status=CASE WHEN status='CANCEL_REQUESTED'
                                    THEN 'CANCELLED' ELSE 'INTERRUPTED' END,
                        finished_at=CASE WHEN status='CANCEL_REQUESTED'
                                         THEN ? ELSE NULL END,
                        worker_id=NULL,
                        lease_token=NULL,
                        available_at=NULL,
                        error_type=CASE WHEN status='CANCEL_REQUESTED'
                                        THEN 'CancelledAfterWorkerLost'
                                        ELSE 'ProcessRestart' END,
                        error_summary=CASE WHEN status='CANCEL_REQUESTED'
                                           THEN 'Cancellation finalized after Worker heartbeat expired.'
                                           ELSE 'Worker heartbeat lost after process restart.' END
                    WHERE job_id=? AND status IN ('RUNNING', 'CANCEL_REQUESTED')
                    """,
                    ((_now(), str(row["job_id"])) for row in rows),
                )
            ids = tuple(str(row["job_id"]) for row in rows)
        return tuple(self.get(job_id) for job_id in ids)

    def recover_interrupted(
        self,
        *,
        job_type_prefix: str | None = None,
    ) -> tuple[JobRecord, ...]:
        """按重试上限把 INTERRUPTED 转入 RETRY_PENDING 或 FAILED。"""

        finished = _now()
        type_filter = ""
        parameters: tuple[Any, ...] = ()
        if job_type_prefix is not None:
            type_filter = " AND job_type LIKE ?"
            parameters = (f"{job_type_prefix}%",)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT job_id, attempt, max_attempts FROM jobs "
                f"WHERE status='INTERRUPTED'{type_filter}",
                parameters,
            ).fetchall()
            for row in rows:
                retry = int(row["attempt"]) < int(row["max_attempts"])
                connection.execute(
                    """
                    UPDATE jobs SET status=?, finished_at=?, worker_id=NULL,
                        lease_token=NULL, available_at=NULL
                    WHERE job_id=? AND status='INTERRUPTED'
                    """,
                    (
                        "RETRY_PENDING" if retry else "FAILED",
                        None if retry else finished,
                        row["job_id"],
                    ),
                )
        return tuple(self.get(str(row["job_id"])) for row in rows)
