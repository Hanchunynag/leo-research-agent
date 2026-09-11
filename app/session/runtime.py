"""一个 Session 私有 SQLite DB 的业务存储。

LangGraph Working State 不在这里保存；这里仅保存 Conversation、Run 元数据
和稳定结果 Projection。Checkpoint 由未来兼容的官方 Checkpointer Adapter 管理。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from app.session.models import RunRecord, RunStatus, SessionRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SessionRuntime:
    """Session 私有 DB 的唯一业务访问入口。"""

    def __init__(self, session_dir: Path, session: SessionRecord) -> None:
        self.session_dir = session_dir.expanduser().resolve()
        self.session = session
        self.database_path = self.session_dir / "session.db"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_info (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    thread_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    worker_id TEXT,
                    trace_id TEXT,
                    checkpoint_ref TEXT,
                    failure_message TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_messages_created
                    ON messages(created_at, message_id);
                CREATE INDEX IF NOT EXISTS ix_runs_status
                    ON agent_runs(status, created_at);
                CREATE TABLE IF NOT EXISTS research_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
                );
                """
            )
            connection.execute(
                """
                INSERT INTO session_info
                    (session_id, title, status, created_at, updated_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    schema_version=excluded.schema_version
                """,
                (
                    self.session.session_id,
                    self.session.title,
                    self.session.status,
                    self.session.created_at,
                    self.session.updated_at,
                    self.session.schema_version,
                ),
            )

    def create_run(self, query: str, *, run_id: str, thread_id: str) -> RunRecord:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Run query 不能为空。")
        created = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs
                    (run_id, query, status, thread_id, created_at)
                VALUES (?, ?, 'PENDING', ?, ?)
                """,
                (run_id, cleaned, thread_id, created),
            )
        return self.get_run(run_id)

    def start_run(self, run_id: str, *, worker_id: str) -> RunRecord:
        started = _now()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE agent_runs
                SET status='RUNNING', started_at=?, worker_id=?
                WHERE run_id=? AND status='PENDING'
                """,
                (started, worker_id, run_id),
            ).rowcount
        if updated != 1:
            raise ValueError(f"Run 不能进入 RUNNING：{run_id}")
        return self.get_run(run_id)

    def append_message(
        self,
        role: str,
        content: str,
        *,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if not role.strip() or not content.strip():
            raise ValueError("Message role/content 不能为空。")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(run_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, role.strip(), content, _now(), _json(dict(metadata or {}))),
            )
            return int(cursor.lastrowid)

    def recent_messages(self, limit: int = 8) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("recent message limit 必须大于 0。")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, run_id, role, content, created_at, metadata_json
                FROM messages ORDER BY message_id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        values = []
        for row in reversed(rows):
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            values.append(item)
        return values

    def complete_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        answer: str,
        citations: list[Mapping[str, Any]],
        evidence: list[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> RunRecord:
        if status not in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED", "WAITING_USER"}:
            raise ValueError(f"非法最终 Run 状态：{status}")
        finished = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status=?, completed_at=?
                WHERE run_id=?
                """,
                (status, finished, run_id),
            )
            connection.execute(
                """
                INSERT INTO research_results
                    (run_id, status, answer, citations_json, evidence_json,
                     metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    answer=excluded.answer,
                    citations_json=excluded.citations_json,
                    evidence_json=excluded.evidence_json,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                (
                    run_id,
                    status,
                    answer,
                    _json(citations),
                    _json(evidence),
                    _json(dict(metadata)),
                    finished,
                ),
            )
        return self.get_run(run_id)

    def fail_run(self, run_id: str, message: str) -> RunRecord:
        return self.complete_run(
            run_id,
            status="FAILED",
            answer="",
            citations=[],
            evidence=[],
            metadata={"error": message},
        )

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Run 不存在：{run_id}")
        value = dict(row)
        return RunRecord(session_id=self.session.session_id, **value)

    def list_runs(self) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_runs ORDER BY created_at, run_id"
            ).fetchall()
        return [RunRecord(session_id=self.session.session_id, **dict(row)) for row in rows]

    def mark_orphans_interrupted(self, *, current_worker_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """
                    UPDATE agent_runs
                    SET status='INTERRUPTED', completed_at=?
                    WHERE status='RUNNING' AND worker_id != ?
                    """,
                    (_now(), current_worker_id),
                ).rowcount
            )
