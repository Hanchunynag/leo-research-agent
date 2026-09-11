"""Session 生命周期、路径解析和 Session-level lock。"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Iterator

from filelock import FileLock

from app.session.models import SessionRecord, SessionStatus
from app.session.runtime import SessionRuntime


_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate(session_id: str) -> str:
    value = session_id.strip()
    if not _SESSION_ID.fullmatch(value):
        raise ValueError("session_id 必须为 1-64 位安全标识符。")
    return value


class SessionManager:
    """所有 Session 路径和生命周期操作的唯一入口。"""

    def __init__(self, project_root: Path, *, runtime_root: Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.runtime_root = (
            runtime_root.expanduser().resolve()
            if runtime_root is not None
            else self.project_root / "data" / "runtime"
        )
        self.sessions_root = self.runtime_root / "sessions"
        self.catalog_path = self.runtime_root / "catalog.db"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.worker_id = f"worker_{os.getpid()}_{secrets.token_hex(4)}"
        self._locks: dict[str, RLock] = {}
        self._registry_lock = RLock()
        self._initialize_catalog()

    def _connect_catalog(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.catalog_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_catalog(self) -> None:
        with self._connect_catalog() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    active_run_id TEXT,
                    schema_version INTEGER NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
            if "project_id" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_catalog_status_updated "
                "ON sessions(status, updated_at)"
            )

    def create(
        self,
        title: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> SessionRecord:
        requested = _validate(session_id) if session_id else None
        value = requested or f"session_{secrets.token_hex(6)}"
        timestamp = _now()
        relative_path = str(Path("sessions") / value)
        session_dir = self.sessions_root / value
        session_dir.mkdir(parents=True, exist_ok=False)
        record = SessionRecord(
            session_id=value,
            title=title.strip() or "Scholar Research Session",
            status="ACTIVE",
            created_at=timestamp,
            updated_at=timestamp,
            relative_path=relative_path,
            project_id=project_id.strip() if project_id and project_id.strip() else None,
            schema_version=2,
        )
        try:
            with self._connect_catalog() as connection:
                connection.execute(
                    """
                    INSERT INTO sessions
                        (session_id, title, status, created_at, updated_at,
                         relative_path, active_run_id, project_id, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        record.session_id,
                        record.title,
                        record.status,
                        record.created_at,
                        record.updated_at,
                        record.relative_path,
                        record.project_id,
                        record.schema_version,
                    ),
                )
        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        SessionRuntime(session_dir, record)
        return record

    def get(self, session_id: str) -> SessionRecord:
        value = _validate(session_id)
        with self._connect_catalog() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id=?", (value,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Session 不存在：{value}")
        return SessionRecord(
            session_id=str(row["session_id"]),
            title=str(row["title"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            relative_path=str(row["relative_path"]),
            active_run_id=row["active_run_id"],
            project_id=row["project_id"] if "project_id" in row.keys() else None,
            schema_version=int(row["schema_version"]),
        )

    def resolve(
        self,
        session_id: str | None,
        *,
        title: str,
        project_id: str | None = None,
    ) -> SessionRecord:
        if session_id is None:
            return self.create(title, project_id=project_id)
        try:
            record = self.get(session_id)
        except KeyError:
            try:
                return self.create(title, session_id=session_id, project_id=project_id)
            except FileExistsError:
                # Another worker may have created the requested Session between
                # the read and the create. Re-read instead of creating a second
                # source of truth.
                return self.get(session_id)
        if record.status == "DELETED":
            raise ValueError(f"Session 已删除：{record.session_id}")
        if project_id and record.project_id and record.project_id != project_id:
            raise ValueError(
                f"Session 属于其他 Project：{record.session_id} / {record.project_id}"
            )
        if project_id and record.project_id is None:
            runtime = self.open(record.session_id)
            runtime.attach_project(project_id)
            with self._connect_catalog() as connection:
                connection.execute(
                    "UPDATE sessions SET project_id=?, updated_at=? WHERE session_id=?",
                    (project_id, _now(), record.session_id),
                )
            record = self.get(record.session_id)
        return record

    def open(self, session_id: str) -> SessionRuntime:
        record = self.get(session_id)
        if record.status == "DELETED":
            raise ValueError(f"Session 已删除：{record.session_id}")
        return SessionRuntime(self.sessions_root / record.session_id, record)

    def list(self, *, include_deleted: bool = False) -> list[SessionRecord]:
        query = "SELECT * FROM sessions"
        params: tuple[object, ...] = ()
        if not include_deleted:
            query += " WHERE status != ?"
            params = ("DELETED",)
        query += " ORDER BY updated_at DESC, session_id"
        with self._connect_catalog() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._record(row) for row in rows]

    def set_active_run(self, session_id: str, run_id: str | None) -> None:
        value = _validate(session_id)
        with self._connect_catalog() as connection:
            updated = connection.execute(
                "UPDATE sessions SET active_run_id=?, updated_at=? WHERE session_id=?",
                (run_id, _now(), value),
            ).rowcount
        if updated != 1:
            raise KeyError(f"Session 不存在：{value}")

    @staticmethod
    def _record(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=str(row["session_id"]),
            title=str(row["title"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            relative_path=str(row["relative_path"]),
            active_run_id=row["active_run_id"],
            project_id=row["project_id"] if "project_id" in row.keys() else None,
            schema_version=int(row["schema_version"]),
        )

    def _set_status(self, session_id: str, status: SessionStatus) -> SessionRecord:
        value = _validate(session_id)
        current = self.get(value)
        if current.status == "DELETED" and status != "DELETED":
            raise ValueError(f"已删除 Session 不能转为 {status}：{value}")
        if status in {"ARCHIVED", "DELETED"}:
            runtime = self.open(value)
            active = {
                run.status
                for run in runtime.list_runs()
                if run.status in {"PENDING", "RUNNING", "WAITING_USER"}
            }
            if active:
                raise ValueError(f"Session 存在活动 Run，不能变更为 {status}：{value}")
        with self._connect_catalog() as connection:
            updated = connection.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE session_id=?",
                (status, _now(), value),
            ).rowcount
        if updated != 1:
            raise KeyError(f"Session 不存在：{value}")
        runtime = SessionRuntime(
            self.sessions_root / value,
            SessionRecord(
                session_id=current.session_id,
                title=current.title,
                status=status,
                created_at=current.created_at,
                updated_at=_now(),
                relative_path=current.relative_path,
                active_run_id=current.active_run_id,
                project_id=current.project_id,
                schema_version=max(2, current.schema_version),
            ),
        )
        with runtime._connect() as connection:
            connection.execute(
                "UPDATE session_info SET status=?, updated_at=? WHERE session_id=?",
                (status, _now(), value),
            )
        return self.get(value)

    def archive(self, session_id: str) -> SessionRecord:
        return self._set_status(session_id, "ARCHIVED")

    def delete(self, session_id: str, *, hard: bool = False) -> SessionRecord | None:
        value = _validate(session_id)
        with self.lock(value):
            self.get(value)
            if not hard:
                return self._set_status(value, "DELETED")
            active = {
                run.status
                for run in self.open(value).list_runs()
                if run.status in {"PENDING", "RUNNING", "WAITING_USER"}
            }
            if active:
                raise ValueError(f"Session 存在活动 Run，不能 Hard Delete：{value}")
            with self._connect_catalog() as connection:
                connection.execute("DELETE FROM sessions WHERE session_id=?", (value,))
            shutil.rmtree(self.sessions_root / value, ignore_errors=False)
            return None

    @contextmanager
    def lock(self, session_id: str, *, timeout: float = 30.0) -> Iterator[None]:
        value = _validate(session_id)
        with self._registry_lock:
            lock = self._locks.setdefault(value, RLock())
        with lock:
            session_dir = self.sessions_root / value
            session_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(session_dir / ".session.lock"), timeout=timeout):
                yield

    def recover_orphans(self) -> int:
        recovered = 0
        for record in self.list(include_deleted=False):
            runtime = self.open(record.session_id)
            recovered += runtime.mark_orphans_interrupted(
                current_worker_id=self.worker_id
            )
        return recovered
