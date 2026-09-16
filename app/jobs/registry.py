"""Durable Worker heartbeat registry for the Scholar control plane."""

from __future__ import annotations

import os
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkerRegistry:
    """Small process-safe registry shared by API and external Workers.

    The registry is observability state only.  It never owns Jobs, Runs, or
    domain transitions; a missing heartbeat is projected as ``STALE`` by the
    reader and does not mutate the lifecycle stores.
    """

    def __init__(self, project_root: Path, *, path: Path | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.path = path or root / "data" / "runtime" / "workers.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    worker_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    hostname TEXT NOT NULL,
                    backend TEXT,
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    stopped_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_workers_heartbeat "
                "ON workers(heartbeat_at, status)"
            )

    def register(
        self,
        worker_id: str,
        *,
        worker_type: str = "scholar",
        backend: str | None = "crewai",
        pid: int | None = None,
        hostname: str | None = None,
    ) -> dict[str, Any]:
        value = worker_id.strip()
        if not value:
            raise ValueError("worker_id 不能为空。")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers(
                    worker_id, worker_type, status, pid, hostname, backend,
                    started_at, heartbeat_at, stopped_at
                ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(worker_id) DO UPDATE SET
                    worker_type=excluded.worker_type,
                    status='RUNNING',
                    pid=excluded.pid,
                    hostname=excluded.hostname,
                    backend=excluded.backend,
                    heartbeat_at=excluded.heartbeat_at,
                    stopped_at=NULL
                """,
                (
                    value,
                    worker_type,
                    os.getpid() if pid is None else pid,
                    hostname or socket.gethostname(),
                    backend,
                    timestamp,
                    timestamp,
                ),
            )
        return self.get(value)

    def heartbeat(self, worker_id: str) -> dict[str, Any]:
        timestamp = _now()
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE workers SET status='RUNNING', heartbeat_at=?, stopped_at=NULL
                WHERE worker_id=?
                """,
                (timestamp, worker_id),
            ).rowcount
        if changed != 1:
            return self.register(worker_id)
        return self.get(worker_id)

    def stop(self, worker_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE workers SET status='STOPPED', stopped_at=?, heartbeat_at=?
                WHERE worker_id=?
                """,
                (_now(), _now(), worker_id),
            )
        try:
            return self.get(worker_id)
        except KeyError:
            return None

    def get(self, worker_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Worker 不存在：{worker_id}")
        return self._row(row)

    def list(self, *, stale_after_seconds: float = 30.0) -> tuple[dict[str, Any], ...]:
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds 不能为负数。")
        threshold = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workers ORDER BY heartbeat_at DESC, worker_id"
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = self._row(row)
            if value["status"] == "RUNNING":
                try:
                    heartbeat = datetime.fromisoformat(str(value["heartbeat_at"]))
                except ValueError:
                    heartbeat = datetime.min.replace(tzinfo=UTC)
                if heartbeat < threshold:
                    value["status"] = "STALE"
            values.append(value)
        return tuple(values)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "worker_id": str(row["worker_id"]),
            "worker_type": str(row["worker_type"]),
            "status": str(row["status"]),
            "pid": row["pid"],
            "hostname": str(row["hostname"]),
            "backend": row["backend"],
            "started_at": str(row["started_at"]),
            "heartbeat_at": str(row["heartbeat_at"]),
            "stopped_at": row["stopped_at"],
        }
