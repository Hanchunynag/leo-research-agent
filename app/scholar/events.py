"""Persistent Scholar RunEvent source of truth.

The Web Console is a projection of this store.  Events are written before a
caller is allowed to publish them, which makes reconnect/replay safe across an
API or Worker restart.  The store deliberately contains bounded trace
metadata, never prompts, chain-of-thought, secrets, or raw model output.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Mapping


_EVENT_TYPES = {
    "RUN_STARTED",
    "RUN_RESUMED",
    "SKILL_SELECTED",
    "CONTEXT_ASSEMBLED",
    "SUBAGENT_STARTED",
    "SUBAGENT_COMPLETED",
    "TOOL_STARTED",
    "TOOL_COMPLETED",
    "RESEARCH_PROGRESS",
    "RESEARCH_COMPLETED",
    "EVIDENCE_VERIFIED",
    "DOMAIN_RESULT",
    "DRAFT_CREATED",
    "REVIEW_STARTED",
    "REVIEW_COMPLETED",
    "PATCH_CREATED",
    "CHECKPOINT_SAVED",
    "WAITING_USER",
    "RUN_INTERRUPTED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
}
_EVENT_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "WAITING_USER",
    "INTERRUPTED",
    "CANCELLED",
}
_FORBIDDEN_KEYS = {
    "prompt",
    "system_prompt",
    "user_prompt",
    "secret",
    "api_key",
    "password",
    "raw_output",
    "chain_of_thought",
}
_USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
}


def _forbidden_key(key: str) -> bool:
    normalized = key.casefold()
    if normalized in _USAGE_KEYS:
        return False
    if normalized in _FORBIDDEN_KEYS:
        return True
    return (
        normalized.startswith(("prompt_", "system_prompt_", "user_prompt_"))
        or normalized.endswith(("_token", "_secret", "_api_key", "_password"))
        or "raw_output" in normalized
        or "chain_of_thought" in normalized
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe(value: Any, *, key: str = "") -> Any:
    if _forbidden_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(child_key): _safe(child, key=str(child_key))
            for child_key, child in value.items()
            if not _forbidden_key(str(child_key))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(child, key=key) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:1000]
    return str(value)[:500]


def _json(value: Any) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _trace_event_type(name: str, kind: str, status: str, metadata: Mapping[str, Any]) -> str:
    normalized = name.casefold()
    state = str(metadata.get("state") or "")
    if normalized in {"run_created", "run_queued"}:
        return "RUN_STARTED"
    if normalized in {"run_resumed", "human_approval_reconciled"}:
        return "RUN_RESUMED"
    if normalized in {"flow_transition", "flowstartedevent", "flowfinishedevent", "methodexecutionstartedevent"}:
        return "RUN_STARTED" if state == "ROUTING" and normalized == "flow_transition" else "DOMAIN_RESULT"
    if normalized in {"crewkickoffstartedevent", "agentexecutionstartedevent", "crew_task_started"}:
        return "SUBAGENT_STARTED"
    if normalized in {"crewkickoffcompletedevent", "agentexecutioncompletedevent", "crew_task_completed"}:
        return "SUBAGENT_COMPLETED"
    if normalized in {"toolusagestartedevent"} or (kind == "tool" and status == "RUNNING"):
        return "TOOL_STARTED"
    if normalized in {"toolusagefinishedevent"} or (kind == "tool" and status in {"COMPLETED", "FAILED"}):
        return "TOOL_COMPLETED"
    if normalized in {"harness_context"}:
        return "CONTEXT_ASSEMBLED"
    if normalized in {"harness_plan"}:
        return "SKILL_SELECTED"
    if normalized in {"research_evidence", "research_completed"}:
        return "RESEARCH_COMPLETED"
    if normalized in {"research_need", "research_stage", "research_coverage_item"}:
        return "RESEARCH_PROGRESS"
    if normalized in {"evidence_validate", "evidence_verified"}:
        return "EVIDENCE_VERIFIED"
    if normalized in {"review_draft", "reviewer_specialist"}:
        return "REVIEW_STARTED" if status == "RUNNING" else "REVIEW_COMPLETED"
    if normalized in {"execute_scholar_skill", "harness_result"}:
        return "DRAFT_CREATED" if str(metadata.get("task_type")) != "SUPPORT_CLAIM" else "DOMAIN_RESULT"
    if "checkpoint" in normalized:
        return "CHECKPOINT_SAVED"
    if normalized in {"waiting_human_approval", "waiting_user"}:
        return "WAITING_USER"
    if normalized in {"run_interrupted"}:
        return "RUN_INTERRUPTED"
    if normalized in {"run_failed", "flow_failed"} or status == "FAILED":
        return "RUN_FAILED"
    return "TOOL_COMPLETED" if kind == "tool" else "DOMAIN_RESULT"


class RunEventStore:
    """Small multi-process-safe SQLite event store shared by API and Worker."""

    def __init__(self, project_root: Path, *, path: Path | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.path = path or root / "data" / "runtime" / "run_events.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        # Journal mode is a database-level setting.  Re-applying WAL on every
        # API read races with the Worker while it is appending an event and
        # can fail on Docker Desktop bind mounts with ``unable to open
        # database file`` / ``disk I/O error``.  Keep the existing database
        # journal mode stable; regular connections only need a busy timeout.
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    cursor INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    duration_ms REAL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(run_id, cursor)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS ix_run_events_run_cursor ON run_events(run_id, cursor)")

    def append(
        self,
        *,
        run_id: str,
        project_id: str,
        session_id: str | None,
        event_type: str,
        node: str,
        status: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
        duration_ms: float | int | None = None,
        timestamp: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        if event_type not in _EVENT_TYPES:
            event_type = "DOMAIN_RESULT"
        if status not in _EVENT_STATUSES:
            status = "FAILED" if status.casefold() in {"failed", "error"} else "COMPLETED"
        with self._lock, self._connect() as connection:
            # The cursor is allocated by multiple processes (API, Worker and
            # possibly a second Worker). A process-local Lock is insufficient;
            # serialize MAX(cursor)+INSERT at the SQLite write-transaction
            # boundary so the UNIQUE(run_id, cursor) contract cannot race.
            connection.execute("BEGIN IMMEDIATE")
            if event_id:
                existing = connection.execute("SELECT * FROM run_events WHERE event_id=?", (event_id,)).fetchone()
                if existing is not None:
                    return self._row(existing)
            row = connection.execute("SELECT COALESCE(MAX(cursor), 0) AS value FROM run_events WHERE run_id=?", (run_id,)).fetchone()
            cursor = int(row["value"]) + 1
            stable_id = event_id or f"{run_id}:{cursor}"
            connection.execute(
                """
                INSERT INTO run_events
                    (event_id, run_id, cursor, project_id, session_id, timestamp,
                     type, node, status, summary, duration_ms, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id,
                    run_id,
                    cursor,
                    project_id,
                    session_id,
                    timestamp or _now(),
                    event_type,
                    str(node)[:200] or "Scholar Run",
                    status,
                    str(summary)[:1000] or event_type,
                    duration_ms,
                    _json(metadata or {}),
                ),
            )
            return self._row(connection.execute("SELECT * FROM run_events WHERE event_id=?", (stable_id,)).fetchone())

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            raise KeyError("RunEvent 不存在")
        return {
            "event_id": str(row["event_id"]),
            "cursor": int(row["cursor"]),
            "run_id": str(row["run_id"]),
            "session_id": row["session_id"],
            "timestamp": str(row["timestamp"]),
            "type": str(row["type"]),
            "node": str(row["node"]),
            "status": str(row["status"]),
            "summary": str(row["summary"]),
            "duration_ms": row["duration_ms"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def append_trace(self, event: Mapping[str, Any], *, run_id: str, project_id: str, session_id: str | None) -> dict[str, Any]:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
        name = str(event.get("name") or event.get("stage") or "step")
        kind = str(event.get("kind") or "step")
        status = str(event.get("status") or "COMPLETED").upper()
        return self.append(
            run_id=run_id,
            project_id=project_id,
            session_id=session_id,
            event_type=_trace_event_type(name, kind, status, metadata),
            node=str(metadata.get("agent_role") or metadata.get("tool_name") or name.replace("_", " ").title()),
            status=status,
            summary=str(metadata.get("summary") or name.replace("_", " ").title()),
            duration_ms=event.get("elapsed_ms"),
            metadata={"trace_name": name, "kind": kind, **dict(metadata)},
            timestamp=str(event.get("timestamp")) if event.get("timestamp") else None,
        )

    def list(self, run_id: str, *, after: int = 0) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id=? AND cursor>? ORDER BY cursor",
                (run_id, after),
            ).fetchall()
        return tuple(self._row(row) for row in rows)

    def count(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS value FROM run_events WHERE run_id=?", (run_id,)).fetchone()
        return int(row["value"] if row else 0)
