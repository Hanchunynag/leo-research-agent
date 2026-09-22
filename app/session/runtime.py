"""One session-private SQLite database for conversation and Run projections."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from app.session.models import RunRecord, RunStatus, SessionRecord
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.orchestration.contracts import RunGoal


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
                    project_id TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    principal_id TEXT NOT NULL DEFAULT 'local',
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
                    job_id TEXT,
                    project_id TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    principal_id TEXT NOT NULL DEFAULT 'local',
                    checkpoint_ref TEXT,
                    failure_message TEXT
                );
                CREATE TABLE IF NOT EXISTS run_goals (
                    run_id TEXT PRIMARY KEY,
                    original_instruction TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_criteria_json TEXT NOT NULL,
                    hard_constraints_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
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
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    checkpoint_ref TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(session_info)")
            }
            if "project_id" not in columns:
                connection.execute("ALTER TABLE session_info ADD COLUMN project_id TEXT")
            if "tenant_id" not in columns:
                connection.execute(
                    "ALTER TABLE session_info ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'"
                )
            if "principal_id" not in columns:
                connection.execute(
                    "ALTER TABLE session_info ADD COLUMN principal_id TEXT NOT NULL DEFAULT 'local'"
                )
            run_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(agent_runs)")
            }
            for column in (
                "job_id",
                "project_id",
                "tenant_id",
                "principal_id",
                "checkpoint_ref",
            ):
                if column not in run_columns:
                    default = " NOT NULL DEFAULT 'local'" if column in {"tenant_id", "principal_id"} else ""
                    connection.execute(
                        f"ALTER TABLE agent_runs ADD COLUMN {column} TEXT{default}"
                    )
            connection.execute(
                """
                INSERT INTO session_info
                    (session_id, title, status, created_at, updated_at, project_id,
                     tenant_id, principal_id, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    project_id=excluded.project_id,
                    schema_version=excluded.schema_version
                """,
                (
                    self.session.session_id,
                    self.session.title,
                    self.session.status,
                    self.session.created_at,
                    self.session.updated_at,
                    self.session.project_id,
                    self.session.tenant_id,
                    self.session.principal_id,
                    self.session.schema_version,
                ),
            )

    def attach_project(self, project_id: str) -> None:
        value = project_id.strip()
        if not value:
            raise ValueError("project_id 不能为空。")
        with self._connect() as connection:
            connection.execute(
                "UPDATE session_info SET project_id=?, updated_at=? WHERE session_id=?",
                (value, _now(), self.session.session_id),
            )

    def create_run(
        self,
        query: str,
        *,
        run_id: str,
        thread_id: str,
        trace_id: str | None = None,
        job_id: str | None = None,
        project_id: str | None = None,
        task_type: str | None = None,
        goal: RunGoal | Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        principal_id: str | None = None,
    ) -> RunRecord:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Run query 不能为空。")
        created = _now()
        from app.orchestration.contracts import RunGoal, default_run_goal

        if goal is None:
            goal_value = default_run_goal(cleaned, task_type)
        elif isinstance(goal, RunGoal):
            goal_value = goal
        else:
            goal_value = RunGoal.model_validate(goal)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs
                    (run_id, query, status, thread_id, created_at, trace_id,
                     job_id, project_id, tenant_id, principal_id)
                VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cleaned,
                    thread_id,
                    created,
                    trace_id,
                    job_id,
                    project_id,
                    tenant_id or self.session.tenant_id,
                    principal_id or self.session.principal_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_goals(
                    run_id, original_instruction, task_type,
                    success_criteria_json, hard_constraints_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    goal_value.original_instruction,
                    goal_value.task_type,
                    _json(goal_value.success_criteria),
                    _json(goal_value.hard_constraints),
                    created,
                ),
            )
        return self.get_run(run_id)

    def get_run_goal(self, run_id: str) -> RunGoal:
        """Read the immutable OriginalGoal stored for ``run_id``."""

        from app.orchestration.contracts import RunGoal, default_run_goal

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_goals WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            # Backward-compatible migration for Runs created before goal
            # persistence existed. The first read materializes the goal once;
            # later Agent calls can never replace it through the runtime API.
            run = self.get_run(run_id)
            goal = default_run_goal(run.query)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO run_goals(
                        run_id, original_instruction, task_type,
                        success_criteria_json, hard_constraints_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        goal.original_instruction,
                        goal.task_type,
                        _json(goal.success_criteria),
                        _json(goal.hard_constraints),
                        _now(),
                    ),
                )
            return goal
        return RunGoal(
            original_instruction=str(row["original_instruction"]),
            task_type=str(row["task_type"]),
            success_criteria=tuple(json.loads(str(row["success_criteria_json"]))),
            hard_constraints=tuple(json.loads(str(row["hard_constraints_json"]))),
        )

    def start_run(self, run_id: str, *, worker_id: str) -> RunRecord:
        started = _now()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE agent_runs
                SET status='RUNNING', started_at=?, worker_id=?
                WHERE run_id=? AND status IN ('PENDING', 'QUEUED', 'INTERRUPTED')
                """,
                (started, worker_id, run_id),
            ).rowcount
        if updated != 1:
            raise ValueError(f"Run 不能进入 RUNNING：{run_id}")
        return self.get_run(run_id)

    def queue_run(self, run_id: str, *, job_id: str | None = None) -> RunRecord:
        """Move a persisted run into the queue without executing cognition."""

        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE agent_runs SET status='QUEUED', job_id=COALESCE(?, job_id)
                WHERE run_id=? AND status IN ('PENDING', 'WAITING_USER', 'INTERRUPTED')
                """,
                (job_id, run_id),
            ).rowcount
        if changed != 1:
            current = self.get_run(run_id)
            if current.status not in {"QUEUED", "RUNNING"}:
                raise ValueError(f"Run 不能进入 QUEUED：{run_id}")
        return self.get_run(run_id)

    def bind_job(self, run_id: str, job_id: str) -> RunRecord:
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE agent_runs SET job_id=? WHERE run_id=?",
                (job_id, run_id),
            ).rowcount
        if changed != 1:
            raise KeyError(f"Run 不存在：{run_id}")
        return self.get_run(run_id)

    def set_checkpoint_ref(self, run_id: str, checkpoint_ref: str | None) -> RunRecord:
        """Record the CrewAI Flow checkpoint reference without copying state."""

        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE agent_runs SET checkpoint_ref=? WHERE run_id=?",
                (checkpoint_ref, run_id),
            ).rowcount
        if updated != 1:
            raise KeyError(f"Run 不存在：{run_id}")
        return self.get_run(run_id)

    def save_checkpoint(self, run_id: str, state: Mapping[str, Any]) -> str:
        """Persist bounded orchestration state and its Run reference.

        This is intentionally a small Session/Run checkpoint projection. It
        stores contracts, counters and domain-result references only; CrewAI
        Memory is never used as a durability mechanism.
        """

        reference = f"session://{self.session.session_id}/runs/{run_id}/manager"
        goal = self.get_run_goal(run_id)
        state_payload = dict(state)
        # Persist a single authoritative envelope. A malicious or stale Flow
        # projection cannot overwrite the immutable goal row.
        state_payload.pop("goal", None)
        payload = _json(
            {
                "run_goal": goal.model_dump(mode="json"),
                "run_state": state_payload,
            }
        )
        with self._connect() as connection:
            updated = connection.execute(
                """
                INSERT INTO run_checkpoints(run_id, checkpoint_ref, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    checkpoint_ref=excluded.checkpoint_ref,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (run_id, reference, payload, _now()),
            ).rowcount
            if updated != 1:
                raise KeyError(f"Run 不存在：{run_id}")
            changed = connection.execute(
                "UPDATE agent_runs SET checkpoint_ref=? WHERE run_id=?",
                (reference, run_id),
            ).rowcount
        if changed != 1:
            raise KeyError(f"Run 不存在：{run_id}")
        return reference

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM run_checkpoints WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row["state_json"]))
        if not isinstance(value, dict):
            return None
        # Old checkpoints were raw FlowState projections. Return the same
        # envelope shape for new callers while keeping legacy fields usable.
        if "run_state" not in value:
            goal = self.get_run_goal(run_id)
            return {"run_goal": goal.model_dump(mode="json"), "run_state": value}
        return value

    def clear_checkpoint(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM run_checkpoints WHERE run_id=?", (run_id,))
            connection.execute(
                "UPDATE agent_runs SET checkpoint_ref=NULL WHERE run_id=?", (run_id,)
            )

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

    def list_messages(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.recent_messages(limit)

    def list_results(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, status, answer, citations_json, evidence_json,
                       metadata_json, created_at
                FROM research_results ORDER BY created_at, result_id
                """
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["citations"] = json.loads(value.pop("citations_json"))
            value["evidence"] = json.loads(value.pop("evidence_json"))
            value["metadata"] = json.loads(value.pop("metadata_json"))
            results.append(value)
        return results

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
        if status not in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED", "WAITING_USER", "WAITING_HUMAN_APPROVAL"}:
            raise ValueError(f"非法最终 Run 状态：{status}")
        storage_status = "WAITING_USER" if status == "WAITING_HUMAN_APPROVAL" else status
        finished = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status=?, completed_at=?
                WHERE run_id=?
                """,
                (storage_status, finished, run_id),
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
                    storage_status,
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
