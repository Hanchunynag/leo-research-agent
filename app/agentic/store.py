"""基于 sqlite3 的本地 Session、Topic、事件与 Evidence Registry。"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
EVENT_TYPES = {
    "user_query",
    "query_analysis",
    "evidence_added",
    "answer",
    "validation",
    "state_delta",
    "compaction",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def stable_json(value: Any) -> str:
    """确定性序列化事件内容，不引入时间戳或随机字段。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_session_id(session_id: str) -> str:
    cleaned = session_id.strip()
    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "session_id 必须为 1-64 位字母、数字、下划线或连字符，"
            "且以字母或数字开头。"
        )
    return cleaned


class AgenticSessionStore:
    """只追加历史事件；可变状态以新事件和投影表更新表达。"""

    def __init__(self, project_root: Path, database_path: Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        if database_path is None:
            self.database_path = (
                self.project_root / "data" / "runtime" / "agentic_sessions.sqlite3"
            )
        else:
            expanded = database_path.expanduser()
            self.database_path = (
                expanded.resolve()
                if expanded.is_absolute()
                else (self.project_root / expanded).resolve()
            )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active_topic_id TEXT,
                    status TEXT NOT NULL,
                    model TEXT,
                    provider TEXT,
                    configuration_fingerprint TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    parent_topic_id TEXT,
                    relation_to_previous TEXT NOT NULL,
                    topic_summary TEXT NOT NULL,
                    user_goal TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    confirmed_facts_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, topic_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, topic_id, ordinal),
                    FOREIGN KEY (session_id, topic_id)
                        REFERENCES topics(session_id, topic_id)
                );
                CREATE TABLE IF NOT EXISTS evidence_registry (
                    session_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section_path_json TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    block_ids_json TEXT NOT NULL,
                    first_seen_turn INTEGER NOT NULL,
                    last_used_turn INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, topic_id, evidence_id),
                    UNIQUE (session_id, topic_id, chunk_id),
                    FOREIGN KEY (session_id, topic_id)
                        REFERENCES topics(session_id, topic_id)
                );
                """
            )

    def create_session(
        self,
        title: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        configuration_fingerprint: str = "",
    ) -> dict[str, Any]:
        """创建一个不含秘密配置的持久化 Session。"""

        generated = session_id or f"session_{secrets.token_hex(6)}"
        cleaned_id = _validate_session_id(generated)
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, title, created_at, updated_at, active_topic_id,
                    status, model, provider, configuration_fingerprint
                ) VALUES (?, ?, ?, ?, NULL, 'active', ?, ?, ?)
                """,
                (
                    cleaned_id,
                    title.strip() or "Scientific RAG Session",
                    timestamp,
                    timestamp,
                    model,
                    provider,
                    configuration_fingerprint,
                ),
            )
        return self.get_session(cleaned_id)

    def get_or_create_session(
        self,
        session_id: str | None,
        title: str,
        *,
        model: str | None,
        provider: str | None,
        configuration_fingerprint: str,
    ) -> tuple[dict[str, Any], bool]:
        """恢复指定 Session，或创建新 Session 并返回是否新建。"""

        if session_id is not None:
            cleaned_id = _validate_session_id(session_id)
            try:
                return self.get_session(cleaned_id), False
            except KeyError:
                return (
                    self.create_session(
                        title,
                        session_id=cleaned_id,
                        model=model,
                        provider=provider,
                        configuration_fingerprint=configuration_fingerprint,
                    ),
                    True,
                )
        return (
            self.create_session(
                title,
                model=model,
                provider=provider,
                configuration_fingerprint=configuration_fingerprint,
            ),
            True,
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """按安全 ID 读取一个 Session。"""

        cleaned_id = _validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (cleaned_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Session 不存在：{cleaned_id}")
        return dict(row)

    def list_sessions(self) -> list[dict[str, Any]]:
        """按最近更新时间列出本地 Session。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, session_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_topics(self, session_id: str) -> list[dict[str, Any]]:
        """按创建顺序列出 Session 的全部 Topic。"""

        cleaned_id = _validate_session_id(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM topics WHERE session_id = ?
                ORDER BY created_at, topic_id
                """,
                (cleaned_id,),
            ).fetchall()
        return [self._decode_topic(dict(row)) for row in rows]

    def get_topic(self, session_id: str, topic_id: str) -> dict[str, Any]:
        """读取 Session 内的指定 Topic 投影。"""

        cleaned_id = _validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM topics WHERE session_id = ? AND topic_id = ?
                """,
                (cleaned_id, topic_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Topic 不存在：{topic_id}")
        return self._decode_topic(dict(row))

    @staticmethod
    def _decode_topic(row: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "entities_json",
            "confirmed_facts_json",
            "open_questions_json",
        ):
            row[field.removesuffix("_json")] = json.loads(str(row.pop(field)))
        return row

    def create_topic(
        self,
        session_id: str,
        *,
        relation: str,
        topic_summary: str,
        user_goal: str,
        entities: Sequence[str],
        parent_topic_id: str | None = None,
    ) -> dict[str, Any]:
        """创建 Topic 或相关子主题，并设为 Session 的活动 Topic。"""

        cleaned_id = _validate_session_id(session_id)
        timestamp = _now()
        with self._connect() as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM topics WHERE session_id = ?",
                    (cleaned_id,),
                ).fetchone()[0]
            )
            topic_id = f"T{count + 1:03d}"
            connection.execute(
                """
                INSERT INTO topics (
                    topic_id, session_id, parent_topic_id, relation_to_previous,
                    topic_summary, user_goal, entities_json,
                    confirmed_facts_json, open_questions_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?)
                """,
                (
                    topic_id,
                    cleaned_id,
                    parent_topic_id,
                    relation,
                    topic_summary,
                    user_goal,
                    stable_json(sorted(set(entities))),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE sessions SET active_topic_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (topic_id, timestamp, cleaned_id),
            )
        return self.get_topic(cleaned_id, topic_id)

    def set_active_topic(self, session_id: str, topic_id: str) -> None:
        """把已有 Topic 设为 Session 当前活动 Topic。"""

        self.get_topic(session_id, topic_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET active_topic_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (topic_id, _now(), _validate_session_id(session_id)),
            )

    def update_topic_state(
        self,
        session_id: str,
        topic_id: str,
        *,
        topic_summary: str,
        user_goal: str,
        entities: Sequence[str],
        confirmed_facts: Sequence[str],
        open_questions: Sequence[str],
    ) -> None:
        """更新 Topic 投影；历史变化仍由 append-only 事件保存。"""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE topics SET topic_summary = ?, user_goal = ?,
                    entities_json = ?, confirmed_facts_json = ?,
                    open_questions_json = ?, updated_at = ?
                WHERE session_id = ? AND topic_id = ?
                """,
                (
                    topic_summary,
                    user_goal,
                    stable_json(sorted(set(entities))),
                    stable_json(list(dict.fromkeys(confirmed_facts))),
                    stable_json(list(dict.fromkeys(open_questions))),
                    _now(),
                    _validate_session_id(session_id),
                    topic_id,
                ),
            )

    def append_event(
        self,
        session_id: str,
        topic_id: str,
        event_type: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        """以单调 ordinal 原子追加一个结构化 Topic 事件。"""

        if event_type not in EVENT_TYPES:
            raise ValueError(f"不支持的 event_type：{event_type}")
        cleaned_id = _validate_session_id(session_id)
        timestamp = _now()
        serialized = stable_json(content)
        with self._connect() as connection:
            ordinal = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(ordinal), 0) + 1 FROM events
                    WHERE session_id = ? AND topic_id = ?
                    """,
                    (cleaned_id, topic_id),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO events (
                    session_id, topic_id, ordinal, event_type, content_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cleaned_id, topic_id, ordinal, event_type, serialized, timestamp),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, cleaned_id),
            )
        return {
            "event_id": int(cursor.lastrowid or 0),
            "session_id": cleaned_id,
            "topic_id": topic_id,
            "ordinal": ordinal,
            "event_type": event_type,
            "content": content,
            "created_at": timestamp,
        }

    def list_events(
        self,
        session_id: str,
        topic_id: str,
    ) -> list[dict[str, Any]]:
        """按 ordinal 返回 Topic 的不可变事件历史。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM events WHERE session_id = ? AND topic_id = ?
                ORDER BY ordinal
                """,
                (_validate_session_id(session_id), topic_id),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["content"] = json.loads(str(value.pop("content_json")))
            events.append(value)
        return events

    def register_evidence(
        self,
        session_id: str,
        topic_id: str,
        candidate: dict[str, Any],
        turn_ordinal: int,
    ) -> tuple[dict[str, Any], bool]:
        """按 chunk_id 注册或复用稳定 Evidence ID。"""

        cleaned_id = _validate_session_id(session_id)
        chunk_id = str(candidate.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError("Evidence 缺少 chunk_id。")
        content = str(candidate.get("content") or "")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM evidence_registry
                WHERE session_id = ? AND topic_id = ? AND chunk_id = ?
                """,
                (cleaned_id, topic_id, chunk_id),
            ).fetchone()
            if existing is not None:
                existing_value = dict(existing)
                if str(existing_value["content_hash"]) != content_hash:
                    connection.execute(
                        """
                        UPDATE evidence_registry SET last_used_turn = ?,
                            content_hash = ?, status = 'active', evidence_json = ?
                        WHERE session_id = ? AND topic_id = ? AND chunk_id = ?
                        """,
                        (
                            turn_ordinal,
                            content_hash,
                            stable_json(candidate),
                            cleaned_id,
                            topic_id,
                            chunk_id,
                        ),
                    )
                    updated = dict(candidate)
                    updated.update(
                        {
                            "session_id": cleaned_id,
                            "topic_id": topic_id,
                            "evidence_id": existing_value["evidence_id"],
                            "content_hash": content_hash,
                            "first_seen_turn": existing_value["first_seen_turn"],
                            "last_used_turn": turn_ordinal,
                            "status": "active",
                        }
                    )
                    return updated, True
                connection.execute(
                    """
                    UPDATE evidence_registry SET last_used_turn = ?, status = 'reused'
                    WHERE session_id = ? AND topic_id = ? AND chunk_id = ?
                    """,
                    (turn_ordinal, cleaned_id, topic_id, chunk_id),
                )
                reused = existing_value
                reused["last_used_turn"] = turn_ordinal
                return self._decode_evidence(reused, status="reused"), False

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM evidence_registry
                    WHERE session_id = ? AND topic_id = ?
                    """,
                    (cleaned_id, topic_id),
                ).fetchone()[0]
            )
            evidence_id = f"E{count + 1:03d}"
            values = (
                cleaned_id,
                topic_id,
                evidence_id,
                chunk_id,
                str(candidate.get("work_id") or ""),
                str(candidate.get("document_id") or ""),
                str(candidate.get("title") or ""),
                stable_json(candidate.get("section_path") or []),
                int(candidate.get("page_start") or 1),
                int(candidate.get("page_end") or 1),
                stable_json(candidate.get("block_ids") or []),
                turn_ordinal,
                turn_ordinal,
                content_hash,
                "active",
                stable_json(candidate),
            )
            connection.execute(
                """
                INSERT INTO evidence_registry (
                    session_id, topic_id, evidence_id, chunk_id, work_id,
                    document_id, title, section_path_json, page_start, page_end,
                    block_ids_json, first_seen_turn, last_used_turn, content_hash,
                    status, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        record = dict(candidate)
        record.update(
            {
                "session_id": cleaned_id,
                "topic_id": topic_id,
                "evidence_id": evidence_id,
                "content_hash": content_hash,
                "first_seen_turn": turn_ordinal,
                "last_used_turn": turn_ordinal,
                "status": "active",
            }
        )
        return record, True

    @staticmethod
    def _decode_evidence(
        row: dict[str, Any],
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        candidate = json.loads(str(row.pop("evidence_json")))
        candidate.update(
            {
                "session_id": row["session_id"],
                "topic_id": row["topic_id"],
                "evidence_id": row["evidence_id"],
                "content_hash": row["content_hash"],
                "first_seen_turn": row["first_seen_turn"],
                "last_used_turn": row["last_used_turn"],
                "status": status or row["status"],
            }
        )
        return candidate

    def list_evidence(
        self,
        session_id: str,
        topic_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出 Session 或指定 Topic 的 Evidence Registry。"""

        cleaned_id = _validate_session_id(session_id)
        query = "SELECT * FROM evidence_registry WHERE session_id = ?"
        parameters: list[Any] = [cleaned_id]
        if topic_id is not None:
            query += " AND topic_id = ?"
            parameters.append(topic_id)
        query += " ORDER BY topic_id, evidence_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode_evidence(dict(row)) for row in rows]

    def session_details(self, session_id: str) -> dict[str, Any]:
        """返回 Session、Topic 及事件和证据计数。"""

        session = self.get_session(session_id)
        topics = self.list_topics(session_id)
        return {
            "session": session,
            "topics": [
                {
                    **topic,
                    "event_count": len(
                        self.list_events(session_id, str(topic["topic_id"]))
                    ),
                    "evidence_count": len(
                        self.list_evidence(session_id, str(topic["topic_id"]))
                    ),
                }
                for topic in topics
            ],
        }
