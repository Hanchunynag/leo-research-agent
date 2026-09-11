"""Legacy AgenticSessionStore 的只读兼容 Adapter。"""

from __future__ import annotations

from typing import Any

from app.session.runtime import SessionRuntime


class LegacySessionAdapter:
    """只读取旧 Store，并把可安全识别的 Conversation 映射到新 Session。"""

    def __init__(self, store: Any) -> None:
        self.store = store

    def snapshot(self, session_id: str) -> dict[str, Any] | None:
        try:
            session = self.store.get_session(session_id)
        except (KeyError, OSError, ValueError):
            return None
        topic_id = session.get("active_topic_id")
        events = (
            self.store.list_events(session_id, str(topic_id))
            if isinstance(topic_id, str) and topic_id
            else []
        )
        return {
            "session": dict(session),
            "topic_id": topic_id,
            "events": [dict(value) for value in events if isinstance(value, dict)],
        }

    def migrate_messages(self, session_id: str, runtime: SessionRuntime) -> bool:
        snapshot = self.snapshot(session_id)
        if snapshot is None:
            return False
        for event in snapshot["events"]:
            content = event.get("content")
            if not isinstance(content, dict):
                continue
            event_type = str(event.get("event_type") or "")
            if event_type == "user_query" and content.get("query"):
                runtime.append_message(
                    "user",
                    str(content["query"]),
                    metadata={"legacy_event_id": event.get("event_id")},
                )
            elif event_type == "answer":
                answer = content.get("answer") or content.get("refusal_reason")
                if answer:
                    runtime.append_message(
                        "assistant",
                        str(answer),
                        metadata={"legacy_event_id": event.get("event_id")},
                    )
        return True

    def evidence(self, session_id: str) -> list[dict[str, Any]]:
        try:
            values = self.store.list_evidence(session_id)
        except (KeyError, OSError, ValueError):
            return []
        return [dict(value) for value in values if isinstance(value, dict)]
