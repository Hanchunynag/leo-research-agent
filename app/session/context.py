"""Session Conversation Context Builder 的最小实现。"""

from __future__ import annotations

from typing import Any

from app.session.runtime import SessionRuntime


class ConversationContextBuilder:
    """只选择当前 Query 和最近消息，不复制 LangGraph Working State。"""

    def __init__(self, recent_limit: int = 8) -> None:
        self.recent_limit = recent_limit

    def build(
        self,
        runtime: SessionRuntime,
        query: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "current_query": query,
            "recent_messages": runtime.recent_messages(self.recent_limit),
            "research_state": None,
            "project_id": project_id,
            "shared_knowledge": "resolved by Research Engine",
        }
