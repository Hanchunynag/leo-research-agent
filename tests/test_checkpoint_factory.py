from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.langchain_agent.checkpoint_factory import open_checkpointer
from app.langchain_agent.graph import LangGraphResearchRuntime


def test_memory_checkpointer_is_explicit_and_scoped() -> None:
    with open_checkpointer("memory") as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


def test_sqlite_checkpointer_reopens_persistent_database(tmp_path: Path) -> None:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        pytest.skip("optional official SQLite Checkpointer is not installed")

    database = tmp_path / "checkpoint.db"
    with open_checkpointer("sqlite", database_path=database) as checkpointer:
        assert isinstance(checkpointer, SqliteSaver)
        checkpointer.setup()
    assert database.is_file()


def test_sqlite_checkpointer_survives_runtime_rebuild_and_resume(tmp_path: Path) -> None:
    class Translation:
        def invoke(self, value: dict[str, Any]) -> dict[str, Any]:
            return {
                "original_query": value["query"],
                "zh_query": value["query"],
                "en_query": value["query"],
                "retrieval_queries": [value["query"]],
                "output_language": "en",
            }

    def retrieve(_state: dict[str, Any]) -> dict[str, Any]:
        return {"answerable": True, "answer": "resumed", "selected_evidence": []}

    database = tmp_path / "persistent-checkpoint.db"
    with open_checkpointer("sqlite", database_path=database) as saver:
        first = LangGraphResearchRuntime(Translation(), retrieve, checkpointer=saver)
        interrupted = first.invoke(
            {
                "original_query": "比较这四篇论文",
                "workspace_id": "default",
                "scope_version": 1,
                "last_search_results": [],
                "max_iterations": 4,
            },
            thread_id="restart-thread",
        )
        assert interrupted["__interrupt__"]

    with open_checkpointer("sqlite", database_path=database) as saver:
        rebuilt = LangGraphResearchRuntime(Translation(), retrieve, checkpointer=saver)
        resumed = rebuilt.resume("restart-thread", "A、B、C、D")
        isolated = rebuilt.invoke(
            {
                "original_query": "What is measured?",
                "workspace_id": "default",
                "scope_version": 1,
                "max_iterations": 1,
            },
            thread_id="other-thread",
        )

    assert resumed["final_answer"] == "resumed"
    assert isolated["final_answer"] == "resumed"
    assert "pending_question" not in isolated or isolated["pending_question"] is None
