from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.langchain_agent.checkpoint_factory import open_checkpointer


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
