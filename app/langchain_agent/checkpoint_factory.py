"""官方 LangGraph Checkpointer Factory。

当前支持的 SQLite 实现来自 LangGraph 1.x / checkpoint 4.x 生态的
``langgraph-checkpoint-sqlite>=3,<4``。连接生命周期由 context manager 管理，
不把 SQLite Connection 泄漏给 Graph 调用方。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from langgraph.checkpoint.memory import InMemorySaver


CheckpointMode = Literal["memory", "sqlite"]


class CheckpointUnavailable(RuntimeError):
    """The configured persistent working-state store cannot be opened."""

    code = "CHECKPOINT_UNAVAILABLE"


@contextmanager
def open_checkpointer(
    mode: CheckpointMode = "memory",
    *,
    database_path: Path | None = None,
) -> Iterator[Any]:
    """在明确的生命周期内打开官方 Checkpointer。"""

    if mode == "memory":
        yield InMemorySaver()
        return
    if mode != "sqlite":
        raise ValueError(f"不支持的 Checkpointer mode：{mode}")
    if database_path is None:
        raise CheckpointUnavailable("CHECKPOINT_UNAVAILABLE: SQLite Checkpointer 必须提供 database_path。")
    path = database_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:
        raise CheckpointUnavailable(
            "SQLite Checkpointer 未安装；请使用 "
            "langgraph-checkpoint-sqlite>=3,<4。"
        ) from error
    try:
        saver_context = SqliteSaver.from_conn_string(str(path))
    except (OSError, RuntimeError) as error:
        raise CheckpointUnavailable(
            f"CHECKPOINT_UNAVAILABLE: 无法打开 SQLite Checkpointer：{path}"
        ) from error
    with saver_context as saver:
        try:
            saver.setup()
        except (OSError, RuntimeError) as error:
            raise CheckpointUnavailable(
                f"CHECKPOINT_UNAVAILABLE: 无法初始化 SQLite Checkpointer：{path}"
            ) from error
        yield saver


def checkpointer_has_checkpoint(checkpointer: Any, thread_id: str) -> bool:
    """Return whether a LangGraph saver contains a working state for a thread.

    This deliberately uses the public saver read contract and also supports
    tiny test doubles.  It never treats an in-memory object as persistent.
    """

    if not isinstance(thread_id, str) or not thread_id.strip():
        return False
    getter = getattr(checkpointer, "get_tuple", None)
    if not callable(getter):
        getter = getattr(checkpointer, "get", None)
    if not callable(getter):
        return False
    config = {"configurable": {"thread_id": thread_id}}
    try:
        value = getter(config)
    except (KeyError, LookupError, TypeError, ValueError):
        return False
    return value is not None
