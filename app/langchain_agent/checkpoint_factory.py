"""官方 LangGraph Checkpointer Factory。

当前支持的 SQLite 实现来自与 LangGraph 0.3.x / checkpoint 2.x 共存的
``langgraph-checkpoint-sqlite==2.0.0``。连接生命周期由 context manager 管理，
不把 SQLite Connection 泄漏给 Graph 调用方。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from langgraph.checkpoint.memory import InMemorySaver


CheckpointMode = Literal["memory", "sqlite"]


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
        raise ValueError("SQLite Checkpointer 必须提供 database_path。")
    path = database_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:
        raise RuntimeError(
            "SQLite Checkpointer 未安装；请使用 "
            "langgraph-checkpoint-sqlite==2.0.0。"
        ) from error
    with SqliteSaver.from_conn_string(str(path)) as saver:
        saver.setup()
        yield saver
