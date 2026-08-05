"""Stable registry records shared by index backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


EpochStatus = Literal["pending", "active", "failed", "superseded"]
DiffKind = Literal[
    "added", "dense_changed", "graph_changed", "changed", "deleted", "unchanged"
]
OperationStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class ChunkVersion:
    chunk_key: str
    chunk_id: str
    document_id: str
    work_id: str
    paper_id: str
    content_hash: str
    dense_text_hash: str
    graph_text_hash: str
    valid_from_epoch: int
    valid_to_epoch: int | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkDiff:
    kind: DiffKind
    chunk_key: str
    current: dict[str, Any] | None
    previous: dict[str, Any] | None
    dense_changed: bool = False
    graph_changed: bool = False


@dataclass(frozen=True)
class IndexOperation:
    operation_id: str
    epoch: int
    store: str
    operation_type: str
    object_id: str
    target_hash: str
    payload: dict[str, Any]
    status: OperationStatus = "pending"
