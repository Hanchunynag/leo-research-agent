"""Deterministic outbox operation identities and execution helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any

from app.index_registry.models import IndexOperation


def operation_id_for(
    epoch: int, store: str, operation_type: str, object_id: str, target_hash: str
) -> str:
    return hashlib.sha256(
        "\x1f".join((str(epoch), store, operation_type, object_id, target_hash)).encode()
    ).hexdigest()


def make_operation(
    *, epoch: int, store: str, operation_type: str, object_id: str,
    target_hash: str, payload: dict[str, Any]
) -> IndexOperation:
    return IndexOperation(
        operation_id=operation_id_for(epoch, store, operation_type, object_id, target_hash),
        epoch=epoch, store=store, operation_type=operation_type, object_id=object_id,
        target_hash=target_hash, payload=payload,
    )


def execute_idempotently(
    operations: Iterable[IndexOperation],
    is_completed: Callable[[str], bool],
    execute: Callable[[IndexOperation], None],
) -> int:
    completed = 0
    for operation in operations:
        if is_completed(operation.operation_id):
            continue
        execute(operation)
        completed += 1
    return completed
