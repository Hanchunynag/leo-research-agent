"""Stable errors for CrewAI orchestration boundaries."""

from __future__ import annotations


class CheckpointPersistenceError(RuntimeError):
    """A recovery-critical Flow checkpoint could not be persisted."""

    code = "CHECKPOINT_PERSIST_FAILED"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
