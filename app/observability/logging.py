"""Structured correlation logging helper."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping


def log_run_event(logger: logging.Logger, message: str, *, context: Mapping[str, Any] | None = None, level: int = logging.INFO) -> None:
    """Emit one JSON object with the standard run correlation fields."""

    payload = {
        "message": message,
        **{
            str(key): value
            for key, value in (context or {}).items()
            if str(key).casefold() not in {"prompt", "secret", "api_key", "password", "token"}
        },
    }
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))

