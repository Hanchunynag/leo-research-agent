"""回答模型配置与异常文本的最小敏感信息脱敏工具。"""

from __future__ import annotations

import re
from collections.abc import Iterable


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_COMMON_KEY = re.compile(r"\b(?:sk|ds)-[A-Za-z0-9_-]{8,}\b")


def redact_sensitive_text(
    value: object,
    *,
    known_secrets: Iterable[str | None] = (),
) -> str:
    """脱敏常见 Key 形式及调用方明确提供的秘密值。"""

    text = str(value)
    for secret in known_secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    text = _SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", text)
    text = _BEARER_TOKEN.sub("Bearer [REDACTED]", text)
    return _COMMON_KEY.sub("[REDACTED]", text)
