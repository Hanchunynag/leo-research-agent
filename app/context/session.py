"""可复用且带完整性指纹的本地 Context Session。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.context.models import ContextBundle
from app.storage import write_json_atomic


CONTEXT_SESSION_SCHEMA_VERSION = "1.0"
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_CONTEXT_ADAPTER = TypeAdapter(ContextBundle)


def _validate_session_id(session_id: str) -> str:
    cleaned = session_id.strip()
    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "context session ID 必须为 1-64 位字母、数字、下划线或连字符，"
            "且以字母或数字开头。"
        )
    return cleaned


def _canonical_context_payload(context: ContextBundle) -> dict[str, Any]:
    payload = context.to_dict()
    payload.pop("evidence_count", None)
    return payload


def context_fingerprint(context: ContextBundle) -> str:
    return _context_payload_fingerprint(_canonical_context_payload(context))


def _context_payload_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextSession:
    session_id: str
    context_hash: str
    created_at: str
    context: ContextBundle

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "context_hash": self.context_hash,
            "created_at": self.created_at,
            "context": _canonical_context_payload(self.context),
        }


class ContextSessionStore:
    """在项目私有 data 目录中保存和读取固定证据快照。"""

    def __init__(self, project_root: Path) -> None:
        self.directory = (
            project_root.expanduser().resolve()
            / "data"
            / "runtime"
            / "context_sessions"
        )

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{_validate_session_id(session_id)}.json"

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).is_file()

    def save(self, session_id: str, context: ContextBundle) -> ContextSession:
        cleaned_id = _validate_session_id(session_id)
        session = ContextSession(
            session_id=cleaned_id,
            context_hash=context_fingerprint(context),
            created_at=datetime.now(UTC).isoformat(),
            context=context,
        )
        write_json_atomic(self.path_for(cleaned_id), session.to_dict())
        return session

    def load(self, session_id: str) -> ContextSession:
        cleaned_id = _validate_session_id(session_id)
        path = self.path_for(cleaned_id)
        if not path.is_file():
            raise FileNotFoundError(f"Context Session 不存在：{cleaned_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Context Session 无法读取：{cleaned_id}") from error
        if not isinstance(payload, dict):
            raise ValueError("Context Session 必须是 JSON object。")
        if payload.get("schema_version") != CONTEXT_SESSION_SCHEMA_VERSION:
            raise ValueError("Context Session schema_version 不受支持。")
        if payload.get("session_id") != cleaned_id:
            raise ValueError("Context Session ID 与文件名不一致。")
        context_hash = payload.get("context_hash")
        created_at = payload.get("created_at")
        if not isinstance(context_hash, str) or not isinstance(created_at, str):
            raise ValueError("Context Session 元数据不完整。")
        context_payload = payload.get("context")
        if not isinstance(context_payload, dict):
            raise ValueError("Context Session 中的 ContextBundle 不合法。")
        actual_hash = _context_payload_fingerprint(context_payload)
        if actual_hash != context_hash:
            raise ValueError("Context Session 完整性指纹不匹配。")
        try:
            context = _CONTEXT_ADAPTER.validate_python(context_payload)
        except ValidationError as error:
            raise ValueError("Context Session 中的 ContextBundle 不合法。") from error
        return ContextSession(cleaned_id, context_hash, created_at, context)
