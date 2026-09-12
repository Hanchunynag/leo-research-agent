"""Small request-level cache for external literature discovery metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from app.storage import write_json_atomic


def literature_request_fingerprint(
    query: str,
    *,
    provider_set: tuple[str, ...] = (),
    requested_from: str | None = None,
    requested_to: str | None = None,
    max_results: int = 10,
    source_policy: str = "default",
) -> str:
    payload = {
        "query": " ".join(query.casefold().split()),
        "provider_set": sorted(provider_set),
        "requested_from": requested_from,
        "requested_to": requested_to,
        "max_results": max_results,
        "source_policy": source_policy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class LiteratureDiscoveryCache:
    """Caches normalized discovery results, never verified Evidence."""

    def __init__(self, project_root: Path, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds < 1:
            raise ValueError("Literature cache TTL 必须大于 0。")
        self.path = project_root.expanduser().resolve() / "data" / "research" / "literature_cache.json"
        self.ttl_seconds = ttl_seconds

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": "1.0", "entries": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != "1.0":
            raise ValueError("Literature cache schema 不受支持。")
        entries = value.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("Literature cache entries 必须是对象。")
        return value

    def get(self, fingerprint: str, *, now: datetime | None = None) -> Mapping[str, Any] | None:
        value = self._load()["entries"].get(fingerprint)
        if not isinstance(value, dict):
            return None
        retrieved = value.get("retrieved_at")
        if not isinstance(retrieved, str):
            return None
        try:
            timestamp = datetime.fromisoformat(retrieved)
        except ValueError:
            return None
        current = now or datetime.now(timezone.utc)
        if timestamp + timedelta(seconds=self.ttl_seconds) <= current:
            return None
        return dict(value)

    def put(self, fingerprint: str, value: Mapping[str, Any], *, now: datetime | None = None) -> None:
        timestamp = now or datetime.now(timezone.utc)
        payload = self._load()
        payload["entries"][fingerprint] = {
            "request_fingerprint": fingerprint,
            "retrieved_at": timestamp.isoformat(),
            "expires_at": (timestamp + timedelta(seconds=self.ttl_seconds)).isoformat(),
            **dict(value),
        }
        write_json_atomic(self.path, payload)
