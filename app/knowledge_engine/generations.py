"""LightRAG 索引代际状态机和 active 指针。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.contracts import IndexGeneration
from app.storage import write_json_atomic


SCHEMA_VERSION = "1.0"
TRANSITIONS = {
    "pending": {"building", "failed"},
    "building": {"validating", "failed"},
    "validating": {"active", "failed"},
    "active": {"retired", "failed"},
    "retired": {"active"},
    "failed": {"building"},
    "superseded": {"active"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexGenerationRepository:
    def __init__(self, project_root: Path, path: Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.path = path or self.project_root / "data" / "knowledge" / "index_generations.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": SCHEMA_VERSION, "generations": [], "active": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("IndexGeneration Schema 不受支持。")
        return value

    def _save(self, value: dict[str, Any]) -> None:
        write_json_atomic(self.path, value)

    def list(self, workspace_id: str | None = None) -> tuple[IndexGeneration, ...]:
        values = self._load()["generations"]
        return tuple(IndexGeneration(**value) for value in values if workspace_id is None or value.get("workspace_id") == workspace_id)

    def get(self, generation_id: str) -> IndexGeneration | None:
        return next((value for value in self.list() if value.generation_id == generation_id), None)

    def active(self, workspace_id: str) -> IndexGeneration | None:
        generation_id = self._load()["active"].get(workspace_id)
        return self.get(generation_id) if isinstance(generation_id, str) else None

    def create(self, generation: IndexGeneration) -> IndexGeneration:
        if self.get(generation.generation_id) is not None:
            raise ValueError(f"IndexGeneration 已存在：{generation.generation_id}")
        value = replace(generation, state="building", created_at=generation.created_at or _now())
        payload = self._load()
        payload["generations"].append(asdict(value))
        self._save(payload)
        return value

    def transition(self, generation_id: str, state: str, *, failure_type: str | None = None) -> IndexGeneration:
        current = self.get(generation_id)
        if current is None:
            raise KeyError(f"IndexGeneration 不存在：{generation_id}")
        if state not in TRANSITIONS[current.state]:
            raise ValueError(f"非法索引状态转换：{current.state} -> {state}")
        updated = replace(current, state=state, failure_type=failure_type, activated_at=_now() if state == "active" else current.activated_at)  # type: ignore[arg-type]
        payload = self._load()
        payload["generations"] = [asdict(updated) if value.get("generation_id") == generation_id else value for value in payload["generations"]]
        self._save(payload)
        return updated

    def activate(self, generation_id: str) -> IndexGeneration:
        target = self.get(generation_id)
        if target is None:
            raise KeyError(f"IndexGeneration 不存在：{generation_id}")
        if target.state not in {"validating", "retired", "superseded"}:
            raise ValueError("只有 validating/retired generation 可以激活。")
        current = self.active(target.workspace_id)
        if current is not None and current.generation_id != generation_id:
            self.transition(current.generation_id, "retired")
        activated = self.transition(generation_id, "active")
        payload = self._load()
        payload["active"][target.workspace_id] = generation_id
        self._save(payload)
        return activated

    def fail(self, generation_id: str, error: BaseException) -> IndexGeneration:
        current = self.get(generation_id)
        if current is None:
            raise KeyError(generation_id)
        if current.state == "failed":
            return current
        return self.transition(generation_id, "failed", failure_type=type(error).__name__)
