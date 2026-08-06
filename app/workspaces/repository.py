"""不改变数据库 Schema 的 Workspace JSON 持久化。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.contracts import ResearchDirection, ResearchScope, ResearchWorkspace, WorkspaceDocument
from app.storage import write_json_atomic


SCHEMA_VERSION = "1.0"


class WorkspaceRepository:
    def __init__(self, project_root: Path, path: Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.path = path or self.project_root / "data" / "knowledge" / "workspaces.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": SCHEMA_VERSION, "workspaces": [], "scopes": [], "documents": [], "directions": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Workspace 文件 Schema 不受支持。")
        return value

    def _save(self, value: dict[str, Any]) -> None:
        write_json_atomic(self.path, value)

    def get_workspace(self, workspace_id: str) -> ResearchWorkspace | None:
        value = next((item for item in self._load()["workspaces"] if item.get("workspace_id") == workspace_id), None)
        return ResearchWorkspace(**value) if value is not None else None

    def put_workspace(self, workspace: ResearchWorkspace) -> None:
        payload = self._load()
        values = [item for item in payload["workspaces"] if item.get("workspace_id") != workspace.workspace_id]
        values.append(asdict(workspace))
        payload["workspaces"] = sorted(values, key=lambda item: item["workspace_id"])
        self._save(payload)

    def get_scope(self, workspace_id: str, scope_version: int) -> ResearchScope | None:
        value = next((item for item in self._load()["scopes"] if item.get("workspace_id") == workspace_id and item.get("scope_version") == scope_version), None)
        return ResearchScope(**value) if value is not None else None

    def put_scope(self, scope: ResearchScope) -> None:
        payload = self._load()
        values = [item for item in payload["scopes"] if not (item.get("workspace_id") == scope.workspace_id and item.get("scope_version") == scope.scope_version)]
        values.append(asdict(scope))
        payload["scopes"] = sorted(values, key=lambda item: (item["workspace_id"], item["scope_version"]))
        self._save(payload)

    def list_workspace_documents(self, workspace_id: str) -> tuple[WorkspaceDocument, ...]:
        return tuple(WorkspaceDocument(**item) for item in self._load()["documents"] if item.get("workspace_id") == workspace_id)

    def replace_workspace_documents(self, workspace_id: str, values: tuple[WorkspaceDocument, ...]) -> None:
        payload = self._load()
        payload["documents"] = [item for item in payload["documents"] if item.get("workspace_id") != workspace_id] + [asdict(value) for value in values]
        payload["documents"].sort(key=lambda item: (item["workspace_id"], item["document_id"], item["added_in_scope_version"]))
        self._save(payload)

    def list_directions(self, workspace_id: str) -> tuple[ResearchDirection, ...]:
        return tuple(ResearchDirection(**item) for item in self._load()["directions"] if item.get("workspace_id") == workspace_id)

    def put_direction(self, direction: ResearchDirection) -> None:
        payload = self._load()
        values = [item for item in payload["directions"] if item.get("direction_id") != direction.direction_id]
        values.append(asdict(direction))
        payload["directions"] = sorted(values, key=lambda item: item["direction_id"])
        self._save(payload)
