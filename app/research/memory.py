"""Workspace/Session/Run/Job 分层状态与长期记忆白名单。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.storage import write_json_atomic


_LONG_TERM_FIELDS = frozenset(
    {
        "confirmed_scope",
        "document_roles",
        "direction_tree",
        "verified_conclusions",
        "pending_hypotheses",
    }
)
_FORBIDDEN = frozenset(
    {
        "hidden_reasoning",
        "chain_of_thought",
        "temporary_guesses",
        "tool_logs",
        "unverified_facts",
    }
)


class ResearchStateStore:
    def __init__(self, project_root: Path) -> None:
        self.path = project_root.expanduser().resolve() / "data" / "research" / "state.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": "1.0",
                "workspace_state": {},
                "session_state": {},
                "run_state": {},
                "job_state": {},
            }
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != "1.0":
            raise ValueError("Research State Schema 不受支持。")
        return value

    def workspace(self, workspace_id: str) -> dict[str, Any]:
        return dict(self._load()["workspace_state"].get(workspace_id, {}))

    def commit_workspace(self, workspace_id: str, updates: Mapping[str, Any]) -> None:
        unknown = set(updates) - _LONG_TERM_FIELDS
        if unknown:
            raise ValueError(f"Workspace 长期状态字段不允许：{sorted(unknown)}")
        if _FORBIDDEN & set(updates):
            raise ValueError("禁止保存隐藏推理、临时猜测、工具日志或未验证事实。")
        payload = self._load()
        current = dict(payload["workspace_state"].get(workspace_id, {}))
        current.update({key: value for key, value in updates.items() if key in _LONG_TERM_FIELDS})
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["workspace_state"][workspace_id] = current
        write_json_atomic(self.path, payload)

    def commit_session(self, session_id: str, summary: Mapping[str, Any]) -> None:
        safe = {
            key: value
            for key, value in summary.items()
            if key in {"workspace_id", "last_query", "last_workflow", "turn_count"}
        }
        payload = self._load()
        payload["session_state"][session_id] = safe
        write_json_atomic(self.path, payload)

    def commit_run(self, run_id: str, summary: Mapping[str, Any]) -> None:
        safe = {
            key: value
            for key, value in summary.items()
            if key in {"workflow", "state", "termination_reason", "selected_evidence_ids"}
        }
        payload = self._load()
        payload["run_state"][run_id] = safe
        write_json_atomic(self.path, payload)

    def commit_job(self, job_id: str, summary: Mapping[str, Any]) -> None:
        safe = {
            key: value
            for key, value in summary.items()
            if key in {"kind", "status", "workspace_id", "recoverable"}
        }
        payload = self._load()
        payload["job_state"][job_id] = safe
        write_json_atomic(self.path, payload)
