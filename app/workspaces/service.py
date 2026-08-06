"""Workspace/Scope 的版本推进和硬过滤规则。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from app.contracts import CandidateEvidence, EvidenceGrade, ResearchDirection, ResearchScope, ResearchWorkspace, WorkspaceDocument
from app.corpus import CanonicalCorpusService
from app.workspaces.repository import WorkspaceRepository


DEFAULT_WORKSPACE_ID = "default"


class WorkspaceService:
    def __init__(self, project_root: Path, repository: WorkspaceRepository | None = None, corpus: CanonicalCorpusService | None = None) -> None:
        self.repository = repository or WorkspaceRepository(project_root)
        self.corpus = corpus or CanonicalCorpusService(project_root)

    def ensure_default(self) -> ResearchWorkspace:
        existing = self.repository.get_workspace(DEFAULT_WORKSPACE_ID)
        if existing is not None:
            return existing
        document_ids = tuple(sorted(value.document_id for value in self.corpus.list_documents()))
        workspace = ResearchWorkspace(DEFAULT_WORKSPACE_ID, 1, "Default workspace")
        scope = ResearchScope(DEFAULT_WORKSPACE_ID, 1, included_document_ids=document_ids)
        memberships = tuple(WorkspaceDocument(DEFAULT_WORKSPACE_ID, document_id, 1) for document_id in document_ids)
        self.repository.put_workspace(workspace)
        self.repository.put_scope(scope)
        self.repository.replace_workspace_documents(DEFAULT_WORKSPACE_ID, memberships)
        return workspace

    def synchronize_default_documents(self) -> ResearchWorkspace:
        """兼容迁移：把新出现的 canonical 文档加入 default workspace。"""

        workspace = self.ensure_default()
        canonical_ids = {value.document_id for value in self.corpus.list_documents()}
        active_ids = set(self.active_document_ids(DEFAULT_WORKSPACE_ID, workspace.scope_version))
        if canonical_ids != active_ids:
            self.replace_scope_documents(DEFAULT_WORKSPACE_ID, canonical_ids)
            refreshed = self.repository.get_workspace(DEFAULT_WORKSPACE_ID)
            assert refreshed is not None
            return refreshed
        return workspace

    def require_scope(self, workspace_id: str, scope_version: int) -> ResearchScope:
        workspace = self.repository.get_workspace(workspace_id)
        if workspace is None:
            raise KeyError(f"Workspace 不存在：{workspace_id}")
        if scope_version > workspace.scope_version:
            raise ValueError(f"scope_version {scope_version} 尚不存在；当前为 {workspace.scope_version}")
        scope = self.repository.get_scope(workspace_id, scope_version)
        if scope is None:
            raise KeyError(f"Scope 快照不存在：{workspace_id}@{scope_version}")
        return scope

    def active_document_ids(self, workspace_id: str, scope_version: int) -> frozenset[str]:
        scope = self.require_scope(workspace_id, scope_version)
        return frozenset(set(scope.included_document_ids) - set(scope.excluded_document_ids))

    def evidence_grade(self, workspace_id: str, scope_version: int, document_id: str) -> EvidenceGrade:
        active = [
            value
            for value in self.repository.list_workspace_documents(workspace_id)
            if value.document_id == document_id
            and value.added_in_scope_version <= scope_version
            and (value.removed_in_scope_version is None or scope_version < value.removed_in_scope_version)
        ]
        if not active:
            raise KeyError(f"文档不属于 Scope：{document_id}")
        return active[-1].evidence_grade

    def filter_candidates(self, workspace_id: str, scope_version: int, candidates: Iterable[CandidateEvidence]) -> tuple[CandidateEvidence, ...]:
        allowed = self.active_document_ids(workspace_id, scope_version)
        scope = self.require_scope(workspace_id, scope_version)
        filtered: list[CandidateEvidence] = []
        for value in candidates:
            if value.workspace_id != workspace_id or value.scope_version != scope_version:
                continue
            if value.document_id not in allowed:
                continue
            direction_id = value.metadata.get("direction_id")
            if direction_id in scope.excluded_direction_ids:
                continue
            filtered.append(value)
        return tuple(filtered)

    def replace_scope_documents(self, workspace_id: str, document_ids: Iterable[str], *, grades: dict[str, str] | None = None) -> ResearchScope:
        workspace = self.repository.get_workspace(workspace_id)
        if workspace is None:
            raise KeyError(f"Workspace 不存在：{workspace_id}")
        requested = tuple(sorted(set(document_ids)))
        known = {value.document_id for value in self.corpus.list_documents()}
        unknown = set(requested) - known
        if unknown:
            raise KeyError(f"Canonical Document 不存在：{sorted(unknown)}")
        next_version = workspace.scope_version + 1
        previous = self.require_scope(workspace_id, workspace.scope_version)
        previous_active = set(previous.included_document_ids) - set(previous.excluded_document_ids)
        scope = ResearchScope(
            workspace_id,
            next_version,
            included_document_ids=requested,
            excluded_document_ids=tuple(sorted(previous_active - set(requested))),
            included_direction_ids=previous.included_direction_ids,
            excluded_direction_ids=previous.excluded_direction_ids,
        )
        memberships = list(self.repository.list_workspace_documents(workspace_id))
        active_by_id = {item.document_id: item for item in memberships if item.removed_in_scope_version is None}
        for document_id in previous_active - set(requested):
            old = active_by_id[document_id]
            memberships[memberships.index(old)] = replace(old, removed_in_scope_version=next_version)
        for document_id in set(requested) - previous_active:
            grade = (grades or {}).get(document_id, "primary")
            memberships.append(WorkspaceDocument(workspace_id, document_id, next_version, evidence_grade=grade))  # type: ignore[arg-type]
        self.repository.replace_workspace_documents(workspace_id, tuple(memberships))
        self.repository.put_scope(scope)
        self.repository.put_workspace(replace(workspace, scope_version=next_version))
        return scope

    def put_direction(self, direction: ResearchDirection) -> None:
        workspace = self.repository.get_workspace(direction.workspace_id)
        if workspace is None:
            raise KeyError(f"Workspace 不存在：{direction.workspace_id}")
        self.repository.put_direction(direction)
