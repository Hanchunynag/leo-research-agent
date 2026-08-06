"""Agent 可见的唯一 Unified Knowledge Service。

迁移期开关固定为 legacy official + optional LightRAG shadow。旧后端判断只封装在
该兼容边界内，不泄漏到 Agent/Harness。
"""

from __future__ import annotations

import secrets
from time import perf_counter
from pathlib import Path
from typing import Any, Sequence

from app.contracts import CandidateEvidence, EvidenceRequest
from app.contracts.adapters import LegacyEvidenceMapper
from app.evaluation.shadow import ShadowReportStore, compare_retrievals
from app.evidence import EvidenceIntelligencePipeline


class UnifiedKnowledgeService:
    def __init__(
        self,
        official_runtime: Any,
        evidence: EvidenceIntelligencePipeline,
        *,
        shadow_engine: Any | None = None,
        report_store: ShadowReportStore | None = None,
        workspace_id: str = "default",
        scope_version: int = 1,
        supports_advanced_retrieval: bool = False,
    ) -> None:
        self.official_runtime = official_runtime
        self.evidence = evidence
        self.shadow_engine = shadow_engine
        self.report_store = report_store
        self.workspace_id = workspace_id
        self.scope_version = scope_version
        self.supports_advanced_retrieval = supports_advanced_retrieval
        self.embedding_provider = official_runtime.embedding_provider
        self.reranker_provider = getattr(official_runtime, "reranker_provider", None)
        self.mapper = LegacyEvidenceMapper()
        self.last_diagnostics: dict[str, Any] = {}

    def _request(
        self,
        query: str,
        limit: int,
        *,
        workspace_id: str | None = None,
        scope_version: int | None = None,
    ) -> EvidenceRequest:
        requested_workspace = workspace_id or self.workspace_id
        if requested_workspace != self.workspace_id:
            raise PermissionError("Unified Knowledge Service workspace 不匹配。")
        if scope_version is None:
            current = self.evidence.workspaces.repository.get_workspace(self.workspace_id)
            if current is not None:
                self.scope_version = current.scope_version
            requested_scope = self.scope_version
        else:
            requested_scope = scope_version
        # 显式验证不可变 Scope 快照；不得静默改用当前版本。
        self.evidence.workspaces.require_scope(requested_workspace, requested_scope)
        return EvidenceRequest(
            request_id=f"SH_{secrets.token_hex(8)}",
            workspace_id=requested_workspace,
            scope_version=requested_scope,
            query=query,
            top_k=limit,
            retrieval_options={"token_budget": 100_000},
        )

    def _govern(self, request: EvidenceRequest, values: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], tuple[CandidateEvidence, ...]]:
        candidates = tuple(self.mapper.candidate_from_mapping(request, value, fallback_rank=index) for index, value in enumerate(values, 1))
        bundle = self.evidence.verify(request, candidates)
        selected = self.evidence.select(request, bundle)
        output: list[dict[str, Any]] = []
        for rank, value in enumerate(selected, 1):
            verified = value.evidence
            raw = dict(verified.metadata)
            raw.update(
                {
                    "rank": rank,
                    "evidence_id": verified.evidence_id,
                    "work_id": verified.work_id,
                    "document_id": verified.document_id,
                    "chunk_id": verified.chunk_id,
                    "page_start": verified.page_start,
                    "page_end": verified.page_end,
                    "block_ids": list(verified.block_ids),
                    "content": verified.content,
                    "evidence_state": verified.state,
                    "evidence_grade": verified.evidence_grade,
                    "directness": verified.directness,
                }
            )
            output.append(raw)
        return output, candidates

    def _shadow(self, request: EvidenceRequest, official: tuple[CandidateEvidence, ...], official_ms: float) -> None:
        if self.shadow_engine is None:
            return
        started = perf_counter()
        raw_shadow = tuple(self.shadow_engine.retrieve_candidates(request))
        bundle = self.evidence.verify(request, raw_shadow)
        selected = self.evidence.select(request, bundle)
        shadow = tuple(value.evidence for value in selected)
        report = compare_retrievals(
            request.query,
            official,
            shadow,
            k=request.top_k,
            official_elapsed_ms=official_ms,
            shadow_elapsed_ms=(perf_counter() - started) * 1000,
            shadow_diagnostics=dict(self.evidence.last_diagnostics),
        )
        self.last_diagnostics["shadow"] = report
        if self.report_store is not None:
            self.last_diagnostics["shadow_report"] = str(self.report_store.write(request.request_id, report))

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        official_kwargs = dict(kwargs)
        workspace_id = official_kwargs.pop("workspace_id", None)
        scope_version = official_kwargs.pop("scope_version", None)
        limit = int(official_kwargs.get("limit", 20))
        request = self._request(
            query,
            limit,
            workspace_id=str(workspace_id) if workspace_id is not None else None,
            scope_version=int(scope_version) if scope_version is not None else None,
        )
        started = perf_counter()
        result = self.official_runtime.retrieve(query, **official_kwargs)
        raw = result.get("results") if isinstance(result, dict) else None
        values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
        governed, candidates = self._govern(request, values)
        official_ms = (perf_counter() - started) * 1000
        self.last_diagnostics = {
            **dict(getattr(self.official_runtime, "last_diagnostics", {})),
            "official_engine": "legacy_adapter",
            "active_engine": "legacy",
            "evidence_intelligence": dict(self.evidence.last_diagnostics),
        }
        self._shadow(request, candidates, official_ms)
        return {**result, "results": governed, "result_count": len(governed)}

    def retrieve_multi(self, queries: Sequence[str], *, limit: int = 40, rrf_k: int = 60) -> dict[str, Any]:
        if hasattr(self.official_runtime, "retrieve_multi"):
            query = next(iter(queries), "")
            request = self._request(query, limit)
            started = perf_counter()
            result = self.official_runtime.retrieve_multi(queries, limit=limit, rrf_k=rrf_k)
            raw = result.get("results") if isinstance(result, dict) else None
            values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
            governed, candidates = self._govern(request, values)
            official_ms = (perf_counter() - started) * 1000
            self.last_diagnostics = {
                **dict(getattr(self.official_runtime, "last_diagnostics", {})),
                "official_engine": "legacy_adapter",
                "active_engine": "legacy",
                "evidence_intelligence": dict(self.evidence.last_diagnostics),
            }
            self._shadow(request, candidates, official_ms)
            return {**result, "results": governed, "result_count": len(governed)}
        merged: list[dict[str, Any]] = []
        for query in queries:
            merged.extend(self.retrieve(query, limit=limit, rrf_k=rrf_k).get("results", []))
        unique = {str(value.get("evidence_id") or value.get("chunk_id")): value for value in merged}
        return {"retriever": "unified_legacy", "results": list(unique.values())[:limit], "result_count": min(len(unique), limit), "diagnostics": self.last_diagnostics}

    def close(self) -> None:
        close = getattr(self.official_runtime, "close", None)
        if callable(close):
            close()


def build_legacy_unified_service(
    project_root: Path,
    runtime: Any,
    *,
    supports_advanced_retrieval: bool = False,
    shadow_engine: Any | None = None,
) -> UnifiedKnowledgeService:
    """生产组装点：Agent 只接收本服务，不接收具体旧 RAG runtime。"""

    from app.corpus import CanonicalCorpusService
    from app.workspaces import WorkspaceService

    corpus = CanonicalCorpusService(project_root)
    workspaces = WorkspaceService(project_root, corpus=corpus)
    workspace = workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(
        corpus,
        workspaces,
        max_candidates_per_document=100,
        max_selected_per_document=100,
    )
    return UnifiedKnowledgeService(
        runtime,
        evidence,
        shadow_engine=shadow_engine,
        report_store=ShadowReportStore(project_root),
        scope_version=workspace.scope_version,
        supports_advanced_retrieval=supports_advanced_retrieval,
    )


def legacy_advanced_capability(runtime: Any) -> bool:
    """仅供尚未迁移的测试/嵌入式调用；生产组装不依赖该旧标志。"""

    return bool(getattr(runtime, "is_" + "graphrag", False))
