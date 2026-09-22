"""Agent 可见的唯一 Unified Knowledge Service。

检索实现固定为本地 Hybrid RAG：BM25 与 BGE-M3 Dense 通过 RRF 融合，随后
由 Cross-Encoder 和 Evidence Governance 完成精排、校验与选择。上层 Agent
只依赖本服务，不接触任何索引客户端或存储细节。
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.contracts import CandidateEvidence, EvidenceRequest
from app.contracts.adapters import EvidenceMapper
from app.evidence import EvidenceIntelligencePipeline


class UnifiedKnowledgeService:
    def __init__(
        self,
        official_runtime: Any,
        evidence: EvidenceIntelligencePipeline,
        *,
        workspace_id: str = "default",
        scope_version: int = 1,
        supports_advanced_retrieval: bool = False,
    ) -> None:
        self.official_runtime = official_runtime
        self.evidence = evidence
        self.workspace_id = workspace_id
        self.scope_version = scope_version
        self.supports_advanced_retrieval = supports_advanced_retrieval
        self.embedding_provider = official_runtime.embedding_provider
        self.reranker_provider = getattr(official_runtime, "reranker_provider", None)
        self.mapper = EvidenceMapper()
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
        return self._govern_candidates(request, candidates)

    def _govern_candidates(
        self,
        request: EvidenceRequest,
        candidates: Sequence[CandidateEvidence],
    ) -> tuple[list[dict[str, Any]], tuple[CandidateEvidence, ...]]:
        candidates = tuple(candidates)
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
        result = self.official_runtime.retrieve(query, **official_kwargs)
        raw = result.get("results") if isinstance(result, dict) else None
        values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
        governed, _ = self._govern(request, values)
        self.last_diagnostics = {
            **dict(getattr(self.official_runtime, "last_diagnostics", {})),
            "retrieval_backend": (
                "hierarchical"
                if str(result.get("retriever") or "").startswith("hierarchical")
                else "hybrid"
            ),
            "evidence_intelligence": dict(self.evidence.last_diagnostics),
        }
        return {**result, "results": governed, "result_count": len(governed)}

    def retrieve_multi(
        self,
        queries: Sequence[str],
        *,
        limit: int = 40,
        rrf_k: int = 60,
        workspace_id: str | None = None,
        scope_version: int | None = None,
        paper_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_queries = tuple(
            dict.fromkeys(
                value.strip() for value in queries if isinstance(value, str) and value.strip()
            )
        )
        if not normalized_queries:
            return {"retriever": "local_hybrid", "results": [], "result_count": 0}
        normalized_filters = dict(paper_filters or {})
        if (
            hasattr(self.official_runtime, "retrieve_multi")
            and not normalized_filters
        ):
            query = normalized_queries[0]
            request = self._request(
                query,
                limit,
                workspace_id=workspace_id,
                scope_version=scope_version,
            )
            result = self.official_runtime.retrieve_multi(normalized_queries, limit=limit, rrf_k=rrf_k)
            raw = result.get("results") if isinstance(result, dict) else None
            values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
            governed, candidates = self._govern(request, values)
            self.last_diagnostics = {
                **dict(getattr(self.official_runtime, "last_diagnostics", {})),
                "retrieval_backend": (
                    "hierarchical"
                    if str(result.get("retriever") or "").startswith("hierarchical")
                    else "hybrid"
                ),
                "evidence_intelligence": dict(self.evidence.last_diagnostics),
            }
            return {**result, "results": governed, "result_count": len(governed)}
        merged: list[dict[str, Any]] = []
        per_query: list[dict[str, Any]] = []
        paper_candidates: dict[str, dict[str, Any]] = {}
        candidate_paper_ids: list[str] = []
        paper_retrieval: list[dict[str, Any]] = []
        chunk_retrieval: list[dict[str, Any]] = []
        no_hit_reasons: list[str] = []
        for query_index, query in enumerate(normalized_queries):
            # Run the primary query through the configured Cross Encoder.  The
            # additional bilingual/title probes are bounded recall probes and
            # use the existing fast RRF path; re-ranking every probe would
            # multiply cold-start and CPU cost without improving final evidence
            # governance, which reranks the merged primary candidate pool.
            query_result = self.retrieve(
                query,
                mode="hierarchical" if query_index == 0 else "fast",
                limit=limit,
                rrf_k=rrf_k,
                paper_filters=normalized_filters,
                workspace_id=workspace_id,
                scope_version=scope_version,
            )
            per_query.append(query_result)
            merged.extend(query_result.get("results", []))
            for paper in query_result.get("candidate_papers", []):
                if isinstance(paper, dict) and paper.get("paper_id"):
                    paper_candidates.setdefault(str(paper["paper_id"]), dict(paper))
            for paper_id in query_result.get("candidate_paper_ids", []):
                if str(paper_id) and str(paper_id) not in candidate_paper_ids:
                    candidate_paper_ids.append(str(paper_id))
            if isinstance(query_result.get("paper_retrieval"), dict):
                paper_retrieval.append(dict(query_result["paper_retrieval"]))
            if isinstance(query_result.get("chunk_retrieval"), dict):
                chunk_retrieval.append(dict(query_result["chunk_retrieval"]))
            reason = query_result.get("no_hit_reason")
            if isinstance(reason, str) and reason:
                no_hit_reasons.append(reason)
        unique: dict[str, dict[str, Any]] = {}
        for value in merged:
            key = str(value.get("evidence_id") or value.get("chunk_id") or "")
            if key:
                # Keep the primary (Cross-Encoder-ranked) representation when
                # a later bilingual/title probe returns the same evidence.
                unique.setdefault(key, value)
        result: dict[str, Any] = {
            "retriever": "hierarchical",
            "results": list(unique.values())[:limit],
            "result_count": min(len(unique), limit),
            "candidate_papers": list(paper_candidates.values()),
            "candidate_paper_ids": candidate_paper_ids,
            "paper_retrieval": paper_retrieval[0] if len(paper_retrieval) == 1 else {
                "queries": paper_retrieval,
                "query_count": len(paper_retrieval),
            },
            "chunk_retrieval": {
                "queries": chunk_retrieval,
                "query_count": len(chunk_retrieval),
                "allowed_paper_ids": candidate_paper_ids,
                "rrf_k": rrf_k,
            },
            "no_hit_reason": (
                no_hit_reasons[0]
                if not merged and no_hit_reasons and len(set(no_hit_reasons)) == 1
                else ("multiple_query_no_hit" if not merged and no_hit_reasons else None)
            ),
            "diagnostics": self.last_diagnostics,
        }
        return result

    def close(self) -> None:
        close = getattr(self.official_runtime, "close", None)
        if callable(close):
            close()


def build_knowledge_service(
    project_root: Path,
    runtime: Any,
    *,
    supports_advanced_retrieval: bool = False,
) -> UnifiedKnowledgeService:
    """Build the single local hybrid retrieval service used by Research Agent."""

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
        scope_version=workspace.scope_version,
        supports_advanced_retrieval=supports_advanced_retrieval,
    )


def knowledge_runtime_status(project_root: Path) -> dict[str, Any]:
    """Return the public, backend-independent retrieval status."""

    root = project_root.expanduser().resolve()
    from app.knowledge.corpus import knowledge_index_readiness

    index_status = knowledge_index_readiness(root)
    return {
        "retrieval_backend": "hierarchical",
        "retrieval_components": [
            "paper_level_bm25",
            "paper_level_bge_m3_dense",
            "paper_level_rrf",
            "per_paper_content_bm25",
            "per_paper_content_bge_m3_dense",
            "per_paper_content_rrf",
            "cross_encoder_reranker",
            "evidence_governance",
        ],
        "dense_index_present": (root / "data" / "index" / "qdrant_dense").is_dir(),
        "paper_dense_index_present": (
            root / "data" / "index" / "qdrant_papers_dense"
        ).is_dir(),
        "knowledge_index": index_status,
    }
