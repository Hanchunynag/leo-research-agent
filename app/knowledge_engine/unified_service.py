"""Agent 可见的唯一 Unified Knowledge Service。

迁移期开关固定为 legacy official + optional LightRAG shadow。旧后端判断只封装在
该兼容边界内，不泄漏到 Agent/Harness。
"""

from __future__ import annotations

import secrets
from time import perf_counter
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        official_engine_name: str = "legacy",
        official_engine: Any | None = None,
        official_generation_id: str | None = None,
        shadow_engine: Any | None = None,
        shadow_engine_name: str | None = None,
        shadow_generation_id: str | None = None,
        report_store: ShadowReportStore | None = None,
        workspace_id: str = "default",
        scope_version: int = 1,
        supports_advanced_retrieval: bool = False,
    ) -> None:
        self.official_runtime = official_runtime
        if official_engine_name not in {"legacy", "lightrag"}:
            raise ValueError(f"未知 Official Engine：{official_engine_name}")
        if official_engine_name == "lightrag" and (
            official_engine is None or not official_generation_id
        ):
            raise ValueError("LightRAG Official 必须提供 Engine 和 Generation Pin。")
        self.official_engine_name = official_engine_name
        self.official_engine = official_engine
        self.official_generation_id = official_generation_id
        self.evidence = evidence
        self.shadow_engine = shadow_engine
        self.shadow_engine_name = shadow_engine_name or (
            "lightrag" if shadow_engine is not None else "none"
        )
        self.shadow_generation_id = shadow_generation_id
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

    def _shadow(self, request: EvidenceRequest, official: tuple[CandidateEvidence, ...], official_ms: float) -> None:
        if self.shadow_engine is None:
            return
        started = perf_counter()
        if hasattr(self.shadow_engine, "retrieve_candidates"):
            raw_shadow = tuple(self.shadow_engine.retrieve_candidates(request))
        elif hasattr(self.shadow_engine, "retrieve"):
            raw = self.shadow_engine.retrieve(request.query, limit=request.top_k)
            values = raw.get("results", []) if isinstance(raw, dict) else []
            raw_shadow = tuple(
                self.mapper.candidate_from_mapping(request, value, fallback_rank=index)
                for index, value in enumerate(values, 1)
                if isinstance(value, dict)
            )
        else:
            raise RuntimeError("Shadow Engine 不支持 retrieve_candidates/retrieve。")
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
        if self.official_engine_name == "lightrag":
            assert self.official_engine is not None
            if getattr(self.official_engine, "serving_generation_id", None) != self.official_generation_id:
                raise RuntimeError("LightRAG Engine 与 Official Generation Pin 不一致。")
            candidates = tuple(self.official_engine.retrieve_candidates(request))
            governed, candidates = self._govern_candidates(request, candidates)
            result: dict[str, Any] = {
                "retriever": "lightrag",
                "results": governed,
                "result_count": len(governed),
            }
        else:
            result = self.official_runtime.retrieve(query, **official_kwargs)
            raw = result.get("results") if isinstance(result, dict) else None
            values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
            governed, candidates = self._govern(request, values)
        official_ms = (perf_counter() - started) * 1000
        self.last_diagnostics = {
            **dict(getattr(self.official_runtime, "last_diagnostics", {})),
            "official_engine": self.official_engine_name,
            "official_generation_id": self.official_generation_id,
            "active_engine": self.official_engine_name,
            "evidence_intelligence": dict(self.evidence.last_diagnostics),
        }
        self._shadow(request, candidates, official_ms)
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
            return {"retriever": f"unified_{self.official_engine_name}", "results": [], "result_count": 0}
        normalized_filters = dict(paper_filters or {})
        if (
            self.official_engine_name == "legacy"
            and hasattr(self.official_runtime, "retrieve_multi")
            and not normalized_filters
        ):
            query = normalized_queries[0]
            request = self._request(
                query,
                limit,
                workspace_id=workspace_id,
                scope_version=scope_version,
            )
            started = perf_counter()
            result = self.official_runtime.retrieve_multi(normalized_queries, limit=limit, rrf_k=rrf_k)
            raw = result.get("results") if isinstance(result, dict) else None
            values = [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []
            governed, candidates = self._govern(request, values)
            official_ms = (perf_counter() - started) * 1000
            self.last_diagnostics = {
                **dict(getattr(self.official_runtime, "last_diagnostics", {})),
                "official_engine": "legacy",
                "official_generation_id": None,
                "active_engine": "legacy",
                "evidence_intelligence": dict(self.evidence.last_diagnostics),
            }
            self._shadow(request, candidates, official_ms)
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


def build_configured_unified_service(
    project_root: Path,
    legacy_runtime: Any,
    completion_provider: Any,
    *,
    llm_model_name: str,
) -> UnifiedKnowledgeService:
    """生产组装点：严格按审计配置和 Generation Pin 选择引擎。"""

    from app.corpus import CanonicalCorpusService
    from app.knowledge_engine.generations import IndexGenerationRepository
    from app.knowledge_engine.lightrag_engine import LightRAGKnowledgeEngine
    from app.knowledge_engine.model_bridge import build_lightrag_client_config
    from app.knowledge_engine.serving import KnowledgeServingConfigRepository
    from app.workspaces import WorkspaceService

    root = project_root.expanduser().resolve()
    serving = KnowledgeServingConfigRepository(root)
    config = serving.load()
    generations = IndexGenerationRepository(root)
    corpus = CanonicalCorpusService(root)
    workspaces = WorkspaceService(root, corpus=corpus)
    workspace = workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(
        corpus,
        workspaces,
        max_candidates_per_document=100,
        max_selected_per_document=100,
    )

    def require_pin(generation_id: str | None) -> Any:
        if not generation_id:
            raise ValueError("LightRAG 服务配置缺少 Generation Pin。")
        generation = generations.get(generation_id)
        if generation is None or generation.state not in {"active", "retired"}:
            raise ValueError(f"Pinned Generation 不可服务：{generation_id}")
        if generation.workspace_id != workspace.workspace_id:
            raise PermissionError("Pinned Generation workspace 不匹配。")
        return generation

    client_config = None

    def lightrag(generation_id: str | None) -> Any:
        nonlocal client_config
        require_pin(generation_id)
        if client_config is None:
            client_config, _ = build_lightrag_client_config(
                legacy_runtime.embedding_provider,
                completion_provider,
                llm_model_name=llm_model_name,
            )
        return LightRAGKnowledgeEngine(
            root,
            corpus=corpus,
            workspaces=workspaces,
            generations=generations,
            client_config=client_config,
            serving_generation_id=generation_id,
        )

    official_engine = (
        lightrag(config.official_generation_id)
        if config.official_engine == "lightrag"
        else None
    )
    if config.official_engine == "lightrag":
        latest = serving.audit_records()[-1:] or ()
        if not latest:
            raise PermissionError("LightRAG Official 缺少 Cutover 审计。")
        record = latest[0]
        details = record.get("details")
        target = record.get("to")
        accepted_operation = record.get("operation") == "rollback_previous" or (
            record.get("operation") == "cutover"
            and isinstance(details, dict)
            and details.get("acceptance_passed") is True
        )
        if (
            not accepted_operation
            or not isinstance(target, dict)
            or target.get("official_engine") != "lightrag"
            or target.get("official_generation_id")
            != config.official_generation_id
        ):
            raise PermissionError("LightRAG Official 配置没有匹配的合格审计记录。")
    shadow: Any | None = None
    if config.shadow_engine == "lightrag":
        shadow = lightrag(config.shadow_generation_id)
    elif config.shadow_engine == "legacy":
        shadow = legacy_runtime
    return UnifiedKnowledgeService(
        legacy_runtime,
        evidence,
        official_engine_name=config.official_engine,
        official_engine=official_engine,
        official_generation_id=config.official_generation_id,
        shadow_engine=shadow,
        shadow_engine_name=config.shadow_engine,
        shadow_generation_id=config.shadow_generation_id,
        report_store=ShadowReportStore(root),
        workspace_id=workspace.workspace_id,
        scope_version=workspace.scope_version,
        supports_advanced_retrieval=config.official_engine == "lightrag",
    )


def legacy_advanced_capability(runtime: Any) -> bool:
    """仅供尚未迁移的测试/嵌入式调用；生产组装不依赖该旧标志。"""

    return bool(getattr(runtime, "is_" + "graphrag", False))
