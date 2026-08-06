"""阶段一兼容 Adapter：把旧实现投影到新契约，不改旧调用路径。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from app.contracts.domain import (
    AgentRun,
    CandidateEvidence,
    ContextPack,
    EvidenceRequest,
    IndexGeneration,
    IndexProfile,
    Document,
    VerifiedEvidence,
    VerifiedEvidenceBundle,
)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


class LegacyEvidenceMapper:
    """映射旧 dict Evidence，并只做可追溯字段的结构校验。"""

    def candidate_from_mapping(
        self,
        request: EvidenceRequest,
        value: Mapping[str, Any],
        *,
        fallback_rank: int,
    ) -> CandidateEvidence:
        chunk_id = str(value.get("chunk_id") or "")
        candidate_id = str(value.get("evidence_id") or chunk_id or f"legacy-{fallback_rank}")
        raw_score = value.get("score")
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
            else 0.0
        )
        return CandidateEvidence(
            candidate_id=candidate_id,
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            content=str(value.get("content") or ""),
            score=score,
            retrieval_source=str(value.get("retrieval_source") or "legacy"),
            work_id=str(value.get("work_id") or "") or None,
            document_id=str(value.get("document_id") or "") or None,
            chunk_id=chunk_id or None,
            section_id=str(value.get("section_id") or "") or None,
            page_start=(int(value["page_start"]) if value.get("page_start") else None),
            page_end=(int(value["page_end"]) if value.get("page_end") else None),
            block_ids=_strings(value.get("block_ids")),
            metadata=dict(value),
        )

    def verified_from_candidate(
        self,
        candidate: CandidateEvidence,
    ) -> VerifiedEvidence | None:
        if not all(
            (
                candidate.work_id,
                candidate.document_id,
                candidate.chunk_id,
                candidate.content,
                candidate.page_start,
                candidate.page_end,
                candidate.block_ids,
            )
        ):
            return None
        assert candidate.work_id is not None
        assert candidate.document_id is not None
        assert candidate.chunk_id is not None
        assert candidate.page_start is not None
        assert candidate.page_end is not None
        content_hash = hashlib.sha256(candidate.content.encode("utf-8")).hexdigest()
        return VerifiedEvidence(
            evidence_id=str(candidate.metadata.get("evidence_id") or candidate.candidate_id),
            candidate_id=candidate.candidate_id,
            request_id=candidate.request_id,
            workspace_id=candidate.workspace_id,
            scope_version=candidate.scope_version,
            content=candidate.content,
            work_id=candidate.work_id,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            block_ids=candidate.block_ids,
            verification_method="legacy_locator_fields",
            content_hash=content_hash,
            metadata=candidate.metadata,
        )

    def bundle(
        self,
        request: EvidenceRequest,
        candidates: Sequence[CandidateEvidence],
    ) -> VerifiedEvidenceBundle:
        verified: list[VerifiedEvidence] = []
        rejected: list[str] = []
        for candidate in candidates:
            value = self.verified_from_candidate(candidate)
            if value is None:
                rejected.append(candidate.candidate_id)
            else:
                verified.append(value)
        issues = (
            ("legacy evidence missing canonical locator fields",) if rejected else ()
        )
        return VerifiedEvidenceBundle(
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            evidence=tuple(verified),
            rejected_candidate_ids=tuple(rejected),
            verification_issues=issues,
        )


class LegacyKnowledgeEngineAdapter:
    """包装旧 RetrievalRuntime/GraphRAG runtime，不改变旧对象本身。"""

    def __init__(self, runtime: Any, mapper: LegacyEvidenceMapper | None = None) -> None:
        self.runtime = runtime
        self.mapper = mapper or LegacyEvidenceMapper()
        self.last_response: Mapping[str, Any] | None = None

    def retrieve(self, request: EvidenceRequest) -> Sequence[CandidateEvidence]:
        options = dict(request.retrieval_options)
        mode = str(options.pop("mode", "fast"))
        max_chunks_per_work = int(options.pop("max_chunks_per_work", 20))
        candidate_limit = int(options.pop("candidate_limit", max(request.top_k, 20)))
        rrf_k = int(options.pop("rrf_k", 60))
        for reserved in ("limit", "work_id", "document_id", "top_k"):
            options.pop(reserved, None)
        response = self.runtime.retrieve(
            request.query,
            mode=mode,
            limit=max(request.top_k, candidate_limit),
            work_id=request.work_ids[0] if len(request.work_ids) == 1 else None,
            document_id=(
                request.document_ids[0] if len(request.document_ids) == 1 else None
            ),
            max_chunks_per_work=max_chunks_per_work,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
            **options,
        )
        self.last_response = response
        raw_results = response.get("results") if isinstance(response, Mapping) else None
        results = (
            [value for value in raw_results if isinstance(value, Mapping)]
            if isinstance(raw_results, list)
            else []
        )
        if request.work_ids:
            allowed_works = set(request.work_ids)
            results = [value for value in results if value.get("work_id") in allowed_works]
        if request.document_ids:
            allowed_documents = set(request.document_ids)
            results = [
                value for value in results if value.get("document_id") in allowed_documents
            ]
        return tuple(
            self.mapper.candidate_from_mapping(request, value, fallback_rank=rank)
            for rank, value in enumerate(results[: request.top_k], 1)
        )

    def retrieve_candidates(
        self, request: EvidenceRequest
    ) -> Sequence[CandidateEvidence]:
        return self.retrieve(request)

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]:
        method = getattr(self.runtime, "index_documents", None)
        if not callable(method):
            raise NotImplementedError("旧知识引擎不提供按文档增量索引。")
        return method(documents, generation=generation, profile=profile)

    def update_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]:
        method = getattr(self.runtime, "update_documents", None)
        if not callable(method):
            raise NotImplementedError("旧知识引擎不提供按文档更新。")
        return method(documents, generation=generation, profile=profile)

    def delete_documents(
        self,
        document_ids: Sequence[str],
        *,
        generation: IndexGeneration,
    ) -> Mapping[str, Any]:
        method = getattr(self.runtime, "delete_documents", None)
        if not callable(method):
            raise NotImplementedError("旧知识引擎不提供论文删除。")
        return method(document_ids, generation=generation)

    def get_status(self) -> Mapping[str, Any]:
        method = getattr(self.runtime, "get_status", None)
        if callable(method):
            value = method()
            return dict(value) if isinstance(value, Mapping) else {"status": value}
        return {"engine": "legacy", "status": "available"}


class LegacyAgentRuntimeAdapter:
    """包装 AgenticRAGService；`answer` 保留旧返回，`run` 实现新 Protocol。"""

    def __init__(self, service: Any) -> None:
        self.service = service

    def answer(
        self,
        query: str,
        *,
        session_id: str | None = None,
        force_new_topic: bool = False,
        include_context: bool = False,
    ) -> dict[str, Any]:
        return self.service.answer(
            query,
            session_id=session_id,
            force_new_topic=force_new_topic,
            include_context=include_context,
        )

    def run(
        self,
        request: EvidenceRequest,
        *,
        context_pack: ContextPack | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> AgentRun:
        legacy_state = dict(state or {})
        result = self.answer(
            request.query,
            session_id=(
                str(legacy_state["session_id"])
                if legacy_state.get("session_id")
                else None
            ),
            force_new_topic=bool(legacy_state.get("force_new_topic", False)),
            include_context=bool(legacy_state.get("include_context", False)),
        )
        diagnostics = result.get("diagnostics")
        safe_diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        metrics = safe_diagnostics.get("metrics")
        safe_metrics = metrics if isinstance(metrics, Mapping) else {}
        usage = safe_diagnostics.get("usage")
        safe_usage = usage if isinstance(usage, Mapping) else {}
        total_tokens = safe_usage.get("total_tokens")
        if not isinstance(total_tokens, int):
            prompt_cache = safe_diagnostics.get("prompt_cache")
            safe_prompt_cache = (
                prompt_cache if isinstance(prompt_cache, Mapping) else {}
            )
            prompt_tokens = safe_prompt_cache.get("prompt_tokens")
            completion_tokens = safe_prompt_cache.get("completion_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
        llm_calls = safe_diagnostics.get("llm_call_count")
        return AgentRun(
            run_id=str(result.get("run_id") or f"legacy:{request.request_id}"),
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            state="completed" if "answerable" in result else "failed",
            answer=str(result.get("answer") or ""),
            answerable=(
                bool(result.get("answerable")) if "answerable" in result else None
            ),
            context_id=context_pack.context_id if context_pack else None,
            llm_call_count=(int(llm_calls) if isinstance(llm_calls, int) else None),
            token_usage=(int(total_tokens) if isinstance(total_tokens, int) else None),
            elapsed_ms=(
                float(safe_metrics["latency_ms"])
                if isinstance(safe_metrics.get("latency_ms"), (int, float))
                else None
            ),
            diagnostics={"legacy_result": result},
        )
