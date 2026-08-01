"""Session-aware Agentic Scientific RAG 的有界端到端编排。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from time import perf_counter
from typing import Any, Sequence

from app.agentic.config import AgenticRAGConfig
from app.agentic.coverage import (
    deterministic_coverage,
    deterministic_repair,
    deterministic_semantic_validation,
)
from app.agentic.models import (
    AgenticAnswerDraft,
    AgenticMetricsRecord,
    CoverageReport,
    EvidenceStatus,
    QueryPlan,
    RetrievalRound,
    RoutingLLMDecision,
    SemanticValidationReport,
)
from app.agentic.planning import QueryPlanner
from app.agentic.prompting import (
    build_topic_messages,
    compact_topic,
    prompt_cache_diagnostics,
)
from app.agentic.provider import AgenticReasoningProvider
from app.agentic.reranking import DirectAnswerReranker
from app.agentic.routing import TopicRouter, extract_entities, rewrite_standalone_query
from app.agentic.store import AgenticSessionStore, stable_json
from app.context.assembly import assemble_context_bundle
from app.context.models import ContextBundle
from app.generation.models import (
    AnswerClaim,
    AnswerDraft,
    CitationValidationIssue,
    CitationValidationReport,
    GroundedAnswer,
)
from app.generation.validation import validate_answer_draft
from app.indexing.tokenization import token_count
from app.indexing.tokenization import normalize_search_text
from app.runtime.retrieval import RetrievalRuntime


def _configuration_fingerprint(config: AgenticRAGConfig) -> str:
    payload = asdict(config)
    if payload.get("session_db_path") is not None:
        payload["session_db_path"] = str(payload["session_db_path"])
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _agentic_to_answer_draft(draft: AgenticAnswerDraft) -> AnswerDraft:
    return AnswerDraft(
        answerable=draft.answerable,
        claims=[
            AnswerClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                source_ids=claim.source_ids,
                category=claim.category,
                evidence_ids=claim.evidence_ids,
            )
            for claim in draft.claims
        ],
        refusal_reason=draft.refusal_reason,
    )


def _render_answer(draft: AgenticAnswerDraft) -> str:
    return "\n".join(
        f"{claim.text.strip()} "
        + "".join(f"[{source_id}]" for source_id in claim.source_ids)
        for claim in draft.claims
    )


def _merge_candidates(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for value in values:
        chunk_id = str(value.get("chunk_id") or "")
        if not chunk_id:
            continue
        existing = merged.get(chunk_id)
        if existing is None or int(value.get("rank") or 10**9) < int(
            existing.get("rank") or 10**9
        ):
            merged[chunk_id] = dict(value)
    return sorted(
        merged.values(),
        key=lambda item: (
            int(item.get("rank") or 10**9),
            str(item.get("chunk_id") or ""),
        ),
    )


class AgenticRAGService:
    """执行路由、规划、增量检索、覆盖、生成、语义验证和状态追加。"""

    def __init__(
        self,
        retrieval_runtime: RetrievalRuntime,
        reasoning_provider: AgenticReasoningProvider,
        session_store: AgenticSessionStore,
        reranker: DirectAnswerReranker,
        config: AgenticRAGConfig,
    ) -> None:
        self.runtime = retrieval_runtime
        self.reasoning_provider = reasoning_provider
        self.store = session_store
        self.reranker = reranker
        self.config = config
        self.planner = QueryPlanner()
        self._stage_fallbacks: list[str] = []
        self.router = TopicRouter(
            retrieval_runtime.embedding_provider,
            config,
            ambiguity_resolver=self._resolve_ambiguous_route,
        )

    def _resolve_ambiguous_route(
        self,
        standalone_query: str,
        topic_summary: str,
        signals: dict[str, float],
    ) -> RoutingLLMDecision:
        try:
            return self.reasoning_provider.resolve_route(
                standalone_query,
                topic_summary,
                signals,
            )
        except Exception:
            self._stage_fallbacks.append("topic_router")
            return RoutingLLMDecision(
                relation="related_subtopic",
                confidence=0.5,
                reason="Router LLM 不可用，模糊区间保守创建相关子主题。",
                context_dependent=False,
                standalone_query=standalone_query,
                reuse_previous_evidence=False,
                requires_new_retrieval=True,
            )

    def _retrieve_queries(self, queries: Sequence[str]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for query in list(dict.fromkeys(value.strip() for value in queries if value.strip())):
            result = self.runtime.retrieve(
                query,
                mode="fast",
                limit=self.config.candidate_limit,
                max_chunks_per_work=20,
                candidate_limit=self.config.candidate_limit,
                rrf_k=self.config.rrf_k,
            )
            raw = result.get("results")
            if isinstance(raw, list):
                values.extend(item for item in raw if isinstance(item, dict))
        return _merge_candidates(values)[: self.config.candidate_limit]

    def _relevant_registry(
        self,
        query: str,
        registry: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """以实体与词面重合筛选已加载证据，避免无条件注入整个 Topic。"""

        normalized_query = normalize_search_text(query)
        query_terms = {
            term for term in normalized_query.split() if len(term) >= 2
        }
        query_entities = set(extract_entities(query))
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item in registry:
            content = normalize_search_text(
                " ".join(
                    [
                        str(item.get("title") or ""),
                        " ".join(item.get("section_path") or []),
                        str(item.get("content") or ""),
                    ]
                )
            )
            entity_overlap = len(query_entities & set(extract_entities(content)))
            term_overlap = sum(term in content for term in query_terms)
            score = float(entity_overlap * 10 + term_overlap)
            if score > 0:
                scored.append((score, str(item.get("evidence_id") or ""), dict(item)))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return [
            dict(item, origin="reused")
            for _, _, item in scored[: self.config.rerank_top_k]
        ]

    def _merge_fresh_and_reused(
        self,
        fresh: Sequence[dict[str, Any]],
        reused: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """为相关历史证据预留候选位，并按 chunk_id 去重。"""

        reused_ids = {str(item.get("chunk_id") or "") for item in reused}
        unique_fresh = [
            item
            for item in _merge_candidates(fresh)
            if str(item.get("chunk_id") or "") not in reused_ids
        ]
        fresh_limit = max(0, self.config.candidate_limit - len(reused))
        return _merge_candidates([*unique_fresh[:fresh_limit], *reused])[
            : self.config.candidate_limit
        ]

    def _coverage(
        self,
        plan: QueryPlan,
        evidence: Sequence[dict[str, Any]],
    ) -> CoverageReport:
        known_ids = {str(item.get("evidence_id")) for item in evidence}
        try:
            report = self.reasoning_provider.check_coverage(plan, evidence)
            expected_subquestions = {item.id for item in plan.subquestions}
            returned_subquestions = {
                item.subquestion_id for item in report.coverage
            }
            if returned_subquestions != expected_subquestions:
                raise ValueError("Coverage 未逐一返回全部 subquestion。")
            returned_ids = {
                evidence_id
                for item in report.coverage
                for evidence_id in item.supporting_evidence_ids
            }
            if not returned_ids.issubset(known_ids):
                raise ValueError("Coverage 返回未知 evidence_id。")
            computed_sufficient = all(
                item.status == "sufficient" for item in report.coverage
            )
            if report.overall_sufficient != computed_sufficient:
                raise ValueError("Coverage 的 overall_sufficient 与逐项状态不一致。")
            return report
        except Exception:
            self._stage_fallbacks.append("coverage")
            return deterministic_coverage(plan, evidence)

    def _semantic_validate(
        self,
        query: str,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        evidence: Sequence[dict[str, Any]],
        structural_valid: bool,
    ) -> SemanticValidationReport:
        if not self.config.semantic_validation_enabled:
            return SemanticValidationReport(
                valid=structural_valid,
                issues=["semantic_validation_disabled"],
                structural_valid=structural_valid,
                semantic_valid=structural_valid,
                claim_results=[],
                requires_retrieval=False,
                followup_queries=[],
            )
        try:
            report = self.reasoning_provider.validate_semantic(
                query,
                plan,
                draft,
                evidence,
            )
            if report.structural_valid != structural_valid:
                report = report.model_copy(update={"structural_valid": structural_valid})
            report = report.model_copy(
                update={"valid": structural_valid and report.semantic_valid}
            )
            expected_claim_ids = {claim.claim_id for claim in draft.claims}
            returned_claim_ids = {
                result.claim_id for result in report.claim_results
            }
            if returned_claim_ids != expected_claim_ids or len(
                report.claim_results
            ) != len(draft.claims):
                raise ValueError("Semantic Validator 未逐一返回全部 Claim。")
            computed_semantic_valid = bool(report.claim_results) and all(
                item.entailment == "entailed"
                and item.query_aligned
                and item.category_correct
                and item.citation_direct
                for item in report.claim_results
            )
            if report.semantic_valid != computed_semantic_valid:
                raise ValueError("Semantic Validator 汇总状态与 Claim 结果不一致。")
            guardrail = deterministic_semantic_validation(
                query,
                plan,
                draft,
                evidence,
                structural_valid=structural_valid,
            )
            if not guardrail.valid:
                self._stage_fallbacks.append("semantic_category_guardrail")
                return guardrail
            return report
        except Exception:
            self._stage_fallbacks.append("semantic_validation")
            return deterministic_semantic_validation(
                query,
                plan,
                draft,
                evidence,
                structural_valid=structural_valid,
            )

    @staticmethod
    def _validate_evidence_mapping(
        draft: AgenticAnswerDraft,
        context: ContextBundle,
        structural: CitationValidationReport,
    ) -> CitationValidationReport:
        issues = list(structural.issues)
        mapping = {item.source_id: item.evidence_id for item in context.evidence}
        known_evidence = {value for value in mapping.values() if value}
        for claim in draft.claims:
            if not claim.evidence_ids:
                issues.append(
                    CitationValidationIssue(
                        "claim_without_evidence_id",
                        "Agentic Claim 必须引用稳定 evidence_id。",
                        claim_id=claim.claim_id,
                    )
                )
            if any(value not in known_evidence for value in claim.evidence_ids):
                issues.append(
                    CitationValidationIssue(
                        "unknown_evidence_id",
                        "Claim 引用了当前证据包不存在的 evidence_id。",
                        claim_id=claim.claim_id,
                    )
                )
            mapped = {mapping.get(source_id) for source_id in claim.source_ids}
            if not set(claim.evidence_ids).issubset(mapped):
                issues.append(
                    CitationValidationIssue(
                        "source_evidence_mismatch",
                        "source_id 与 evidence_id 映射不一致。",
                        claim_id=claim.claim_id,
                    )
                )
        return CitationValidationReport(
            valid=not issues,
            issues=issues,
            citations=structural.citations if not issues else [],
        )

    def _register_round_evidence(
        self,
        session_id: str,
        topic_id: str,
        turn_ordinal: int,
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        registered: list[dict[str, Any]] = []
        new_count = 0
        reused_count = 0
        for candidate in candidates:
            record, created = self.store.register_evidence(
                session_id,
                topic_id,
                candidate,
                turn_ordinal,
            )
            value = dict(candidate)
            value["evidence_id"] = record["evidence_id"]
            value["origin"] = "newly_retrieved" if created else "reused"
            registered.append(value)
            if created:
                new_count += 1
                self.store.append_event(
                    session_id,
                    topic_id,
                    "evidence_added",
                    {
                        "evidence_id": record["evidence_id"],
                        "chunk_id": record["chunk_id"],
                        "work_id": record.get("work_id"),
                        "document_id": record.get("document_id"),
                        "title": record.get("title"),
                        "section_path": record.get("section_path"),
                        "page_start": record.get("page_start"),
                        "page_end": record.get("page_end"),
                        "block_ids": record.get("block_ids"),
                        "content": record.get("content"),
                        "content_hash": record["content_hash"],
                    },
                )
            else:
                reused_count += 1
        return registered, new_count, reused_count

    def _refusal_result(
        self,
        query: str,
        context: ContextBundle,
        reason: str,
    ) -> GroundedAnswer:
        validation = CitationValidationReport(
            False,
            [CitationValidationIssue("insufficient_coverage", reason)],
            [],
        )
        return GroundedAnswer(
            query=query,
            answerable=False,
            answer="",
            claims=[],
            citations=[],
            refusal_reason=reason,
            validation=validation,
            context=context,
            diagnostics={},
        )

    def _final_evidence(
        self,
        coverage: CoverageReport,
        evidence_by_id: dict[str, dict[str, Any]],
        latest_reranked: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """优先选择 Coverage 直接证据，再按本轮精排补足最终上下文。"""

        supporting_ids = list(
            dict.fromkeys(
                evidence_id
                for item in coverage.coverage
                for evidence_id in item.supporting_evidence_ids
            )
        )
        ordered = [
            evidence_by_id[value]
            for value in supporting_ids
            if value in evidence_by_id
        ]
        ordered.extend(
            item
            for item in latest_reranked
            if str(item.get("evidence_id")) not in supporting_ids
        )
        selected = [dict(item) for item in ordered[: self.config.final_top_k]]
        for index, item in enumerate(selected, 1):
            item["source_id"] = f"S{index}"
        return selected

    def _build_context(
        self,
        query: str,
        evidence: Sequence[dict[str, Any]],
        round_count: int,
    ) -> ContextBundle:
        """使用现有 Context Builder 构造兼容 fast 输出的证据包。"""

        return assemble_context_bundle(
            query,
            "accurate" if self.config.reranker_enabled else "fast",
            list(evidence),
            token_budget=6000,
            max_evidence=self.config.final_top_k,
            max_evidence_per_work=2,
            retrieval_diagnostics={
                "retriever": "agentic_hybrid_rrf",
                "retrieval_rounds": round_count,
            },
        )

    def _append_source_mapping(
        self,
        session_id: str,
        topic_id: str,
        evidence: Sequence[dict[str, Any]],
        coverage: CoverageReport,
        plan: QueryPlan,
    ) -> None:
        """追加本次答案临时 S 编号到稳定 Evidence ID 的映射。"""

        self.store.append_event(
            session_id,
            topic_id,
            "state_delta",
            {
                "current_sources": [
                    {
                        "source_id": item["source_id"],
                        "evidence_id": item["evidence_id"],
                        "chunk_id": item["chunk_id"],
                        "origin": item.get("origin"),
                    }
                    for item in evidence
                ],
                "coverage": coverage.model_dump(mode="json"),
                "answer_constraints": plan.answer_constraints,
            },
        )

    def answer(
        self,
        query: str,
        *,
        session_id: str | None = None,
        force_new_topic: bool = False,
        include_context: bool = False,
    ) -> dict[str, Any]:
        """执行最多 N 轮检索和一次 Repair，并追加全部状态事件。"""

        started = perf_counter()
        compaction_count = 0
        self._stage_fallbacks = []
        configuration_fingerprint = _configuration_fingerprint(self.config)
        session, session_created = self.store.get_or_create_session(
            session_id,
            query,
            model=getattr(self.reasoning_provider, "model_name", None),
            provider=type(self.reasoning_provider).__name__,
            configuration_fingerprint=configuration_fingerprint,
        )
        sid = str(session["session_id"])
        current_topic = (
            self.store.get_topic(sid, str(session["active_topic_id"]))
            if session.get("active_topic_id")
            else None
        )
        previous_event_count = (
            len(self.store.list_events(sid, str(current_topic["topic_id"])))
            if current_topic is not None
            else 0
        )
        registry = (
            self.store.list_evidence(sid, str(current_topic["topic_id"]))
            if current_topic is not None
            else []
        )
        preliminary_query = (
            rewrite_standalone_query(query, str(current_topic["topic_summary"]))
            if current_topic is not None
            else query
        )
        preliminary = (
            self._retrieve_queries([preliminary_query])
            if current_topic is not None and not force_new_topic
            else []
        )
        route = self.router.route(
            query,
            current_topic,
            preliminary,
            registry,
            force_new_topic=force_new_topic,
        )
        if route.relation != "same_topic" or current_topic is None:
            parent = (
                str(current_topic["topic_id"])
                if current_topic is not None
                and route.relation == "related_subtopic"
                else None
            )
            topic = self.store.create_topic(
                sid,
                relation=route.relation,
                topic_summary=route.standalone_query,
                user_goal=route.standalone_query,
                entities=extract_entities(route.standalone_query),
                parent_topic_id=parent,
            )
            preliminary = []
            registry = []
            previous_event_count = 0
        else:
            topic = current_topic
            self.store.set_active_topic(sid, str(topic["topic_id"]))
        topic_id = str(topic["topic_id"])
        plan = self.planner.plan(route.standalone_query)
        user_event = self.store.append_event(
            sid,
            topic_id,
            "user_query",
            {"query": query},
        )
        self.store.append_event(
            sid,
            topic_id,
            "query_analysis",
            {
                "routing": route.model_dump(mode="json"),
                "query_plan": plan.model_dump(mode="json"),
            },
        )

        round_reports: list[RetrievalRound] = []
        reranker_diagnostics: list[dict[str, Any]] = []
        evidence_by_id: dict[str, dict[str, Any]] = {}
        total_new = 0
        total_reused = 0
        queries = list(plan.retrieval_queries)
        coverage = CoverageReport(
            overall_sufficient=False,
            coverage=[],
            followup_queries=queries,
        )
        last_reranked: list[dict[str, Any]] = []
        for round_number in range(1, self.config.max_retrieval_rounds + 1):
            fresh = [
                *(preliminary if round_number == 1 else []),
                *self._retrieve_queries(queries),
            ]
            reused = (
                self._relevant_registry(route.standalone_query, registry)
                if route.reuse_previous_evidence
                else []
            )
            retrieved = self._merge_fresh_and_reused(fresh, reused)
            reranked = self.reranker.rerank(
                route.standalone_query,
                retrieved,
                self.config.rerank_top_k,
                plan,
            )
            reranker_diagnostics.append(dict(self.reranker.last_diagnostics))
            registered, new_count, reused_count = self._register_round_evidence(
                sid,
                topic_id,
                int(user_event["ordinal"]),
                reranked,
            )
            for value in registered:
                evidence_by_id[str(value["evidence_id"])] = value
            total_new += new_count
            total_reused += reused_count
            last_reranked = registered
            coverage = self._coverage(plan, list(evidence_by_id.values()))
            status: EvidenceStatus = (
                "sufficient"
                if coverage.overall_sufficient
                else (
                    "partial"
                    if any(item.status != "missing" for item in coverage.coverage)
                    else "missing"
                )
            )
            round_reports.append(
                RetrievalRound(
                    round=round_number,
                    queries=list(queries),
                    candidate_count=len(retrieved),
                    reranked_count=len(registered),
                    new_evidence_count=new_count,
                    coverage_status=status,
                )
            )
            if coverage.overall_sufficient:
                break
            queries = coverage.followup_queries
            if not queries:
                break

        final_evidence = self._final_evidence(
            coverage,
            evidence_by_id,
            last_reranked,
        )
        context = self._build_context(
            route.standalone_query,
            final_evidence,
            len(round_reports),
        )
        if not coverage.overall_sufficient:
            reason = "; ".join(
                item.missing_information
                for item in coverage.coverage
                if item.status != "sufficient"
            ) or "达到最大检索轮数后证据覆盖仍不足。"
            answer = self._refusal_result(query, context, reason)
            semantic = SemanticValidationReport(
                valid=False,
                issues=[reason],
                structural_valid=False,
                semantic_valid=False,
                claim_results=[],
                requires_retrieval=False,
                followup_queries=[],
            )
            prompt_cache: dict[str, Any] = prompt_cache_diagnostics(
                [], new_message_count=0, usage=None
            )
            repair_used = False
        else:
            self._append_source_mapping(
                sid,
                topic_id,
                final_evidence,
                coverage,
                plan,
            )
            events = self.store.list_events(sid, topic_id)
            messages = build_topic_messages(events)
            threshold = int(
                self.config.model_context_window
                * self.config.context_compaction_threshold
            )
            if token_count(stable_json(messages)) >= threshold:
                compact_topic(
                    self.store,
                    sid,
                    topic_id,
                    recent_event_count=self.config.recent_events_after_compaction,
                )
                compaction_count += 1
                events = self.store.list_events(sid, topic_id)
                messages = build_topic_messages(events)
                previous_event_count = 0
            new_message_count = max(0, len(events) - previous_event_count)
            try:
                draft, generation_metadata = self.reasoning_provider.generate_answer(
                    messages
                )
            except Exception as error:
                self._stage_fallbacks.append("answer_generation")
                draft = AgenticAnswerDraft(
                    answerable=False,
                    claims=[],
                    refusal_reason=f"回答模型未返回合法结构：{type(error).__name__}",
                )
                generation_metadata = {}
            structural = validate_answer_draft(
                _agentic_to_answer_draft(draft),
                context,
            )
            structural = self._validate_evidence_mapping(draft, context, structural)
            semantic = self._semantic_validate(
                route.standalone_query,
                plan,
                draft,
                final_evidence,
                structural.valid,
            )
            if (
                draft.answerable
                and semantic.requires_retrieval
                and len(round_reports) < self.config.max_retrieval_rounds
            ):
                followup_queries = semantic.followup_queries or [
                    route.standalone_query
                ]
                retrieved = self._merge_fresh_and_reused(
                    self._retrieve_queries(followup_queries),
                    self._relevant_registry(
                        route.standalone_query,
                        list(evidence_by_id.values()),
                    ),
                )
                reranked = self.reranker.rerank(
                    route.standalone_query,
                    retrieved,
                    self.config.rerank_top_k,
                    plan,
                )
                reranker_diagnostics.append(dict(self.reranker.last_diagnostics))
                registered, new_count, reused_count = self._register_round_evidence(
                    sid,
                    topic_id,
                    int(user_event["ordinal"]),
                    reranked,
                )
                for value in registered:
                    evidence_by_id[str(value["evidence_id"])] = value
                total_new += new_count
                total_reused += reused_count
                last_reranked = registered
                coverage = self._coverage(plan, list(evidence_by_id.values()))
                semantic_round_status: EvidenceStatus = (
                    "sufficient"
                    if coverage.overall_sufficient
                    else "partial"
                )
                round_reports.append(
                    RetrievalRound(
                        round=len(round_reports) + 1,
                        queries=list(followup_queries),
                        candidate_count=len(retrieved),
                        reranked_count=len(registered),
                        new_evidence_count=new_count,
                        coverage_status=semantic_round_status,
                    )
                )
                final_evidence = self._final_evidence(
                    coverage,
                    evidence_by_id,
                    last_reranked,
                )
                context = self._build_context(
                    route.standalone_query,
                    final_evidence,
                    len(round_reports),
                )
                self._append_source_mapping(
                    sid,
                    topic_id,
                    final_evidence,
                    coverage,
                    plan,
                )
                events = self.store.list_events(sid, topic_id)
                messages = build_topic_messages(events)
                new_message_count = max(0, len(events) - previous_event_count)
                try:
                    draft, generation_metadata = (
                        self.reasoning_provider.generate_answer(messages)
                    )
                except Exception as error:
                    self._stage_fallbacks.append("answer_generation_after_retrieval")
                    draft = AgenticAnswerDraft(
                        answerable=False,
                        claims=[],
                        refusal_reason=(
                            "补充检索后的回答模型未返回合法结构："
                            f"{type(error).__name__}"
                        ),
                    )
                    generation_metadata = {}
                structural = validate_answer_draft(
                    _agentic_to_answer_draft(draft),
                    context,
                )
                structural = self._validate_evidence_mapping(
                    draft,
                    context,
                    structural,
                )
                semantic = self._semantic_validate(
                    route.standalone_query,
                    plan,
                    draft,
                    final_evidence,
                    structural.valid,
                )
            repair_used = False
            if draft.answerable and not semantic.valid:
                repair_used = True
                try:
                    draft = self.reasoning_provider.repair_answer(
                        plan,
                        draft,
                        semantic,
                        final_evidence,
                    )
                except Exception:
                    self._stage_fallbacks.append("answer_repair")
                    draft = deterministic_repair(draft, semantic)
                structural = validate_answer_draft(
                    _agentic_to_answer_draft(draft),
                    context,
                )
                structural = self._validate_evidence_mapping(draft, context, structural)
                semantic = self._semantic_validate(
                    route.standalone_query,
                    plan,
                    draft,
                    final_evidence,
                    structural.valid,
                )
            final_valid = structural.valid and semantic.valid
            if draft.answerable and final_valid:
                answer = GroundedAnswer(
                    query=query,
                    answerable=True,
                    answer=_render_answer(draft),
                    claims=_agentic_to_answer_draft(draft).claims,
                    citations=structural.citations,
                    refusal_reason=None,
                    validation=structural,
                    context=context,
                    diagnostics={},
                )
            else:
                reason = draft.refusal_reason or (
                    "回答在一次修复后仍未通过 Claim-Citation 语义验证。"
                )
                answer = GroundedAnswer(
                    query=query,
                    answerable=False,
                    answer="",
                    claims=[],
                    citations=[],
                    refusal_reason=reason,
                    validation=structural,
                    context=context,
                    diagnostics={},
                )
            prompt_cache = prompt_cache_diagnostics(
                messages,
                new_message_count=new_message_count,
                usage=(
                    generation_metadata.get("usage")
                    if isinstance(generation_metadata.get("usage"), dict)
                    else None
                ),
            )

        validation_payload = semantic.model_dump(mode="json")
        validation_payload["structural"] = answer.validation.to_dict()
        validation_payload["valid"] = bool(
            answer.validation.valid and semantic.valid and answer.answerable
        )
        self.store.append_event(
            sid,
            topic_id,
            "answer",
            {
                "answerable": answer.answerable,
                "answer": answer.answer,
                "claims": [claim.to_dict() for claim in answer.claims],
                "refusal_reason": answer.refusal_reason,
            },
        )
        self.store.append_event(sid, topic_id, "validation", validation_payload)
        confirmed = [
            *topic.get("confirmed_facts", []),
            *(claim.text for claim in answer.claims if answer.answerable),
        ]
        open_questions = (
            []
            if coverage.overall_sufficient and answer.answerable
            else [
                item.missing_information
                for item in coverage.coverage
                if item.status != "sufficient"
            ]
        )
        state_delta = {
            "confirmed_facts_added": [
                claim.text for claim in answer.claims if answer.answerable
            ],
            "open_questions": open_questions,
            "evidence_reused": total_reused,
            "evidence_added": total_new,
        }
        self.store.append_event(sid, topic_id, "state_delta", state_delta)
        updated_summary = str(topic["topic_summary"])
        entities = list(
            dict.fromkeys(
                [*topic.get("entities", []), *extract_entities(route.standalone_query)]
            )
        )
        self.store.update_topic_state(
            sid,
            topic_id,
            topic_summary=updated_summary,
            user_goal=str(topic["user_goal"]),
            entities=entities,
            confirmed_facts=confirmed,
            open_questions=open_questions,
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        metrics = AgenticMetricsRecord(
            retrieval_rounds=len(round_reports),
            new_evidence_count=total_new,
            reused_evidence_count=total_reused,
            coverage_sufficient=coverage.overall_sufficient,
            citation_count=len(answer.citations),
            entailed_claim_count=sum(
                item.entailment == "entailed" for item in semantic.claim_results
            ),
            total_claim_count=len(semantic.claim_results),
            cache_hit_rate=prompt_cache.get("cache_hit_rate"),
            latency_ms=elapsed_ms,
            citation_precision=(
                sum(
                    item.entailment == "entailed"
                    for item in semantic.claim_results
                )
                / len(semantic.claim_results)
                if semantic.claim_results
                else None
            ),
            claim_entailment_accuracy=(
                sum(
                    item.entailment == "entailed"
                    for item in semantic.claim_results
                )
                / len(semantic.claim_results)
                if semantic.claim_results
                else None
            ),
            first_turn=session_created,
            compaction_count=compaction_count,
        )
        result = answer.to_dict(include_context=include_context)
        result["validation"] = validation_payload
        result["session"] = {
            "session_id": sid,
            "topic_id": topic_id,
            "relation": route.relation,
            "standalone_query": route.standalone_query,
            "session_created": session_created,
        }
        result["query_analysis"] = {
            "routing": route.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
        }
        result["retrieval_rounds"] = [
            item.model_dump(mode="json") for item in round_reports
        ]
        result["coverage"] = coverage.model_dump(mode="json")
        result["diagnostics"] = {
            "retrieval_mode": "agentic",
            "retriever": "hybrid_rrf",
            "candidate_limit": self.config.candidate_limit,
            "rrf_k": self.config.rrf_k,
            "embedding_model": getattr(
                self.runtime.embedding_provider,
                "model_name",
                None,
            ),
            "embedding_revision": getattr(
                self.runtime.embedding_provider,
                "revision",
                None,
            ),
            "reranker": (
                reranker_diagnostics[-1]
                if reranker_diagnostics
                else {
                    "enabled": self.config.reranker_enabled,
                    "candidate_count": 0,
                    "output_count": 0,
                    "elapsed_ms": 0.0,
                    "fallback_used": not self.config.reranker_enabled,
                }
            ),
            "reranker_rounds": reranker_diagnostics,
            "retrieval_rounds": [
                item.model_dump(mode="json") for item in round_reports
            ],
            "prompt_cache": prompt_cache,
            "repair_used": repair_used,
            "stage_fallbacks": list(dict.fromkeys(self._stage_fallbacks)),
            "metrics": metrics.model_dump(mode="json"),
            "elapsed_ms": elapsed_ms,
        }
        return result
