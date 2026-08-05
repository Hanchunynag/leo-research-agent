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
from app.agentic.harness import (
    AgenticRunHarness,
    AgenticRunPolicy,
    AgenticStage,
    TerminationReason,
)
from app.agentic.models import (
    AgenticAnswerDraft,
    AgenticMetricsRecord,
    CoverageItem,
    CoverageReport,
    EvidenceStatus,
    QueryPlan,
    RetrievalRound,
    RoutingLLMDecision,
    SemanticValidationReport,
)
from app.agentic.planning import QueryPlanner
from app.agentic.query_expansion import AdaptiveQueryExpander
from app.agentic.query_validation import QueryDriftValidator
from app.agentic.prompting import (
    build_topic_messages,
    compact_topic,
    prompt_cache_diagnostics,
)
from app.agentic.provider import AgenticReasoningProvider, StructuredOutputError
from app.agentic.reranking import DirectAnswerReranker
from app.agentic.routing import TopicRouter, extract_entities, rewrite_standalone_query
from app.agentic.selection import CoverageAwareEvidenceSelector
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
from app.indexing.tokenization import normalize_search_text, token_count
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


def _generation_failure(error: Exception, *, stage: str) -> dict[str, Any]:
    """将生成异常转换为不含原始响应和秘密的诊断。"""

    if isinstance(error, StructuredOutputError):
        return error.to_diagnostics()
    return {
        "stage": stage,
        "failure_kind": type(error).__name__,
        "validation_issues": [],
    }


def _generation_failure_message(details: dict[str, Any]) -> str:
    kind = str(details.get("failure_kind") or "unknown")
    issues = details.get("validation_issues")
    suffix = ""
    if isinstance(issues, list) and issues and isinstance(issues[0], dict):
        location = str(issues[0].get("location") or "$")
        suffix = f"，首个不匹配字段为 {location}"
    return (
        f"回答模型返回的结构不符合约束（{kind}{suffix}）。"
        "检索证据已保留，这不是证据不足。"
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
        self.policy = AgenticRunPolicy(
            max_retrieval_rounds=config.max_retrieval_rounds,
            max_structure_repairs=config.max_structure_repairs,
            max_answer_repairs=config.max_answer_repairs,
            max_total_latency_ms=config.max_total_latency_ms,
            fail_closed=config.fail_closed,
            allow_model_downloads=config.allow_model_downloads,
        )
        self.planner = QueryPlanner()
        expansion_provider = getattr(reasoning_provider, "provider", None)
        self.query_expander = AdaptiveQueryExpander(
            expansion_provider if hasattr(expansion_provider, "chat_completion") else None,
            max_variants=config.max_query_variants,
        )
        self.query_drift_validator = QueryDriftValidator(retrieval_runtime.embedding_provider)
        self.evidence_selector = CoverageAwareEvidenceSelector(
            mmr_lambda=config.evidence_mmr_lambda,
            max_per_work=config.max_final_evidence_per_work,
            min_directness_grade=config.min_final_directness_grade,
        )
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
        if getattr(self.runtime, "is_graphrag", False):
            result = self.runtime.retrieve_multi(
                queries, limit=self.config.max_cross_query_candidates,
                rrf_k=self.config.rrf_k,
            )
            raw = result.get("results")
            return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
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
        deterministic = deterministic_coverage(plan, evidence)
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
            if (
                plan.intent == "synthesis"
                and plan.target_category == "method"
                and deterministic.overall_sufficient
                and not report.overall_sufficient
            ):
                # 综述问题不应被“没有单篇论文直接写出整体路线”拒答。
                self._stage_fallbacks.append("coverage_compositional_synthesis")
                return deterministic
            return report
        except Exception:
            self._stage_fallbacks.append("coverage")
            return deterministic

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

    @staticmethod
    def _restrict_draft_to_context(
        draft: AgenticAnswerDraft,
        context: ContextBundle,
    ) -> tuple[AgenticAnswerDraft, int, int]:
        """只保留当前 source_id 与 evidence_id 稳定映射内的引用。"""

        if not draft.answerable:
            return draft, 0, 0
        mapping = {
            item.source_id: item.evidence_id
            for item in context.evidence
            if item.evidence_id
        }
        kept = []
        dropped = 0
        narrowed = 0
        for claim in draft.claims:
            claimed_evidence = set(claim.evidence_ids)
            valid_sources = [
                source_id
                for source_id in claim.source_ids
                if mapping.get(source_id) in claimed_evidence
            ]
            valid_evidence = list(
                dict.fromkeys(
                    str(mapping[source_id]) for source_id in valid_sources
                )
            )
            if not valid_sources or not valid_evidence:
                dropped += 1
                continue
            if (
                valid_sources != claim.source_ids
                or valid_evidence != claim.evidence_ids
            ):
                narrowed += 1
            kept.append(
                claim.model_copy(
                    update={
                        "source_ids": valid_sources,
                        "evidence_ids": valid_evidence,
                    }
                )
            )
        if not kept:
            return (
                AgenticAnswerDraft(
                    answerable=False,
                    claims=[],
                    refusal_reason=(
                        "回答模型未使用本轮 Context 中的合法证据映射。"
                    ),
                ),
                dropped,
                narrowed,
            )
        return (
            draft.model_copy(update={"claims": kept, "refusal_reason": None}),
            dropped,
            narrowed,
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
                        "authors": record.get("authors") or [],
                        "year": record.get("year"),
                        "doi": record.get("doi"),
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
        query: str,
        plan: QueryPlan,
        coverage: CoverageReport,
        evidence_by_id: dict[str, dict[str, Any]],
        latest_reranked: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """使用 Coverage 必选与 MMR 去冗余/多样性策略分配最终证据预算。"""

        candidates = [
            *evidence_by_id.values(),
            *latest_reranked,
        ]
        return self.evidence_selector.select(
            query,
            plan,
            coverage,
            candidates,
            self.config.final_top_k,
        )

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
            # 来源上限已由 Selector 执行；此处不再二次丢弃 Coverage 必选证据。
            max_evidence_per_work=self.config.final_top_k,
            retrieval_diagnostics={
                "retriever": "agentic_hybrid_rrf",
                "retrieval_rounds": round_count,
            },
        )

    @staticmethod
    def _align_evidence_with_context(
        selected: Sequence[dict[str, Any]],
        context: ContextBundle,
    ) -> list[dict[str, Any]]:
        """以 Context Builder 实际保留的条目为唯一可引用证据集。"""

        by_chunk = {
            str(item.get("chunk_id") or ""): item
            for item in selected
            if item.get("chunk_id")
        }
        aligned: list[dict[str, Any]] = []
        for context_item in context.evidence:
            original = by_chunk.get(context_item.chunk_id, {})
            value = {**original, **context_item.to_dict()}
            value["source_id"] = context_item.source_id
            value["evidence_id"] = context_item.evidence_id
            aligned.append(value)
        return aligned

    @staticmethod
    def _context_preserves_coverage(
        coverage: CoverageReport,
        evidence: Sequence[dict[str, Any]],
    ) -> bool:
        available = {
            str(item.get("evidence_id") or "")
            for item in evidence
            if item.get("evidence_id")
        }
        return all(
            bool(available & set(item.supporting_evidence_ids))
            for item in coverage.coverage
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
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "section_path": item.get("section_path"),
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
        harness = AgenticRunHarness(self.policy)
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
        with harness.stage(AgenticStage.ROUTING) as routing_trace:
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
            routing_trace.update(
                {
                    "relation": route.relation,
                    "context_dependent": route.context_dependent,
                    "preliminary_candidate_count": len(preliminary),
                }
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
        with harness.stage(AgenticStage.PLANNING) as planning_trace:
            plan = self.planner.plan(route.standalone_query)
            planning_trace.update(
                {
                    "intent": plan.intent,
                    "target_category": plan.target_category,
                    "subquestion_count": len(plan.subquestions),
                    "retrieval_query_count": len(plan.retrieval_queries),
                }
            )
        accepted_retrieval_queries = list(plan.retrieval_queries)
        query_validation_diagnostics: dict[str, Any] = {}
        if getattr(self.runtime, "is_graphrag", False):
            harness.begin_query_expansion()
            with harness.stage(AgenticStage.QUERY_EXPANDING) as expansion_trace:
                expansion = self.query_expander.expand(route.standalone_query, plan)
                harness.record_query_variants(len(expansion.queries))
                expansion_trace.update({"complexity": expansion.complexity,
                    "retrieval_mode": expansion.retrieval_mode,
                    "generated_query_count": len(expansion.queries)})
            with harness.stage(AgenticStage.QUERY_VALIDATING) as validation_trace:
                validation = self.query_drift_validator.validate(expansion)
                accepted_retrieval_queries = [value.text for value in validation.accepted_queries]
                query_validation_diagnostics = validation.model_dump(mode="json")
                validation_trace.update({"accepted_query_count": len(validation.accepted_queries),
                    "rejected_query_count": len(validation.rejected_queries),
                    "query_drift_reasons": [reason for decision in validation.decisions
                                             for reason in decision.reasons]})
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
        generation_failure: dict[str, Any] | None = None
        queries = accepted_retrieval_queries
        coverage = CoverageReport(
            overall_sufficient=False,
            coverage=[],
            followup_queries=queries,
        )
        last_reranked: list[dict[str, Any]] = []
        while harness.can_retrieve():
            round_number = harness.begin_retrieval_round()
            if getattr(self.runtime, "is_graphrag", False):
                with harness.stage(AgenticStage.RETRIEVAL_DISPATCHING,
                                   attempt=round_number,
                                   details={"query_count": len(queries)}) as retrieval_trace:
                    fresh = [*(preliminary if round_number == 1 else []),
                             *self._retrieve_queries(queries)]
                    reused = (self._relevant_registry(route.standalone_query, registry)
                              if route.reuse_previous_evidence else [])
                    retrieved = self._merge_fresh_and_reused(fresh, reused)
                    retrieval_trace.update({"candidate_count": len(retrieved),
                                            "reused_candidate_count": len(reused)})
                route_diagnostics = getattr(self.runtime, "last_diagnostics", {})
                for stage, route_name in (
                    (AgenticStage.LEXICAL_RETRIEVING, "lexical"),
                    (AgenticStage.DENSE_RETRIEVING, "dense"),
                    (AgenticStage.GRAPH_RETRIEVING, "graph_direct"),
                    (AgenticStage.COMMUNITY_RETRIEVING, "community"),
                ):
                    with harness.stage(stage, attempt=round_number, details={
                        "candidate_count": route_diagnostics.get(
                            "per_route_candidate_count", {}).get(route_name, 0)
                    }):
                        pass
                with harness.stage(AgenticStage.QUERY_FUSING, attempt=round_number,
                    details={"cross_query_fusion_count": len(retrieved)}):
                    pass
            else:
                with harness.stage(
                    AgenticStage.RETRIEVING,
                    attempt=round_number,
                    details={"query_count": len(queries)},
                ) as retrieval_trace:
                    fresh = [*(preliminary if round_number == 1 else []),
                             *self._retrieve_queries(queries)]
                    reused = (self._relevant_registry(route.standalone_query, registry)
                              if route.reuse_previous_evidence else [])
                    retrieved = self._merge_fresh_and_reused(fresh, reused)
                    retrieval_trace.update({"candidate_count": len(retrieved),
                                            "reused_candidate_count": len(reused)})
            with harness.stage(
                AgenticStage.RERANKING,
                attempt=round_number,
                details={"candidate_count": len(retrieved)},
            ) as rerank_trace:
                reranked = self.reranker.rerank(
                    route.standalone_query,
                    retrieved,
                    self.config.rerank_top_k,
                    plan,
                )
                rerank_trace.update(
                    {
                        "output_count": len(reranked),
                        "fallback_used": bool(
                            self.reranker.last_diagnostics.get("fallback_used")
                        ),
                    }
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
            with harness.stage(
                AgenticStage.COVERAGE_CHECKING,
                attempt=round_number,
                details={"evidence_count": len(evidence_by_id)},
            ) as coverage_trace:
                coverage = self._coverage(plan, list(evidence_by_id.values()))
                if (getattr(self.runtime, "is_graphrag", False) and
                        getattr(self.runtime, "last_diagnostics", {}).get(
                            "relationship_status") == "none"):
                    relation_reason = ("当前知识库分别包含相关实体的定义，但没有检索到"
                                       "能够证明二者关系的直接证据或可靠路径。")
                    coverage = CoverageReport(overall_sufficient=False,
                        coverage=[CoverageItem(subquestion_id=item.id, status="missing",
                            supporting_evidence_ids=[], missing_information=relation_reason)
                                  for item in plan.subquestions], followup_queries=[])
                coverage_trace.update(
                    {
                        "overall_sufficient": coverage.overall_sufficient,
                        "followup_query_count": len(coverage.followup_queries),
                    }
                )
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
            if getattr(self.runtime, "is_graphrag", False):
                queries = queries[: self.config.max_focused_queries_per_round]
            if not queries:
                break

        with harness.stage(AgenticStage.CONTEXT_BUILDING) as context_trace:
            final_evidence = self._final_evidence(
                route.standalone_query,
                plan,
                coverage,
                evidence_by_id,
                last_reranked,
            )
            context = self._build_context(
                route.standalone_query,
                final_evidence,
                len(round_reports),
            )
            selected_before_context = len(final_evidence)
            final_evidence = self._align_evidence_with_context(
                final_evidence,
                context,
            )
            context_coverage_preserved = self._context_preserves_coverage(
                coverage,
                final_evidence,
            )
            context_trace.update(
                {
                    "evidence_count": len(final_evidence),
                    "selected_before_context": selected_before_context,
                    "token_count": context.token_count,
                    "coverage_preserved": context_coverage_preserved
                    and bool(
                        self.evidence_selector.last_diagnostics.get(
                            "coverage_preserved"
                        )
                    ),
                }
            )
        selection_sufficient = bool(
            self.evidence_selector.last_diagnostics.get("coverage_preserved")
        ) and context_coverage_preserved
        budget_exhausted = harness.deadline_exceeded()
        if (
            budget_exhausted
            or not coverage.overall_sufficient
            or not selection_sufficient
        ):
            reason = (
                "Harness 总运行时限已耗尽。"
                if budget_exhausted
                else (
                    "; ".join(
                        item.missing_information
                        for item in coverage.coverage
                        if item.status != "sufficient"
                    )
                    or (
                        "最终证据预算无法保留所有子问题的直接证据："
                        + ", ".join(
                            self.evidence_selector.last_diagnostics.get(
                                "uncovered_subquestions", []
                            )
                        )
                        if not selection_sufficient
                        else "达到最大检索轮数后证据覆盖仍不足。"
                    )
                )
            )
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
                with harness.stage(AgenticStage.COMPACTING) as compaction_trace:
                    report = compact_topic(
                        self.store,
                        sid,
                        topic_id,
                        recent_event_count=self.config.recent_events_after_compaction,
                    )
                    compaction_count += 1
                    events = self.store.list_events(sid, topic_id)
                    messages = build_topic_messages(events)
                    previous_event_count = 0
                    compaction_trace.update(
                        {
                            "before_tokens": report.before_tokens,
                            "after_tokens": report.after_tokens,
                            "retained_evidence_count": len(
                                report.retained_evidence_ids
                            ),
                        }
                    )
            new_message_count = max(0, len(events) - previous_event_count)
            with harness.stage(AgenticStage.GENERATING) as generation_trace:
                try:
                    draft, generation_metadata = (
                        self.reasoning_provider.generate_answer(messages)
                    )
                except Exception as error:
                    self._stage_fallbacks.append("answer_generation")
                    generation_trace["fallback_used"] = True
                    generation_trace["error_type"] = type(error).__name__
                    generation_failure = _generation_failure(
                        error,
                        stage="answer",
                    )
                    generation_trace["structured_output"] = generation_failure
                    draft = AgenticAnswerDraft(
                        answerable=False,
                        claims=[],
                        refusal_reason=_generation_failure_message(
                            generation_failure
                        ),
                    )
                    generation_metadata = {}
                draft, citation_dropped, citation_narrowed = (
                    self._restrict_draft_to_context(draft, context)
                )
                if citation_dropped or citation_narrowed:
                    self._stage_fallbacks.append("citation_scope_guardrail")
                structure_repairs = generation_metadata.get(
                    "structure_repair_attempts",
                    0,
                )
                if isinstance(structure_repairs, int):
                    harness.record_structure_repairs(structure_repairs)
                generation_trace.update(
                    {
                        "answerable": draft.answerable,
                        "claim_count": len(draft.claims),
                        "citation_claims_dropped": citation_dropped,
                        "citation_claims_narrowed": citation_narrowed,
                        "structure_repair_attempts": (
                            structure_repairs
                            if isinstance(structure_repairs, int)
                            else 0
                        ),
                    }
                )
            with harness.stage(
                AgenticStage.STRUCTURAL_VALIDATING
            ) as structural_trace:
                structural = validate_answer_draft(
                    _agentic_to_answer_draft(draft),
                    context,
                )
                structural = self._validate_evidence_mapping(
                    draft,
                    context,
                    structural,
                )
                structural_trace.update(
                    {
                        "valid": structural.valid,
                        "issue_count": len(structural.issues),
                    }
                )
            with harness.stage(
                AgenticStage.SEMANTIC_VALIDATING
            ) as semantic_trace:
                semantic = self._semantic_validate(
                    route.standalone_query,
                    plan,
                    draft,
                    final_evidence,
                    structural.valid,
                )
                semantic_trace.update(
                    {
                        "valid": semantic.valid,
                        "requires_retrieval": semantic.requires_retrieval,
                        "claim_count": len(semantic.claim_results),
                    }
                )
            if (
                draft.answerable
                and semantic.requires_retrieval
                and harness.can_retrieve()
            ):
                round_number = harness.begin_retrieval_round()
                followup_queries = semantic.followup_queries or [
                    route.standalone_query
                ]
                with harness.stage(
                    AgenticStage.RETRIEVING,
                    attempt=round_number,
                    details={"query_count": len(followup_queries), "semantic": True},
                ) as retrieval_trace:
                    retrieved = self._merge_fresh_and_reused(
                        self._retrieve_queries(followup_queries),
                        self._relevant_registry(
                            route.standalone_query,
                            list(evidence_by_id.values()),
                        ),
                    )
                    retrieval_trace["candidate_count"] = len(retrieved)
                with harness.stage(
                    AgenticStage.RERANKING,
                    attempt=round_number,
                    details={"candidate_count": len(retrieved), "semantic": True},
                ) as rerank_trace:
                    reranked = self.reranker.rerank(
                        route.standalone_query,
                        retrieved,
                        self.config.rerank_top_k,
                        plan,
                    )
                    rerank_trace.update(
                        {
                            "output_count": len(reranked),
                            "fallback_used": bool(
                                self.reranker.last_diagnostics.get("fallback_used")
                            ),
                        }
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
                with harness.stage(
                    AgenticStage.COVERAGE_CHECKING,
                    attempt=round_number,
                    details={"evidence_count": len(evidence_by_id), "semantic": True},
                ) as coverage_trace:
                    coverage = self._coverage(plan, list(evidence_by_id.values()))
                    coverage_trace.update(
                        {
                            "overall_sufficient": coverage.overall_sufficient,
                            "followup_query_count": len(coverage.followup_queries),
                        }
                    )
                semantic_round_status: EvidenceStatus = (
                    "sufficient"
                    if coverage.overall_sufficient
                    else "partial"
                )
                round_reports.append(
                    RetrievalRound(
                        round=round_number,
                        queries=list(followup_queries),
                        candidate_count=len(retrieved),
                        reranked_count=len(registered),
                        new_evidence_count=new_count,
                        coverage_status=semantic_round_status,
                    )
                )
                with harness.stage(
                    AgenticStage.CONTEXT_BUILDING,
                    attempt=round_number,
                ) as context_trace:
                    final_evidence = self._final_evidence(
                        route.standalone_query,
                        plan,
                        coverage,
                        evidence_by_id,
                        last_reranked,
                    )
                    context = self._build_context(
                        route.standalone_query,
                        final_evidence,
                        len(round_reports),
                    )
                    selected_before_context = len(final_evidence)
                    final_evidence = self._align_evidence_with_context(
                        final_evidence,
                        context,
                    )
                    context_coverage_preserved = (
                        self._context_preserves_coverage(
                            coverage,
                            final_evidence,
                        )
                    )
                    context_trace.update(
                        {
                            "evidence_count": len(final_evidence),
                            "selected_before_context": selected_before_context,
                            "token_count": context.token_count,
                            "coverage_preserved": context_coverage_preserved
                            and bool(
                                self.evidence_selector.last_diagnostics.get(
                                    "coverage_preserved"
                                )
                            ),
                        }
                    )
                selection_sufficient = bool(
                    self.evidence_selector.last_diagnostics.get(
                        "coverage_preserved"
                    )
                ) and context_coverage_preserved
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
                with harness.stage(
                    AgenticStage.GENERATING,
                    attempt=2,
                    details={"after_semantic_retrieval": True},
                ) as generation_trace:
                    try:
                        draft, generation_metadata = (
                            self.reasoning_provider.generate_answer(messages)
                        )
                    except Exception as error:
                        self._stage_fallbacks.append(
                            "answer_generation_after_retrieval"
                        )
                        generation_trace["fallback_used"] = True
                        generation_trace["error_type"] = type(error).__name__
                        generation_failure = _generation_failure(
                            error,
                            stage="answer_after_retrieval",
                        )
                        generation_trace["structured_output"] = generation_failure
                        draft = AgenticAnswerDraft(
                            answerable=False,
                            claims=[],
                            refusal_reason=_generation_failure_message(
                                generation_failure
                            ),
                        )
                        generation_metadata = {}
                    draft, citation_dropped, citation_narrowed = (
                        self._restrict_draft_to_context(draft, context)
                    )
                    if citation_dropped or citation_narrowed:
                        self._stage_fallbacks.append(
                            "citation_scope_guardrail_after_retrieval"
                        )
                    structure_repairs = generation_metadata.get(
                        "structure_repair_attempts",
                        0,
                    )
                    if isinstance(structure_repairs, int):
                        harness.record_structure_repairs(structure_repairs)
                    generation_trace.update(
                        {
                            "answerable": draft.answerable,
                            "claim_count": len(draft.claims),
                            "citation_claims_dropped": citation_dropped,
                            "citation_claims_narrowed": citation_narrowed,
                            "structure_repair_attempts": (
                                structure_repairs
                                if isinstance(structure_repairs, int)
                                else 0
                            ),
                        }
                    )
                with harness.stage(
                    AgenticStage.STRUCTURAL_VALIDATING,
                    attempt=2,
                ) as structural_trace:
                    structural = validate_answer_draft(
                        _agentic_to_answer_draft(draft),
                        context,
                    )
                    structural = self._validate_evidence_mapping(
                        draft,
                        context,
                        structural,
                    )
                    structural_trace.update(
                        {
                            "valid": structural.valid,
                            "issue_count": len(structural.issues),
                        }
                    )
                with harness.stage(
                    AgenticStage.SEMANTIC_VALIDATING,
                    attempt=2,
                ) as semantic_trace:
                    semantic = self._semantic_validate(
                        route.standalone_query,
                        plan,
                        draft,
                        final_evidence,
                        structural.valid,
                    )
                    semantic_trace.update(
                        {
                            "valid": semantic.valid,
                            "requires_retrieval": semantic.requires_retrieval,
                            "claim_count": len(semantic.claim_results),
                        }
                    )
            repair_used = False
            if (
                draft.answerable
                and not semantic.valid
                and harness.can_repair_answer()
            ):
                repair_used = True
                repair_attempt = harness.begin_answer_repair()
                with harness.stage(
                    AgenticStage.REPAIRING,
                    attempt=repair_attempt,
                ) as repair_trace:
                    try:
                        draft = self.reasoning_provider.repair_answer(
                            plan,
                            draft,
                            semantic,
                            final_evidence,
                        )
                    except Exception as error:
                        self._stage_fallbacks.append("answer_repair")
                        repair_trace["fallback_used"] = True
                        repair_trace["error_type"] = type(error).__name__
                        draft = deterministic_repair(draft, semantic)
                    draft, citation_dropped, citation_narrowed = (
                        self._restrict_draft_to_context(draft, context)
                    )
                    if citation_dropped or citation_narrowed:
                        self._stage_fallbacks.append(
                            "citation_scope_guardrail_after_repair"
                        )
                    repair_trace.update(
                        {
                            "answerable": draft.answerable,
                            "claim_count": len(draft.claims),
                            "citation_claims_dropped": citation_dropped,
                            "citation_claims_narrowed": citation_narrowed,
                        }
                    )
                with harness.stage(
                    AgenticStage.STRUCTURAL_VALIDATING,
                    attempt=2,
                ) as structural_trace:
                    structural = validate_answer_draft(
                        _agentic_to_answer_draft(draft),
                        context,
                    )
                    structural = self._validate_evidence_mapping(
                        draft,
                        context,
                        structural,
                    )
                    structural_trace.update(
                        {
                            "valid": structural.valid,
                            "issue_count": len(structural.issues),
                        }
                    )
                with harness.stage(
                    AgenticStage.SEMANTIC_VALIDATING,
                    attempt=2,
                ) as semantic_trace:
                    semantic = self._semantic_validate(
                        route.standalone_query,
                        plan,
                        draft,
                        final_evidence,
                        structural.valid,
                    )
                    semantic_trace.update(
                        {
                            "valid": semantic.valid,
                            "requires_retrieval": semantic.requires_retrieval,
                            "claim_count": len(semantic.claim_results),
                        }
                    )
            deadline_after_validation = harness.deadline_exceeded()
            final_valid = (
                structural.valid
                and semantic.valid
                and not deadline_after_validation
            )
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
                reason = (
                    "Harness 总运行时限已耗尽。"
                    if deadline_after_validation
                    else (
                        draft.refusal_reason
                        or "回答在一次修复后仍未通过 Claim-Citation 语义验证。"
                    )
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

        deadline_exhausted = harness.deadline_exceeded()
        if answer.answerable:
            outcome = {
                "code": "answered",
                "stage": "completed",
                "message": "回答已通过结构和语义验证。",
                "retryable": False,
            }
        elif generation_failure is not None:
            outcome = {
                "code": "generation_failed",
                "stage": str(generation_failure.get("stage") or "answer"),
                "message": answer.refusal_reason,
                "retryable": True,
                "details": generation_failure,
            }
        elif deadline_exhausted:
            outcome = {
                "code": "budget_exhausted",
                "stage": "harness",
                "message": answer.refusal_reason,
                "retryable": True,
            }
        elif not coverage.overall_sufficient or not selection_sufficient:
            outcome = {
                "code": "insufficient_evidence",
                "stage": "coverage",
                "message": answer.refusal_reason,
                "retryable": False,
            }
        else:
            outcome = {
                "code": "validation_failed",
                "stage": "semantic_validation",
                "message": answer.refusal_reason,
                "retryable": True,
            }

        with harness.stage(AgenticStage.PERSISTING) as persistence_trace:
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
                    "outcome": outcome,
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
                    [
                        *topic.get("entities", []),
                        *extract_entities(route.standalone_query),
                    ]
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
            persistence_trace.update(
                {
                    "answerable": answer.answerable,
                    "claim_count": len(answer.claims),
                    "event_count_added": 3,
                }
            )
        if outcome["code"] == "answered":
            termination_reason = TerminationReason.COMPLETED
        elif outcome["code"] == "insufficient_evidence":
            termination_reason = TerminationReason.INSUFFICIENT_COVERAGE
        elif outcome["code"] == "generation_failed":
            termination_reason = TerminationReason.GENERATION_FAILED
        elif outcome["code"] == "budget_exhausted":
            termination_reason = TerminationReason.BUDGET_EXHAUSTED
        else:
            termination_reason = TerminationReason.SEMANTIC_VALIDATION_FAILED
        harness.finish(
            answerable=answer.answerable,
            reason=termination_reason,
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
        result["outcome"] = outcome
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
            "evidence_selection": dict(self.evidence_selector.last_diagnostics),
            "retrieval_rounds": [
                item.model_dump(mode="json") for item in round_reports
            ],
            "prompt_cache": prompt_cache,
            "repair_used": repair_used,
            "stage_fallbacks": list(dict.fromkeys(self._stage_fallbacks)),
            "query_validation": query_validation_diagnostics,
            "metrics": metrics.model_dump(mode="json"),
            "elapsed_ms": elapsed_ms,
            "harness": harness.diagnostics(),
        }
        return result
