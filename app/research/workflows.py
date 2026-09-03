"""四类受预算约束的 Research Workflow。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence

from app.research.context import PhaseContextPack, ResearchContextManager
from app.research.coverage import DimensionCoverageAnalyzer, QueryFrameBuilder
from app.research.harness import (
    BudgetExceeded,
    HarnessState,
    RecoveryLevel,
    ResearchRunHarness,
)
from app.research.tools import ToolGatewayRegistry
from app.research.provisional import (
    abstract_evidence_from_paper,
    register_abstract_evidence,
)
from app.generation.security import redact_sensitive_text


class WorkflowName(StrEnum):
    DIRECT_QA = "direct_qa"
    RELATION_REASONING = "relation_reasoning"
    DEEP_RESEARCH = "deep_research"
    RESEARCH_BOOTSTRAP = "research_bootstrap"


class AnswerGenerator(Protocol):
    def generate(self, context: PhaseContextPack) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class WorkflowExecution:
    draft: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    scope: Mapping[str, Any]
    conflicts: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, Any]
    details: Mapping[str, Any]


def _tool_context(workflow: WorkflowName, request: Any) -> dict[str, Any]:
    return {
        "workflow": workflow.value,
        "workspace_id": request.workspace_id,
        "scope_version": request.scope_version,
        "permissions": tuple(request.permissions),
    }


def _selected(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = value.get("results")
    values = raw if isinstance(raw, list) else []
    return tuple(
        dict(item)
        for item in values
        if isinstance(item, Mapping)
        and (item.get("evidence_state") or item.get("state")) == "selected"
    )


def _conflicts(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return ()
    raw = diagnostics.get("conflicts")
    return tuple(dict(item) for item in raw if isinstance(item, Mapping)) if isinstance(raw, list) else ()


def _generation_failure(error: Exception, *, stage: str = "answer") -> dict[str, Any]:
    """将回答模型异常转换为可展示、且不泄露密钥的诊断。"""

    details: dict[str, Any] = {
        "stage": stage,
        "failure_kind": type(error).__name__,
    }
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        details["http_status"] = status_code
    if response is not None:
        try:
            body = str(response.text or "")
        except Exception:
            body = ""
        if body:
            details["response_body"] = redact_sensitive_text(body)[:500]
    diagnostic_builder = getattr(error, "to_diagnostics", None)
    if callable(diagnostic_builder):
        try:
            diagnostics = diagnostic_builder()
        except Exception:
            diagnostics = None
        if isinstance(diagnostics, Mapping):
            # Structured provider diagnostics intentionally exclude prompts and
            # model content, so they are safe to surface to the Job inspector.
            details["provider_diagnostics"] = dict(diagnostics)
    return details


def _generation_failure_message(details: Mapping[str, Any]) -> str:
    kind = str(details.get("failure_kind") or "unknown")
    status = details.get("http_status")
    location = f"HTTP {status}, " if isinstance(status, int) else ""
    return (
        f"回答模型请求失败（{location}{kind}）。"
        "检索证据已保留，这不是证据不足。"
    )


class BaseWorkflow:
    name: WorkflowName

    def __init__(
        self,
        gateway: ToolGatewayRegistry,
        contexts: ResearchContextManager,
        generator: AnswerGenerator,
    ) -> None:
        self.gateway = gateway
        self.contexts = contexts
        self.generator = generator

    @staticmethod
    def _degrade_and_continue(
        harness: ResearchRunHarness,
        action: str,
    ) -> None:
        BaseWorkflow._record_recovery(
            harness,
            RecoveryLevel.BACKEND_DEGRADATION,
            action,
            "continued_with_verified_evidence",
        )

    @staticmethod
    def _record_recovery(
        harness: ResearchRunHarness,
        level: RecoveryLevel,
        action: str,
        outcome: str,
    ) -> None:
        harness.transition(HarnessState.RECOVERING)
        harness.recover(level, action, outcome=outcome)
        harness.transition(HarnessState.EXECUTING)

    def _scope(self, request: Any, harness: ResearchRunHarness) -> Mapping[str, Any]:
        result = self.gateway.invoke(
            "workspace.read_scope",
            {"workspace_id": request.workspace_id, "scope_version": request.scope_version},
            context=_tool_context(self.name, request),
            harness=harness,
        )
        self._notify(request, "scope_ready", "Scope 已确认，准备检索。", 0.34)
        return result

    def _retrieve(self, request: Any, harness: ResearchRunHarness, query: str) -> Mapping[str, Any]:
        translated = tuple(
            value.strip()
            for value in getattr(request, "retrieval_queries", ())
            if isinstance(value, str) and value.strip()
        )
        # 对原问题使用翻译 Tool 生成的中英文等价查询；对于关系/缺口等
        # workflow 内部查询，也保留它们，避免丢失明确的任务约束。
        query_variants = tuple(dict.fromkeys((query, *translated)))
        arguments: dict[str, Any] = {
            "query": query,
            "query_variants": list(query_variants),
            "workspace_id": request.workspace_id,
            "scope_version": request.scope_version,
            "top_k": request.top_k,
        }
        target_document_ids = tuple(
            value.strip()
            for value in getattr(request, "target_document_ids", ())
            if isinstance(value, str) and value.strip()
        )
        if target_document_ids:
            arguments["target_document_ids"] = list(target_document_ids)
        result = self.gateway.invoke(
            "knowledge.retrieve",
            arguments,
            context=_tool_context(self.name, request),
            harness=harness,
        )
        self._notify(request, "retrieved", "双语 RAG 召回完成，正在整理 Selected Evidence。", 0.56)
        return result

    @staticmethod
    def _notify(request: Any, stage: str, message: str, progress: float) -> None:
        callback = getattr(request, "progress_callback", None)
        if not callable(callback):
            return
        try:
            callback(stage, message, progress)
        except Exception:
            # 进度 UI 故障不得影响科研问答主链。
            return

    def _generate(
        self,
        request: Any,
        harness: ResearchRunHarness,
        scope: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        timeline_constraints = tuple(
            "论文发表时间元数据（仅用于时间排序，不能替代正文证据）："
            f"{value.get('query_title') or value.get('matched_title') or 'unknown'} — "
            f"{value.get('publication_year') or 'unknown'}"
            for value in getattr(request, "timeline_metadata", ())
            if isinstance(value, Mapping)
        )
        agent_constraints = tuple(
            str(value).strip()
            for value in getattr(request, "agent_instructions", ())
            if str(value).strip()
        )
        context = self.contexts.generator_pack(
            query=request.query,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            scope_constraints=(
                tuple(str(value) for value in scope.get("constraints", []))
                + timeline_constraints
                + agent_constraints
            ),
            evidence=evidence,
            conflicts=conflicts,
            output_format={
                "answerable": "boolean",
                "claims": "[{claim_id,text,category,source_ids,evidence_ids}]",
                "refusal_reason": "string|null",
                "conflicts_acknowledged": "boolean",
                **(
                    {"output_language": request.output_language}
                    if getattr(request, "output_language", None)
                    else {}
                ),
            },
            harness=harness,
        )
        self._notify(request, "generation_start", "正在调用回答模型生成带引用草稿。", 0.64)
        harness.consume("llm_calls")
        draft = self.generator.generate(context)
        self._notify(request, "generated", "回答草稿生成完成，正在执行证据校验。", 0.80)
        usage = draft.get("usage") if isinstance(draft, Mapping) else None
        if isinstance(usage, Mapping):
            harness.consume("total_tokens", int(usage.get("total_tokens") or 0))
            harness.record_provider_usage("answer_generation", dict(usage))
        return dict(draft)


class DirectQAWorkflow(BaseWorkflow):
    name = WorkflowName.DIRECT_QA

    def execute(self, request: Any, harness: ResearchRunHarness) -> WorkflowExecution:
        with harness.step("SCOPE_CHECK"):
            scope = self._scope(request, harness)
        with harness.step("RETRIEVE"):
            retrieval = self._retrieve(request, harness, request.query)
            evidence = _selected(retrieval)
        with harness.step("EVIDENCE_EVALUATE") as trace:
            trace["selected_count"] = len(evidence)
            conflicts = _conflicts(retrieval)
        draft: Mapping[str, Any]
        generation_failure: dict[str, Any] | None = None
        if not evidence:
            draft = {
                "answerable": False,
                "claims": [],
                "refusal_reason": "当前 Scope 中没有 Selected Evidence。",
            }
        else:
            try:
                with harness.step("GENERATE"):
                    draft = self._generate(request, harness, scope, evidence, conflicts)
            except BudgetExceeded:
                raise
            except Exception as error:
                generation_failure = _generation_failure(error)
                draft = {
                    "answerable": False,
                    "claims": [],
                    "refusal_reason": _generation_failure_message(generation_failure),
                }
        return WorkflowExecution(
            draft,
            evidence,
            scope,
            conflicts,
            {"sufficient": bool(evidence)},
            {
                "primary_generation_calls": harness.usage.llm_calls,
                **({"generation_failure": generation_failure} if generation_failure else {}),
            },
        )


class RelationReasoningWorkflow(BaseWorkflow):
    name = WorkflowName.RELATION_REASONING

    def execute(self, request: Any, harness: ResearchRunHarness) -> WorkflowExecution:
        with harness.step("SCOPE_CHECK"):
            scope = self._scope(request, harness)
        with harness.step("QUERY_FRAME"):
            query_frame = QueryFrameBuilder().build(request.query)
            framed = (
                f"关系与机制：{request.query}；必需维度："
                + ", ".join(query_frame.required_dimensions)
            )
        with harness.step("RELATIONAL_RETRIEVE"):
            retrieval = self._retrieve(request, harness, framed)
            evidence = list(_selected(retrieval))
        conflicts = _conflicts(retrieval)
        coverage = DimensionCoverageAnalyzer().evaluate(
            query_frame, evidence, conflicts
        )
        sufficient = coverage.overall_sufficient and len(evidence) >= 2
        gap_failed = False
        if not sufficient and harness.usage.retrieval_rounds < harness.policy.max_retrieval_rounds:
            self._record_recovery(
                harness,
                RecoveryLevel.LIMITED_RETRIEVAL,
                "one_bounded_gap_retrieval",
                "continuing",
            )
            with harness.step("OPTIONAL_GAP_RETRIEVE"):
                try:
                    gap = self._retrieve(
                        request,
                        harness,
                        "限定证据缺口："
                        + ", ".join(coverage.missing_dimensions)
                        + f"；原问题：{request.query}",
                    )
                except (ConnectionError, TimeoutError, OSError):
                    gap_failed = True
                    self._degrade_and_continue(harness, "continue_after_gap_retrieval_failure")
                else:
                    seen = {str(value.get("evidence_id")) for value in evidence}
                    evidence.extend(
                        value for value in _selected(gap) if str(value.get("evidence_id")) not in seen
                    )
                    coverage = DimensionCoverageAnalyzer().evaluate(
                        query_frame, evidence, conflicts
                    )
                    sufficient = coverage.overall_sufficient and len(evidence) >= 2
        with harness.step("EVIDENCE_EVALUATE") as trace:
            trace.update(
                {
                    "selected_count": len(evidence),
                    "coverage_sufficient": sufficient,
                    "missing_dimensions": list(coverage.missing_dimensions),
                }
            )
        draft: Mapping[str, Any]
        generation_failure: dict[str, Any] | None = None
        if not evidence:
            draft = {"answerable": False, "claims": [], "refusal_reason": "关系证据不足。"}
        else:
            try:
                with harness.step("GENERATE"):
                    draft = self._generate(request, harness, scope, evidence, conflicts)
            except BudgetExceeded:
                raise
            except Exception as error:
                generation_failure = _generation_failure(error)
                draft = {
                    "answerable": False,
                    "claims": [],
                    "refusal_reason": _generation_failure_message(generation_failure),
                }
        return WorkflowExecution(
            draft,
            tuple(evidence),
            scope,
            conflicts,
            {**coverage.to_dict(), "sufficient": sufficient},
            {
                "gap_retrievals": max(0, harness.usage.retrieval_rounds - 1),
                "gap_retrieval_failed": gap_failed,
                "query_frame": query_frame.to_dict(),
                **({"generation_failure": generation_failure} if generation_failure else {}),
            },
        )


class DeepResearchWorkflow(BaseWorkflow):
    name = WorkflowName.DEEP_RESEARCH

    def execute(self, request: Any, harness: ResearchRunHarness) -> WorkflowExecution:
        if not request.explicit_deep_research:
            raise PermissionError("DeepResearch 必须由用户显式开启。")
        with harness.step("SCOPE_CHECK"):
            scope = self._scope(request, harness)
        with harness.step("QUESTION_DECOMPOSE"):
            queries = [request.query, f"反例与不同路线：{request.query}"]
        evidence: list[Mapping[str, Any]] = []
        retrieval: Mapping[str, Any] = {}
        for query in queries[: harness.policy.max_retrieval_rounds]:
            with harness.step("MULTI_RETRIEVE"):
                retrieval = self._retrieve(request, harness, query)
                known = {str(value.get("evidence_id")) for value in evidence}
                evidence.extend(value for value in _selected(retrieval) if str(value.get("evidence_id")) not in known)
        discovery: Mapping[str, Any] = {}
        provisional: dict[str, Any] = {"evidence": [], "evidence_count": 0, "fulltext_jobs": []}
        external_search_failed = False
        if len(evidence) < 3 and harness.policy.max_external_searches:
            with harness.step("EXTERNAL_LITERATURE_DISCOVERY"):
                try:
                    discovery = self.gateway.invoke(
                        "literature.search",
                        {"query": request.query, "limit": 10},
                        context=_tool_context(self.name, request),
                        harness=harness,
                    )
                except (ConnectionError, TimeoutError, OSError):
                    external_search_failed = True
                    self._degrade_and_continue(harness, "continue_after_external_search_failure")
                    self._record_recovery(
                        harness,
                        RecoveryLevel.SCOPE_REDUCTION,
                        "reduce_to_local_verified_evidence",
                        "continuing",
                    )
                else:
                    raw_discovered = discovery.get("results")
                    discovered_papers = (
                        [value for value in raw_discovered if isinstance(value, Mapping)][:5]
                        if isinstance(raw_discovered, list)
                        else []
                    )
                    if request.project_root is not None:
                        provisional = register_abstract_evidence(
                            request.project_root,
                            request.session_id or harness.run_id,
                            discovered_papers,
                        )
                    else:
                        provisional_values = [
                            value
                            for ordinal, paper in enumerate(discovered_papers, 1)
                            if (value := abstract_evidence_from_paper(
                                paper,
                                session_id=request.session_id or harness.run_id,
                                ordinal=ordinal,
                            )) is not None
                        ]
                        provisional = {
                            "evidence": provisional_values,
                            "evidence_count": len(provisional_values),
                            "fulltext_jobs": [],
                        }
                    evidence.extend(
                        value
                        for value in provisional.get("evidence", [])
                        if isinstance(value, Mapping)
                        and str(value.get("evidence_id"))
                        not in {str(item.get("evidence_id")) for item in evidence}
                    )
        conflicts = _conflicts(retrieval)
        draft: Mapping[str, Any]
        generation_failure: dict[str, Any] | None = None
        if not evidence:
            draft = {"answerable": False, "claims": [], "refusal_reason": "本地证据不足；外部发现结果尚未验证。"}
        else:
            try:
                with harness.step("GENERATE"):
                    draft = self._generate(request, harness, scope, evidence, conflicts)
            except BudgetExceeded:
                raise
            except Exception as error:
                generation_failure = _generation_failure(error)
                draft = {
                    "answerable": False,
                    "claims": [],
                    "refusal_reason": _generation_failure_message(generation_failure),
                }
        return WorkflowExecution(
            draft,
            tuple(evidence),
            scope,
            conflicts,
            {"sufficient": len(evidence) >= 3},
            {
                "discovered_count": len(discovery.get("results", []))
                if isinstance(discovery.get("results"), list)
                else 0,
                "external_search_failed": external_search_failed,
                "provisional_evidence_count": int(provisional.get("evidence_count") or 0),
                "provisional_evidence_path": provisional.get("path"),
                **({"generation_failure": generation_failure} if generation_failure else {}),
            },
        )


class ResearchBootstrapWorkflow(BaseWorkflow):
    name = WorkflowName.RESEARCH_BOOTSTRAP

    def _resolve_job_result(
        self,
        value: Mapping[str, Any],
        request: Any,
        harness: ResearchRunHarness,
    ) -> tuple[Mapping[str, Any] | None, str | None]:
        job_id = value.get("job_id")
        if not job_id:
            return value, None
        status = self.gateway.invoke(
            "job.get_status",
            {"job_id": str(job_id)},
            context=_tool_context(self.name, request),
            harness=harness,
        )
        state = str(status.get("status") or "").casefold()
        if state == "succeeded":
            result = status.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("已完成 Job 缺少结构化 result。")
            return result, None
        if state in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Bootstrap Job {job_id} {state}: {status.get('error_type')}"
            )
        return None, str(job_id)

    def execute(self, request: Any, harness: ResearchRunHarness) -> WorkflowExecution:
        with harness.step("KNOWLEDGE_READINESS_CHECK"):
            scope = self._scope(request, harness)
        with harness.step("TOPIC_NORMALIZATION"):
            normalized = " ".join(request.query.split())
        with harness.step("PROVISIONAL_SCOPE"):
            provisional = {"query": normalized, "roles": ["high_quality_review", "latest_review", "foundational", "representative", "counterexample"]}
        with harness.step("LITERATURE_DISCOVERY"):
            discovery = self.gateway.invoke(
                "literature.search",
                {"query": normalized, "limit": 10},
                context=_tool_context(self.name, request),
                harness=harness,
            )
        raw = discovery.get("results")
        candidates = [dict(value) for value in raw if isinstance(value, Mapping)][:5] if isinstance(raw, list) else []
        with harness.step("CANDIDATE_SCREENING") as trace:
            trace["candidate_count"] = len(candidates)
        lightweight: list[Mapping[str, Any]] = []
        for value in candidates[:2]:
            paper_id = str(value.get("paper_id") or value.get("id") or "")
            if not paper_id:
                continue
            with harness.step("LIGHTWEIGHT_PARSE"):
                metadata = self.gateway.invoke(
                    "literature.get_metadata",
                    {"paper_id": paper_id},
                    context=_tool_context(self.name, request),
                    harness=harness,
                )
                lightweight.append(metadata)
        with harness.step("SCOPE_PROPOSAL"):
            proposal = {**provisional, "candidate_ids": [value.get("paper_id") for value in candidates]}
        authorized = request.bootstrap_confirmed or request.limited_autonomy
        if not authorized:
            return WorkflowExecution(
                {"answerable": False, "claims": [], "refusal_reason": "需要用户确认候选 Scope 后再正式入库。"},
                (),
                scope,
                (),
                {"sufficient": False},
                {"scope_proposal": proposal, "awaiting_confirmation": True},
            )
        ingested: list[str] = []
        for value in candidates[:1]:
            paper_id = str(value.get("paper_id") or value.get("id") or "")
            if not paper_id:
                continue
            with harness.step("FORMAL_INGESTION"):
                downloaded = self.gateway.invoke(
                    "literature.download",
                    {"paper_id": paper_id},
                    context=_tool_context(self.name, request),
                    harness=harness,
                )
                downloaded_result, pending_job_id = self._resolve_job_result(
                    downloaded, request, harness
                )
                if pending_job_id is not None:
                    return WorkflowExecution(
                        {
                            "answerable": False,
                            "claims": [],
                            "refusal_reason": "文献下载已提交后台任务，完成后可恢复原问题。",
                        },
                        (),
                        scope,
                        (),
                        {"sufficient": False},
                        {
                            "scope_proposal": proposal,
                            "pending_job_id": pending_job_id,
                            "pending_stage": "literature.download",
                            "resumed_original_question": False,
                        },
                    )
                assert downloaded_result is not None
                parsed = self.gateway.invoke(
                    "document.parse",
                    {
                        "path": str(downloaded_result.get("path") or ""),
                        "mode": "formal",
                    },
                    context=_tool_context(self.name, request),
                    harness=harness,
                )
                parsed_result, pending_job_id = self._resolve_job_result(
                    parsed, request, harness
                )
                if pending_job_id is not None:
                    return WorkflowExecution(
                        {
                            "answerable": False,
                            "claims": [],
                            "refusal_reason": "文献解析已提交后台任务，完成后可恢复原问题。",
                        },
                        (),
                        scope,
                        (),
                        {"sufficient": False},
                        {
                            "scope_proposal": proposal,
                            "pending_job_id": pending_job_id,
                            "pending_stage": "document.parse",
                            "resumed_original_question": False,
                        },
                    )
                assert parsed_result is not None
                if parsed_result.get("document_id"):
                    ingested.append(str(parsed_result["document_id"]))
        resumed_request = request
        if ingested:
            updated_scope = self.gateway.invoke(
                "workspace.update_scope",
                {
                    "workspace_id": request.workspace_id,
                    "scope_version": request.scope_version,
                    "document_ids": ingested,
                },
                context=_tool_context(self.name, request),
                harness=harness,
            )
            resumed_request = replace(
                request,
                scope_version=int(updated_scope["scope_version"]),
            )
            scope = self._scope(resumed_request, harness)
        with harness.step("RESUME_ORIGINAL_QUESTION"):
            retrieval = self._retrieve(resumed_request, harness, request.query)
            evidence = _selected(retrieval)
            generation_failure: dict[str, Any] | None = None
            if evidence:
                try:
                    draft = self._generate(
                        resumed_request,
                        harness,
                        scope,
                        evidence,
                        _conflicts(retrieval),
                    )
                except BudgetExceeded:
                    raise
                except Exception as error:
                    generation_failure = _generation_failure(error)
                    draft = {
                        "answerable": False,
                        "claims": [],
                        "refusal_reason": _generation_failure_message(generation_failure),
                    }
            else:
                draft = {
                    "answerable": False,
                    "claims": [],
                    "refusal_reason": "候选文献已处理，但原问题仍缺少 Selected Evidence。",
                }
        return WorkflowExecution(
            draft,
            evidence,
            scope,
            _conflicts(retrieval),
            {"sufficient": bool(evidence)},
            {
                "scope_proposal": proposal,
                "ingested_document_ids": ingested,
                "resumed_original_question": True,
                "resumed_scope_version": resumed_request.scope_version,
                "lightweight_count": len(lightweight),
                **({"generation_failure": generation_failure} if generation_failure else {}),
            },
        )
