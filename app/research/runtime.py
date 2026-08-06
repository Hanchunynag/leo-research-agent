"""Workflow 路由、Research Harness 运行时与 Unified Service 组装边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from app.research.context import ResearchContextManager
from app.research.harness import (
    BudgetExceeded,
    HarnessState,
    RecoveryLevel,
    ResearchBudgetPolicy,
    ResearchRunHarness,
    RunTraceStore,
)
from app.research.memory import ResearchStateStore
from app.research.tools import ToolGatewayRegistry, ToolHandler, build_default_gateway
from app.research.validation import ClaimEvidenceValidator, TieredClaimEvidenceValidator
from app.research.workflows import (
    AnswerGenerator,
    DeepResearchWorkflow,
    DirectQAWorkflow,
    RelationReasoningWorkflow,
    ResearchBootstrapWorkflow,
    WorkflowExecution,
    WorkflowName,
)


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    query: str
    workspace_id: str
    scope_version: int
    workflow: WorkflowName | None = None
    session_id: str | None = None
    top_k: int = 10
    explicit_deep_research: bool = False
    bootstrap_confirmed: bool = False
    limited_autonomy: bool = False
    workspace_summary: Mapping[str, Any] | None = None
    recent_conversation: tuple[Mapping[str, str], ...] = ()
    permissions: frozenset[str] = frozenset(
        {
            "knowledge.read",
            "workspace.read",
            "workspace.write",
            "literature.search",
            "literature.read",
            "literature.download",
            "document.parse",
            "job.read",
        }
    )

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.workspace_id.strip() or self.scope_version < 1:
            raise ValueError("WorkflowRequest query/workspace/scope 无效。")


_POLICIES = {
    WorkflowName.DIRECT_QA: ResearchBudgetPolicy(
        max_steps=10,
        max_llm_calls=1,
        max_tool_calls=2,
        max_retrieval_rounds=1,
        max_external_searches=0,
        max_repairs=1,
        max_context_tokens=4_000,
        max_total_tokens=10_000,
    ),
    WorkflowName.RELATION_REASONING: ResearchBudgetPolicy(
        max_steps=12,
        max_llm_calls=1,
        max_tool_calls=3,
        max_retrieval_rounds=2,
        max_external_searches=0,
        max_repairs=1,
        max_context_tokens=5_000,
        max_total_tokens=12_000,
    ),
    WorkflowName.DEEP_RESEARCH: ResearchBudgetPolicy(
        max_steps=18,
        max_llm_calls=2,
        max_tool_calls=8,
        max_retrieval_rounds=3,
        max_external_searches=2,
        max_repairs=1,
        max_context_tokens=8_000,
        max_total_tokens=24_000,
    ),
    WorkflowName.RESEARCH_BOOTSTRAP: ResearchBudgetPolicy(
        max_steps=22,
        max_llm_calls=1,
        max_tool_calls=12,
        max_retrieval_rounds=1,
        max_external_searches=1,
        max_repairs=1,
        max_context_tokens=6_000,
        max_total_tokens=16_000,
    ),
}


class ResearchRuntime:
    def __init__(
        self,
        gateway: ToolGatewayRegistry,
        generator: AnswerGenerator,
        *,
        contexts: ResearchContextManager | None = None,
        validator: ClaimEvidenceValidator | TieredClaimEvidenceValidator | None = None,
        state_store: ResearchStateStore | None = None,
        trace_store: RunTraceStore | None = None,
    ) -> None:
        self.gateway = gateway
        self.generator = generator
        self.contexts = contexts or ResearchContextManager()
        self.validator = validator or TieredClaimEvidenceValidator()
        self.state_store = state_store
        self.trace_store = trace_store

    @staticmethod
    def route(request: WorkflowRequest) -> WorkflowName:
        if request.workflow is not None:
            return request.workflow
        summary = request.workspace_summary or {}
        if int(summary.get("document_count") or 0) == 0 or any(
            bool(summary.get(key))
            for key in ("evidence_insufficient", "out_of_scope", "missing_dimensions")
        ):
            return WorkflowName.RESEARCH_BOOTSTRAP
        relation_markers = ("relationship", "relation", "how are", "关系", "关联", "机制")
        if any(value in request.query.casefold() for value in relation_markers):
            return WorkflowName.RELATION_REASONING
        return WorkflowName.DIRECT_QA

    def run(self, request: WorkflowRequest) -> dict[str, Any]:
        workflow_name = self.route(request)
        harness = ResearchRunHarness(workflow_name.value, _POLICIES[workflow_name])
        execution: WorkflowExecution | None = None
        validation: Any | None = None
        final_draft: dict[str, Any] = {
            "answerable": False,
            "claims": [],
            "refusal_reason": "运行尚未完成。",
        }
        try:
            harness.transition(HarnessState.CONTEXT_PREPARING)
            with harness.step("ROUTER_CONTEXT"):
                router_context = self.contexts.router_pack(
                    query=request.query,
                    workspace_id=request.workspace_id,
                    scope_version=request.scope_version,
                    workspace_summary=request.workspace_summary or {},
                    recent_conversation=request.recent_conversation,
                    harness=harness,
                )
            harness.transition(HarnessState.PLANNING)
            with harness.step("WORKFLOW_SELECT") as trace:
                trace["workflow"] = workflow_name.value
                trace["router_context_tokens"] = router_context.token_count
            workflow = self._workflow(workflow_name)
            harness.transition(HarnessState.EXECUTING)
            execution = workflow.execute(request, harness)
            final_draft = dict(execution.draft)
            harness.transition(HarnessState.EVALUATING)
            with harness.step("CLAIM_VALIDATE") as trace:
                validation = self.validator.validate(
                    final_draft,
                    execution.evidence,
                    scope_constraints=tuple(str(value) for value in execution.scope.get("constraints", [])),
                    conflicts=execution.conflicts,
                )
                trace.update(
                    {
                        "valid": validation.valid,
                        "issue_codes": [value.code for value in validation.issues],
                        "coverage": dict(execution.coverage),
                        "conflict_count": len(execution.conflicts),
                        "semantic_judge_calls": validation.judge_call_count,
                        "semantic_input_tokens": validation.semantic_input_tokens,
                        "semantic_output_tokens": validation.semantic_output_tokens,
                    }
                )
            if not validation.valid:
                harness.transition(HarnessState.RECOVERING)
                harness.consume("repairs")
                final_draft = self.validator.deterministic_repair(final_draft, validation)
                harness.recover(
                    RecoveryLevel.FORMAT_OR_ARGUMENT,
                    "remove_or_narrow_unsupported_claims",
                    outcome="succeeded" if final_draft.get("answerable") else "refused",
                )
                harness.transition(HarnessState.COMMITTING)
            else:
                harness.transition(HarnessState.COMMITTING)
            with harness.step("COMMIT_SAFE_STATE"):
                self._commit(request, harness, workflow_name, execution, final_draft)
            answerable = bool(final_draft.get("answerable"))
            harness.finish(
                HarnessState.COMPLETED if answerable else HarnessState.REFUSED,
                "completed" if answerable else "insufficient_verified_evidence",
            )
        except BudgetExceeded as error:
            final_draft = self._safe_termination(harness, str(error), RecoveryLevel.SAFE_TERMINATION)
        except Exception as error:
            final_draft = self._safe_termination(
                harness,
                type(error).__name__,
                RecoveryLevel.BACKEND_DEGRADATION,
            )
        selected_evidence: list[Mapping[str, Any]] = (
            list(execution.evidence) if execution else []
        )
        result: dict[str, Any] = {
            "query": request.query,
            "workflow": workflow_name.value,
            "answerable": bool(final_draft.get("answerable")),
            "claims": list(final_draft.get("claims") or []),
            "refusal_reason": final_draft.get("refusal_reason"),
            "selected_evidence": selected_evidence,
            "coverage": dict(execution.coverage) if execution else {},
            "conflicts": list(execution.conflicts) if execution else [],
            "validation": {
                "valid": bool(validation.valid) if validation is not None else False,
                "issues": [asdict(value) for value in validation.issues] if validation is not None else [],
                "semantic_results": [
                    asdict(value) for value in validation.semantic_results
                ]
                if validation is not None
                else [],
                "semantic_judge_calls": validation.judge_call_count
                if validation is not None
                else 0,
                "semantic_input_tokens": validation.semantic_input_tokens
                if validation is not None
                else 0,
                "semantic_output_tokens": validation.semantic_output_tokens
                if validation is not None
                else 0,
            },
            "diagnostics": harness.diagnostics(),
            "workflow_details": dict(execution.details) if execution else {},
        }
        if self.trace_store is not None:
            result["trace_path"] = str(
                self.trace_store.write(
                    harness,
                    {
                        "answerable": result["answerable"],
                        "selected_evidence_ids": [
                            value.get("evidence_id") for value in selected_evidence
                        ],
                    },
                )
            )
        return result

    def _workflow(self, name: WorkflowName) -> Any:
        values = {
            WorkflowName.DIRECT_QA: DirectQAWorkflow,
            WorkflowName.RELATION_REASONING: RelationReasoningWorkflow,
            WorkflowName.DEEP_RESEARCH: DeepResearchWorkflow,
            WorkflowName.RESEARCH_BOOTSTRAP: ResearchBootstrapWorkflow,
        }
        return values[name](self.gateway, self.contexts, self.generator)

    def _commit(
        self,
        request: WorkflowRequest,
        harness: ResearchRunHarness,
        workflow: WorkflowName,
        execution: WorkflowExecution,
        draft: Mapping[str, Any],
    ) -> None:
        if self.state_store is None:
            return
        if request.session_id:
            self.state_store.commit_session(
                request.session_id,
                {"workspace_id": request.workspace_id, "last_query": request.query, "last_workflow": workflow.value, "turn_count": 1},
            )
        self.state_store.commit_run(
            harness.run_id,
            {
                "workflow": workflow.value,
                "state": "committing",
                "termination_reason": None,
                "selected_evidence_ids": [value.get("evidence_id") for value in execution.evidence],
            },
        )
        supported = [dict(value) for value in draft.get("claims", []) if isinstance(value, Mapping)]
        if supported:
            current = self.state_store.workspace(request.workspace_id)
            existing = list(current.get("verified_conclusions") or [])
            self.state_store.commit_workspace(
                request.workspace_id,
                {"verified_conclusions": (existing + supported)[-100:]},
            )

    @staticmethod
    def _safe_termination(
        harness: ResearchRunHarness,
        reason: str,
        level: RecoveryLevel,
    ) -> dict[str, Any]:
        if harness.state in {HarnessState.EXECUTING, HarnessState.EVALUATING}:
            harness.transition(HarnessState.RECOVERING)
        if harness.state == HarnessState.RECOVERING:
            harness.recover(level, "save_state_and_terminate", outcome="refused")
            harness.transition(HarnessState.COMMITTING)
        if harness.state == HarnessState.COMMITTING:
            harness.finish(HarnessState.REFUSED, f"safe_termination:{reason}")
        elif harness.state not in {HarnessState.COMPLETED, HarnessState.FAILED, HarnessState.REFUSED}:
            # Context/Planning failure cannot legally enter RECOVERING; terminate as failed.
            harness.state = HarnessState.FAILED
            harness.state_history.append(HarnessState.FAILED.value)
            harness.termination_reason = f"failed_before_execution:{reason}"
        return {
            "answerable": False,
            "claims": [],
            "refusal_reason": "运行已安全终止；不会在证据或预算不足时生成确定性科研结论。",
        }


class HarnessAgentService:
    """兼容 answer 门面；内部只调用 ResearchRuntime，不接触知识后端。"""

    def __init__(
        self,
        runtime: ResearchRuntime,
        *,
        workspace_id: str = "default",
        scope_version: int = 1,
        workspace_summary: Mapping[str, Any] | None = None,
        session_store: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.workspace_id = workspace_id
        self.scope_version = scope_version
        self.workspace_summary = dict(workspace_summary or {"document_count": 1})
        self.session_store = session_store
        self.model_name = model_name

    def answer(
        self,
        query: str,
        *,
        session_id: str | None = None,
        workflow: str | None = None,
        deep_research: bool = False,
        bootstrap_confirmed: bool = False,
        workspace_summary: Mapping[str, Any] | None = None,
        force_new_topic: bool = False,
        include_context: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        sid, topic_id, session_created, recent = self._prepare_session(
            query,
            session_id=session_id,
            force_new_topic=force_new_topic,
        )
        selected = WorkflowName(workflow) if workflow else None
        raw = self.runtime.run(
            WorkflowRequest(
                query=query,
                workspace_id=self.workspace_id,
                scope_version=self.scope_version,
                workflow=selected,
                session_id=sid,
                explicit_deep_research=deep_research,
                bootstrap_confirmed=bootstrap_confirmed,
                workspace_summary=workspace_summary or self.workspace_summary,
                recent_conversation=recent,
            )
        )
        result = self._present(
            raw,
            session_id=sid,
            topic_id=topic_id,
            session_created=session_created,
            include_context=include_context,
        )
        self._persist_session(result, topic_id=topic_id)
        return result

    def _prepare_session(
        self,
        query: str,
        *,
        session_id: str | None,
        force_new_topic: bool,
    ) -> tuple[str | None, str | None, bool, tuple[Mapping[str, str], ...]]:
        if self.session_store is None:
            return session_id, None, False, ()
        session, created = self.session_store.get_or_create_session(
            session_id,
            query,
            model=self.model_name,
            provider=type(self.runtime.generator).__name__,
            configuration_fingerprint="research-harness-v1",
        )
        sid = str(session["session_id"])
        active = session.get("active_topic_id")
        if force_new_topic or not active:
            topic = self.session_store.create_topic(
                sid,
                relation="new_topic" if active else "same_topic",
                topic_summary=query,
                user_goal=query,
                entities=(),
                parent_topic_id=str(active) if force_new_topic and active else None,
            )
        else:
            topic = self.session_store.get_topic(sid, str(active))
            self.session_store.set_active_topic(sid, str(active))
        topic_id = str(topic["topic_id"])
        recent: list[Mapping[str, str]] = []
        for event in self.session_store.list_events(sid, topic_id):
            content = event.get("content")
            if not isinstance(content, Mapping):
                continue
            if event.get("event_type") == "user_query" and content.get("query"):
                recent.append({"role": "user", "content": str(content["query"])})
            elif event.get("event_type") == "answer":
                answer = content.get("answer") or content.get("refusal_reason")
                if answer:
                    recent.append({"role": "assistant", "content": str(answer)})
        return sid, topic_id, created, tuple(recent[-4:])

    @staticmethod
    def _present(
        raw: Mapping[str, Any],
        *,
        session_id: str | None,
        topic_id: str | None,
        session_created: bool,
        include_context: bool,
    ) -> dict[str, Any]:
        claims = [dict(value) for value in raw.get("claims", []) if isinstance(value, Mapping)]
        selected = [
            dict(value)
            for value in raw.get("selected_evidence", [])
            if isinstance(value, Mapping)
        ]
        evidence_by_id = {str(value.get("evidence_id")): value for value in selected}
        citations: list[dict[str, Any]] = []
        for claim in claims:
            evidence_ids = claim.get("evidence_ids")
            ids = evidence_ids if isinstance(evidence_ids, list) else []
            source_ids = claim.get("source_ids")
            sources = source_ids if isinstance(source_ids, list) else []
            for index, evidence_id in enumerate(ids):
                evidence = evidence_by_id.get(str(evidence_id))
                if evidence is None:
                    continue
                citations.append(
                    {
                        "claim_id": str(claim.get("claim_id") or ""),
                        "source_id": str(sources[index] if index < len(sources) else evidence.get("source_id") or ""),
                        "evidence_id": str(evidence_id),
                        "chunk_id": str(evidence.get("chunk_id") or ""),
                        "work_id": str(evidence.get("work_id") or evidence.get("document_id") or ""),
                        "document_id": str(evidence.get("document_id") or ""),
                        "title": str(evidence.get("title") or ""),
                        "section_path": list(evidence.get("section_path") or []),
                        "page_start": int(evidence.get("page_start") or 1),
                        "page_end": int(evidence.get("page_end") or evidence.get("page_start") or 1),
                        "block_ids": list(evidence.get("block_ids") or []),
                    }
                )
        answer = "\n".join(
            f"{str(claim.get('text') or '').strip()} "
            + "".join(f"[{source}]" for source in claim.get("source_ids", []))
            for claim in claims
            if str(claim.get("text") or "").strip()
        ).strip()
        answerable = bool(raw.get("answerable"))
        validation = dict(raw.get("validation") or {})
        validation["citations"] = citations
        outcome = {
            "code": "answered" if answerable else "insufficient_evidence",
            "stage": "completed" if answerable else "evaluation",
            "message": "回答已通过 Research Harness 验证。" if answerable else str(raw.get("refusal_reason") or "证据不足。"),
            "retryable": False,
        }
        harness = dict(raw.get("diagnostics") or {})
        result: dict[str, Any] = {
            "schema_version": "1.0",
            "query": str(raw.get("query") or ""),
            "answerable": answerable,
            "answer": answer,
            "claims": claims,
            "citations": citations,
            "refusal_reason": raw.get("refusal_reason"),
            "outcome": outcome,
            "validation": validation,
            "session": {
                "session_id": session_id,
                "topic_id": topic_id,
                "relation": "same_topic",
                "standalone_query": str(raw.get("query") or ""),
                "session_created": session_created,
            },
            "coverage": {
                "overall_sufficient": bool((raw.get("coverage") or {}).get("sufficient")),
                "coverage": dict(raw.get("coverage") or {}),
            },
            "retrieval_rounds": [
                {"round": value + 1}
                for value in range(int((harness.get("usage") or {}).get("retrieval_rounds") or 0))
            ],
            "selected_evidence": selected,
            "workflow": raw.get("workflow"),
            "workflow_details": dict(raw.get("workflow_details") or {}),
            "conflicts": list(raw.get("conflicts") or []),
            "diagnostics": {
                "retrieval_mode": "research_harness",
                "workflow": raw.get("workflow"),
                "harness": harness,
                "trace_path": raw.get("trace_path"),
            },
        }
        if include_context:
            result["context"] = {"selected_evidence": selected}
        return result

    def _persist_session(self, result: Mapping[str, Any], *, topic_id: str | None) -> None:
        if self.session_store is None or topic_id is None:
            return
        session = result.get("session")
        if not isinstance(session, Mapping) or not session.get("session_id"):
            return
        sid = str(session["session_id"])
        query = str(result.get("query") or "")
        self.session_store.append_event(sid, topic_id, "user_query", {"query": query})
        evidence = [dict(value) for value in result.get("selected_evidence", []) if isinstance(value, Mapping)]
        if evidence:
            self.session_store.append_event(sid, topic_id, "evidence_added", {"evidence": evidence})
            turn = len([value for value in self.session_store.list_events(sid, topic_id) if value.get("event_type") == "user_query"])
            for value in evidence:
                if value.get("chunk_id"):
                    self.session_store.register_evidence(sid, topic_id, value, turn)
        self.session_store.append_event(
            sid,
            topic_id,
            "answer",
            {
                "answerable": bool(result.get("answerable")),
                "answer": str(result.get("answer") or ""),
                "claims": list(result.get("claims") or []),
                "refusal_reason": result.get("refusal_reason"),
                "outcome": dict(result.get("outcome") or {}),
            },
        )
        self.session_store.append_event(
            sid,
            topic_id,
            "validation",
            dict(result.get("validation") or {}),
        )
        claims = [str(value.get("text") or "") for value in result.get("claims", []) if isinstance(value, Mapping)]
        self.session_store.append_event(
            sid,
            topic_id,
            "state_delta",
            {"confirmed_facts_added": claims if result.get("answerable") else [], "open_questions": [] if result.get("answerable") else [query]},
        )
        topic = self.session_store.get_topic(sid, topic_id)
        confirmed = [*topic.get("confirmed_facts", []), *(claims if result.get("answerable") else [])]
        self.session_store.update_topic_state(
            sid,
            topic_id,
            topic_summary=str(topic.get("topic_summary") or query),
            user_goal=str(topic.get("user_goal") or query),
            entities=topic.get("entities", []),
            confirmed_facts=confirmed,
            open_questions=[] if result.get("answerable") else [query],
        )


def unified_tool_handlers(
    knowledge: Any,
    workspaces: Any,
    *,
    extras: Mapping[str, ToolHandler] | None = None,
) -> dict[str, ToolHandler]:
    """组装层可访问服务；Agent/Workflow 只能看到 Tool Gateway。"""

    def retrieve(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        workspace_id = str(arguments["workspace_id"])
        scope_version = int(arguments["scope_version"])
        if workspace_id != str(context.get("workspace_id")) or scope_version != int(context.get("scope_version") or 0):
            raise PermissionError("Tool 参数与 Run scope 不一致。")
        if workspace_id != str(getattr(knowledge, "workspace_id", workspace_id)):
            raise PermissionError("Unified Knowledge Service workspace 不匹配。")
        value = knowledge.retrieve(
            str(arguments["query"]),
            limit=int(arguments.get("top_k") or 10),
            workspace_id=workspace_id,
            scope_version=scope_version,
        )
        results = value.get("results") if isinstance(value, Mapping) else None
        selected = (
            [
                {**dict(item), "source_id": str(item.get("source_id") or f"S{index}")}
                for index, item in enumerate(results, 1)
                if isinstance(item, Mapping) and item.get("evidence_state") == "selected"
            ]
            if isinstance(results, list)
            else []
        )
        diagnostics = dict(getattr(knowledge, "last_diagnostics", {}))
        intelligence = diagnostics.get("evidence_intelligence")
        if isinstance(intelligence, Mapping):
            raw_conflicts = intelligence.get("conflicts")
            if isinstance(raw_conflicts, list):
                diagnostics["conflicts"] = [
                    {"supporting_evidence_id": str(value[0]), "opposing_evidence_id": str(value[1])}
                    for value in raw_conflicts
                    if isinstance(value, (list, tuple)) and len(value) == 2
                ]
        return {"results": selected, "diagnostics": diagnostics}

    def read_scope(arguments: Mapping[str, Any], _: Mapping[str, Any]) -> Mapping[str, Any]:
        scope = workspaces.require_scope(str(arguments["workspace_id"]), int(arguments["scope_version"]))
        return {**asdict(scope), "constraints": []}

    def update_scope(arguments: Mapping[str, Any], _: Mapping[str, Any]) -> Mapping[str, Any]:
        workspace_id = str(arguments["workspace_id"])
        expected_version = int(arguments["scope_version"])
        current = workspaces.repository.get_workspace(workspace_id)
        if current is None or current.scope_version != expected_version:
            raise ValueError("Workspace Scope 已变化，拒绝基于过期版本更新。")
        # Tool 语义是把已确认文献加入 Scope；底层 replace API 接收完整集合。
        # 组合层必须合并当前成员，避免 Bootstrap 意外排除既有论文。
        current_ids = set(workspaces.active_document_ids(workspace_id, expected_version))
        requested_ids = {str(value) for value in arguments["document_ids"]}
        scope = workspaces.replace_scope_documents(
            workspace_id,
            current_ids | requested_ids,
        )
        return asdict(scope)

    handlers: dict[str, ToolHandler] = {
        "knowledge.retrieve": retrieve,
        "workspace.read_scope": read_scope,
        "workspace.update_scope": update_scope,
    }
    handlers.update(extras or {})
    return handlers


def build_research_runtime(
    project_root: Path,
    knowledge: Any,
    workspaces: Any,
    generator: AnswerGenerator,
    *,
    extra_tools: Mapping[str, ToolHandler] | None = None,
) -> ResearchRuntime:
    gateway = build_default_gateway(unified_tool_handlers(knowledge, workspaces, extras=extra_tools))
    return ResearchRuntime(
        gateway,
        generator,
        state_store=ResearchStateStore(project_root),
        trace_store=RunTraceStore(project_root),
    )


def build_harness_agent_service(
    project_root: Path,
    knowledge: Any,
    workspaces: Any,
    reasoning_provider: Any,
    *,
    session_store: Any | None = None,
    extra_tools: Mapping[str, ToolHandler] | None = None,
) -> HarnessAgentService:
    """生产组合入口；具体服务只在这里注入 Gateway handlers。"""

    from app.research.adapters import AgenticReasoningGeneratorAdapter

    workspace = workspaces.ensure_default()
    scope = workspaces.require_scope(workspace.workspace_id, workspace.scope_version)
    generator = AgenticReasoningGeneratorAdapter(reasoning_provider)
    runtime = build_research_runtime(
        project_root,
        knowledge,
        workspaces,
        generator,
        extra_tools=extra_tools,
    )
    return HarnessAgentService(
        runtime,
        workspace_id=workspace.workspace_id,
        scope_version=workspace.scope_version,
        workspace_summary={
            "document_count": len(set(scope.included_document_ids) - set(scope.excluded_document_ids)),
        },
        session_store=session_store,
        model_name=getattr(reasoning_provider, "model_name", None),
    )
