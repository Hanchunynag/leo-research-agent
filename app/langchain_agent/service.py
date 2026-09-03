"""生产 LangChain Agent：翻译 Tool → 双语 RAG Tool → 原有生成与证据验证。"""

from __future__ import annotations

import secrets
from time import perf_counter
from typing import Any, Mapping, Sequence, cast

from langchain_core.runnables import RunnableConfig

from app.research.adapters import AgenticReasoningGeneratorAdapter
from app.research.runtime import (
    HarnessAgentService,
    build_research_runtime,
    unified_tool_handlers,
)

from app.langchain_agent.skills import (
    AnswerGenerationSkill,
    BilingualRetrievalSkill,
    ClaimValidationSkill,
    ScopeReadSkill,
    TranslationSkill,
)
from app.langchain_agent.graph import (
    LLMActionDecider,
    LangGraphResearchRuntime,
    ResearchAgentState,
    is_interrupted,
)
from app.langchain_agent.tools import build_research_tool_handlers


class LangChainHarnessAgentService:
    """维持 Web/CLI answer 契约的 LangChain 编排门面。

    生成 Prompt、Claim 结构、Selected-Evidence 边界与验证器仍复用已经在
    生产验证过的实现。LangChain 负责所有 Agent 前置工具编排，并固定翻译
    Tool 在召回 Tool 之前执行，而不是交给模型自由决定。
    """

    def __init__(
        self,
        delegate: HarnessAgentService,
        *,
        translation_skill: TranslationSkill,
        skill_names: Sequence[str],
        tool_handlers: Mapping[str, Any] | None = None,
        action_decider: Any | None = None,
        scope_reader: Any | None = None,
        validation_llm_call: bool = False,
    ) -> None:
        self.delegate = delegate
        self.translation_skill = translation_skill
        self.skill_names = tuple(skill_names)
        self._graph_callback: Any | None = None
        self._answer_options: dict[str, Any] = {}
        self._validation_llm_call = validation_llm_call
        self._graph_runtime = LangGraphResearchRuntime(
            translation_skill.runnable,
            self._research_tool,
            scope_reader=scope_reader,
            action_decider=action_decider,
            tool_handlers=tool_handlers,
        )

    def _run_research(self, state: Mapping[str, Any]) -> dict[str, Any]:
        translation = state.get("translation")
        details = dict(translation) if isinstance(translation, Mapping) else {}
        values = details.get("retrieval_queries")
        kwargs = dict(state.get("kwargs") or {})
        planned = kwargs.pop("retrieval_queries", None)
        queries = (
            tuple(value for value in planned if isinstance(value, str) and value.strip())
            if isinstance(planned, (list, tuple))
            else (
                tuple(value for value in values if isinstance(value, str) and value.strip())
                if isinstance(values, list)
                else (str(state["query"]),)
            )
        )
        progress_callback = state.get("progress_callback")
        if callable(progress_callback):
            try:
                progress_callback("retrieval_start", "开始执行 Scope 检查和双语 RAG。", 0.32)
            except Exception:
                pass
        result = self.delegate.answer(
            str(state["query"]),
            retrieval_queries=queries,
            progress_callback=progress_callback,
            translation_llm_call=True,
            **kwargs,
        )
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics["retrieval_mode"] = "langchain_bilingual_rag"
        diagnostics["langchain"] = {
            "chain": "LangGraph(translation -> reference -> classify -> plan -> scope -> papers -> metadata -> agent -> tool -> final)",
            "skills": list(self.skill_names),
            "translation": {
                key: details[key]
                for key in (
                    "original_query",
                    "zh_query",
                    "en_query",
                    "retrieval_queries",
                    "translation_status",
                    "translation_failure_kind",
                    "output_language",
                    "llm_execution",
                )
                if key in details
            },
        }
        result["diagnostics"] = diagnostics
        workflow_details = dict(result.get("workflow_details") or {})
        workflow_details["bilingual_query"] = diagnostics["langchain"]["translation"]
        result["workflow_details"] = workflow_details
        return result

    def _research_tool(self, state: ResearchAgentState) -> Mapping[str, Any]:
        translation = state.get("translation")
        plan = state.get("research_plan")
        task_type = str(plan.get("task_type") or "") if isinstance(plan, Mapping) else ""
        structured_task = task_type in {
            "timeline",
            "compare",
            "author_analysis",
            "multi_paper_summary",
        }
        details = dict(translation) if isinstance(translation, Mapping) else {}
        translated_queries = details.get("retrieval_queries")
        base_queries = (
            tuple(value for value in translated_queries if isinstance(value, str) and value.strip())
            if isinstance(translated_queries, list)
            else (str(state["original_query"]),)
        )
        retrieval_queries = self._planned_retrieval_queries(
            base_queries,
            state,
            plan if isinstance(plan, Mapping) else {},
        )
        return self._run_research(
            {
                "query": state["original_query"],
                "translation": details,
                "kwargs": {
                    **self._answer_options,
                    "output_language": state.get("output_language", "en"),
                    # Structured plans use deterministic graph transitions;
                    # account only for controller calls that can actually run.
                    "agent_llm_calls": (
                        2
                        if self._graph_runtime.action_decider is not None and not structured_task
                        else 0
                    ),
                    "validation_llm_call": self._validation_llm_call,
                    "retrieval_queries": retrieval_queries,
                    "target_document_ids": self._target_document_ids(state, structured_task),
                    "paper_filters": self._paper_filters(
                        state,
                        plan if isinstance(plan, Mapping) else {},
                        structured_task,
                    ),
                    "planner_result": dict(plan) if isinstance(plan, Mapping) else {},
                    "timeline_metadata": list(state.get("publication_metadata") or []),
                    "agent_instructions": self._agent_instructions(
                        plan if isinstance(plan, Mapping) else {}
                    ),
                },
                "progress_callback": self._graph_callback,
            }
        )

    @staticmethod
    def _target_document_ids(
        state: Mapping[str, Any],
        structured_task: bool,
    ) -> tuple[str, ...]:
        if not structured_task:
            return ()
        return tuple(
            dict.fromkeys(
                str(value.get("document_id") or "").strip()
                for value in state.get("papers") or state.get("selected_documents") or []
                if isinstance(value, Mapping) and str(value.get("document_id") or "").strip()
            )
        )

    @staticmethod
    def _paper_filters(
        state: Mapping[str, Any],
        plan: Mapping[str, Any],
        structured_task: bool,
    ) -> dict[str, Any]:
        """Translate bounded planner constraints into Paper-level filters."""
        filters: dict[str, Any] = {}
        constraints = plan.get("time_constraints")
        if isinstance(constraints, Mapping):
            for key in ("year_from", "year_to"):
                value = constraints.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    filters[key] = value
        if structured_task:
            document_ids = list(
                LangChainHarnessAgentService._target_document_ids(state, True)
            )
            if document_ids:
                filters["document_ids"] = document_ids
        return filters

    @staticmethod
    def _planned_retrieval_queries(
        base_queries: Sequence[str],
        state: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Add bounded, paper-specific evidence probes for structured tasks.

        A corpus-level query alone can return many excellent chunks from a few
        papers while omitting one paper that the timeline or comparison has
        explicitly promised to cover.  The planner therefore adds an exact
        title probe per resolved paper; retrieval, ranking, evidence selection,
        and all validation remain owned by the existing RAG pipeline.
        """

        values = [value.strip() for value in base_queries if isinstance(value, str) and value.strip()]
        if not plan.get("per_paper_evidence"):
            return tuple(dict.fromkeys(values))
        for paper in state.get("papers") or state.get("selected_documents") or []:
            if not isinstance(paper, Mapping):
                continue
            title = str(paper.get("title") or "").strip()
            if title:
                values.append(f"{title} method contribution limitation")
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _agent_instructions(plan: Mapping[str, Any]) -> tuple[str, ...]:
        """Convert deterministic Agent plans into synthesis-only constraints."""

        task_type = str(plan.get("task_type") or "")
        if task_type == "timeline":
            return (
                "按已提供的发表年份从早到晚输出原子 claims；年份未知的论文最后输出。",
                "每个 claim 仅总结有 Selected Evidence 支持的方法、贡献或局限，并同时提供 category、source_ids 和 evidence_ids。",
            )
        if task_type == "compare":
            dimensions = plan.get("comparison_dimensions")
            labels = ", ".join(str(value) for value in dimensions) if isinstance(dimensions, list) else "method, data_and_experiment, contribution, limitation"
            return (
                f"按比较维度 {labels} 输出可验证的原子 claims，并在 category 中标示对应维度。",
                "每个比较结论必须仅使用对应论文的 Selected Evidence，并提供 source_ids 和 evidence_ids；缺少证据时明确拒答。",
            )
        if task_type == "author_analysis":
            return (
                "将作者研究方向与按年份的变化分开陈述；作者归属和年份使用 metadata，技术结论必须引用 Selected Evidence。",
            )
        return ()

    @staticmethod
    def _documents_from_evidence(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            document_id = str(value.get("document_id") or "")
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            documents.append(
                {
                    "document_id": document_id,
                    "title": str(value.get("title") or document_id),
                    "year": value.get("year"),
                }
            )
        return documents

    def _session_documents(self, session_id: str | None) -> list[dict[str, Any]]:
        if self.delegate.session_store is None or not session_id:
            return []
        try:
            return self._documents_from_evidence(
                self.delegate.session_store.list_evidence(session_id)
            )
        except (KeyError, ValueError, OSError):
            return []

    def answer(
        self,
        query: str,
        *,
        langchain_config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行固定 Skill Chain；传入 callbacks 可查看完整 LangChain 运行树。"""

        thread_id = str(
            (langchain_config or {}).get("configurable", {}).get("thread_id")
            if isinstance(langchain_config, Mapping)
            else ""
        ) or str(kwargs.get("session_id") or "") or f"thread_{secrets.token_hex(8)}"
        self._graph_callback = kwargs.get("progress_callback")
        self._answer_options = {
            key: value for key, value in kwargs.items() if key != "progress_callback"
        }
        is_timeline_query = any(
            marker in query.casefold()
            for marker in ("演进", "演化", "研究路线", "发展路线", "技术路线", "时间线", "evolution", "progression", "roadmap", "timeline")
        )
        graph_started = perf_counter()
        initial: ResearchAgentState = {
            "messages": [{"role": "user", "content": query}],
            "original_query": query,
            "workspace_id": self.delegate.workspace_id,
            "scope_version": self.delegate.scope_version,
            "last_search_results": self._session_documents(kwargs.get("session_id")),
            "selected_documents": [],
            "retrieval_evidence": [],
            "retrieved_evidence": [],
            "evidence_gap": [],
            "iteration": 0,
            # One metadata lookup is performed per paper before the final RAG
            # call; allow the timeline skill to cover the whole workspace.
            "max_iterations": 12 if is_timeline_query else 4,
            "pending_question": None,
            "final_answer": None,
            "observation": None,
            "trace": [],
            "tool_calls": [],
            "state_changes": [],
            "publication_metadata": [],
        }
        state = self._graph_runtime.invoke(
            initial,
            thread_id=thread_id,
            progress_callback=kwargs.get("progress_callback"),
        )
        if is_interrupted(state):
            return self._clarification_result(state, thread_id)
        result = dict(state.get("result") or {})
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics["langgraph"] = self._graph_diagnostics(
            state, thread_id, elapsed_ms=(perf_counter() - graph_started) * 1000
        )
        result["diagnostics"] = diagnostics
        return result

    def resume(
        self,
        thread_id: str,
        user_input: str,
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Resume a clarification-paused graph with the same LangGraph thread."""

        self._graph_callback = progress_callback
        graph_started = perf_counter()
        state = self._graph_runtime.resume(
            thread_id,
            user_input,
            progress_callback=progress_callback,
        )
        if is_interrupted(state):
            return self._clarification_result(state, thread_id)
        result = dict(state.get("result") or {})
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics["langgraph"] = self._graph_diagnostics(
            state, thread_id, elapsed_ms=(perf_counter() - graph_started) * 1000
        )
        result["diagnostics"] = diagnostics
        return result

    @staticmethod
    def _graph_diagnostics(
        state: Mapping[str, Any], thread_id: str, *, elapsed_ms: float | None = None
    ) -> dict[str, Any]:
        usage = list(state.get("llm_usage") or [])
        return {
            "thread_id": thread_id,
            "current_skill": state.get("current_skill"),
            "current_task": state.get("current_task"),
            "task_type": state.get("task_type"),
            "research_intent": state.get("research_intent"),
            "keywords": list(state.get("keywords") or []),
            "time_constraints": dict(state.get("time_constraints") or {}),
            "expected_evidence": list(state.get("expected_evidence") or []),
            "need_retrieval": bool(state.get("need_retrieval", True)),
            "research_plan": dict(state.get("research_plan") or {}),
            "papers": list(state.get("papers") or []),
            "timeline": list(state.get("timeline") or []),
            "evidence_by_paper": dict(state.get("evidence_by_paper") or {}),
            "reference_map": dict(state.get("reference_map") or {}),
            "iteration": int(state.get("iteration") or 0),
            "tool_calls": list(state.get("tool_calls") or []),
            "state_changes": list(state.get("state_changes") or []),
            "steps": list(state.get("trace") or []),
            "latency_ms": round(elapsed_ms, 3) if elapsed_ms is not None else None,
            "token_usage": {
                "calls": usage,
                "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
            },
            "final_answer": state.get("final_answer"),
            "evidence_ids": [
                str(item.get("evidence_id"))
                for item in state.get("retrieved_evidence", [])
                if isinstance(item, Mapping) and item.get("evidence_id")
            ],
            "llm_usage": usage,
            "publication_metadata": list(state.get("publication_metadata") or []),
        }

    @staticmethod
    def _clarification_result(state: Mapping[str, Any], thread_id: str) -> dict[str, Any]:
        question = str(state.get("pending_question") or "请补充论文范围。")
        return {
            "schema_version": "1.0",
            "query": str(state.get("original_query") or ""),
            "answerable": False,
            "answer": question,
            "claims": [],
            "citations": [],
            "refusal_reason": question,
            "outcome": {
                "code": "clarification_required",
                "stage": "reference_resolver",
                "message": question,
                "retryable": True,
            },
            "validation": {"valid": False, "issues": []},
            "session": {"session_id": None, "topic_id": None, "relation": "clarification", "standalone_query": str(state.get("original_query") or "")},
            "coverage": {"overall_sufficient": False, "coverage": []},
            "retrieval_rounds": [],
            "selected_evidence": [],
            "workflow": "langgraph",
            "workflow_details": {"pending_question": question},
            "diagnostics": {"retrieval_mode": "langchain_bilingual_rag", "langgraph": {**LangChainHarnessAgentService._graph_diagnostics(state, thread_id), "interrupted": True}},
        }


def build_langchain_agent_service(
    project_root: Any,
    knowledge: Any,
    workspaces: Any,
    reasoning_provider: Any,
    *,
    session_store: Any | None = None,
    extra_tools: Mapping[str, Any] | None = None,
) -> LangChainHarnessAgentService:
    """唯一生产组装点：Agent 检索必经 LangChain 双 Tool 编排。"""

    workspace = workspaces.ensure_default()
    scope = workspaces.require_scope(workspace.workspace_id, workspace.scope_version)
    retrieval_skill = BilingualRetrievalSkill(knowledge)
    scope_skill = ScopeReadSkill(workspaces)
    translation_skill = TranslationSkill(reasoning_provider)
    handlers = unified_tool_handlers(knowledge, workspaces, extras=extra_tools)
    local_handlers = build_research_tool_handlers(
        project_root, workspaces, translation_tool=translation_skill.tool
    )
    for name, handler in local_handlers.items():
        external_handler = handlers.get(name)
        if name == "literature.resolve_publication_date" and external_handler is not None:
            def resolve_publication_date(
                arguments: Mapping[str, Any],
                context: Mapping[str, Any],
                *,
                local=handler,
                external=external_handler,
            ) -> Mapping[str, Any]:
                # Canonical papers already carry publication year.  Use it
                # first to keep a six-paper timeline fast and deterministic;
                # only unresolved titles go to external academic APIs.
                local_result = dict(local(arguments, context))
                if local_result.get("status") == "resolved" and local_result.get("publication_year"):
                    return local_result
                return dict(external(arguments, context))

            handlers[name] = resolve_publication_date
        else:
            handlers.setdefault(name, handler)
    handlers["knowledge.retrieve"] = retrieval_skill.gateway_handler()
    handlers["workspace.read_scope"] = scope_skill.gateway_handler()
    generation_skill = AnswerGenerationSkill(
        AgenticReasoningGeneratorAdapter(reasoning_provider)
    )
    provider = getattr(reasoning_provider, "provider", None)
    # Cross-paper synthesis/evolution claims are high-risk.  Give the existing
    # tiered validator one bounded semantic Judge call when the production
    # ChatModel is available; fixture providers keep the offline path.
    semantic_judge = None
    if callable(getattr(provider, "chat_completion", None)) and hasattr(provider, "config"):
        from app.langchain_agent.skills import ChatCompletionHighRiskJudge

        semantic_judge = ChatCompletionHighRiskJudge(provider)
    validation_skill = ClaimValidationSkill(high_risk_judge=semantic_judge)
    runtime = build_research_runtime(
        project_root,
        knowledge,
        workspaces,
        generation_skill,
        extra_tools=handlers,
        validator=cast(Any, validation_skill),
    )
    delegate = HarnessAgentService(
        runtime,
        workspace_id=workspace.workspace_id,
        scope_version=workspace.scope_version,
        workspace_summary={
            "document_count": len(
                set(scope.included_document_ids) - set(scope.excluded_document_ids)
            ),
        },
        session_store=session_store,
        model_name=getattr(reasoning_provider, "model_name", None),
    )
    # Existing fixture providers intentionally do not opt into a third controller
    # call; the production OpenAI-compatible provider does. This keeps old tests
    # and lightweight deployments compatible while enabling genuine LLM decisions.
    action_decider = (
        LLMActionDecider(provider)
        if callable(getattr(provider, "chat_completion", None)) and hasattr(provider, "config")
        else None
    )
    return LangChainHarnessAgentService(
        delegate,
        translation_skill=translation_skill,
        skill_names=(
            type(translation_skill).__name__,
            scope_skill.name,
            retrieval_skill.name,
            type(generation_skill).__name__,
            type(validation_skill).__name__,
        ),
        tool_handlers=handlers,
        action_decider=action_decider,
        scope_reader=lambda state: scope_skill.read_scope(
            state["workspace_id"], state["scope_version"]
        ),
        validation_llm_call=semantic_judge is not None,
    )
