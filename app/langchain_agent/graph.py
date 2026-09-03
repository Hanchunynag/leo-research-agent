"""Stateful LangGraph runtime around the existing LangChain/RAG services.

The graph is deliberately thin: retrieval and evidence validation remain owned by the
existing Research Harness.  LangGraph owns the durable conversation state, action loop,
clarification interrupt, and resume boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.generation.openai_compatible import parse_json_object
from app.langchain_agent.planning import (
    ResearchTaskType,
    build_research_plan,
    classify_research_task,
    group_evidence_by_paper,
    sort_timeline,
)


class ResearchAgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    current_task: str
    current_skill: str | None
    workspace_id: str
    scope_version: int
    last_search_results: list[dict[str, Any]]
    selected_documents: list[dict[str, Any]]
    reference_map: dict[str, Any]
    original_query: str
    translated_query: str | None
    translation: dict[str, Any]
    retrieval_evidence: list[dict[str, Any]]
    retrieved_evidence: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    answer: str | None
    need_more_search: bool
    evidence_gap: list[str]
    iteration: int
    max_iterations: int
    pending_question: str | None
    clarification_response: str | None
    final_answer: str | None
    result: dict[str, Any] | None
    observation: dict[str, Any] | None
    action: dict[str, Any] | None
    last_tool: str | None
    trace: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    state_changes: list[dict[str, Any]]
    output_language: str
    llm_usage: list[dict[str, Any]]
    scope: dict[str, Any]
    publication_metadata: list[dict[str, Any]]
    task_type: ResearchTaskType
    research_plan: dict[str, Any]
    research_intent: str
    keywords: list[str]
    time_constraints: dict[str, Any]
    expected_evidence: list[str]
    need_retrieval: bool
    papers: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    evidence_by_paper: dict[str, list[str]]
    paper_candidates: list[dict[str, Any]]
    candidate_paper_ids: list[str]
    hierarchical_retrieval: dict[str, Any]
    coverage_report: dict[str, Any]
    provisional_evidence: list[dict[str, Any]]
    fulltext_jobs: list[dict[str, Any]]


ActionDecider = Callable[[ResearchAgentState], Mapping[str, Any]]
ResearchTool = Callable[[ResearchAgentState], Mapping[str, Any]]


class LLMActionDecider:
    """Use the configured Chat Completions model for the Agent decision node."""

    def __init__(self, provider: Any) -> None:
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("Agent action decider 需要 chat_completion Provider。")
        self.provider = provider

    def __call__(self, state: ResearchAgentState) -> Mapping[str, Any]:
        if state.get("pending_question") and not state.get("clarification_response"):
            fallback: dict[str, Any] = {"type": "clarify", "question": state["pending_question"]}
        elif state.get("observation") is not None:
            fallback = {"type": "final"}
        else:
            fallback = {"type": "tool", "tool_name": "knowledge.retrieve", "arguments": {}}
        try:
            response = self.provider.chat_completion(
                [
                    {"role": "system", "content": (
                        "You are the LangGraph research Agent controller. Return one JSON action only: "
                        "{type: tool|clarify|final, tool_name?, arguments?, question?}. "
                        "Use knowledge.retrieve for evidence questions, clarify missing references, "
                        "and final only after an observation. Never answer the research question."
                    )},
                    {"role": "user", "content": str({
                        "query": state.get("original_query", ""),
                        "skill": state.get("current_skill"),
                        "task_type": state.get("task_type"),
                        "plan": state.get("research_plan", {}),
                        "has_observation": state.get("observation") is not None,
                        "pending_question": state.get("pending_question"),
                        "publication_metadata": state.get("publication_metadata", []),
                        "iteration": state.get("iteration", 0),
                    })},
                ],
                max_tokens=400,
            )
            content = response["choices"][0]["message"]["content"]
            action = parse_json_object(str(content))
            if action.get("type") not in {"tool", "clarify", "final"}:
                raise ValueError("action.type 无效")
            # The LLM controller is deliberately not a free-form tool caller.
            # All preparation tools are fixed graph transitions; a controller
            # may only request the bounded local retrieval gateway.
            if action.get("type") == "tool" and action.get("tool_name") != "knowledge.retrieve":
                raise ValueError("LLM controller 只能调用固定 knowledge.retrieve")
            if action.get("type") == "final" and state.get("observation") is None:
                return fallback
            if action.get("type") == "clarify" and not state.get("pending_question"):
                return fallback
            usage = response.get("usage") if isinstance(response, Mapping) else None
            safe_usage = {
                str(key): value
                for key, value in (usage or {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if safe_usage:
                action["_usage"] = safe_usage
            return action
        except Exception:
            return fallback


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {
        **{character: index for index, character in enumerate("零一二三四五六七八九")},
        "两": 2,
    }
    if value == "十":
        return 10
    if len(value) == 2 and value[0] == "十":
        return 10 + digits.get(value[1], 0)
    if len(value) == 2 and value[1] == "十":
        return digits.get(value[0], 0) * 10
    if len(value) == 3 and value[1] == "十":
        return digits.get(value[0], 0) * 10 + digits.get(value[2], 0)
    return digits.get(value)


def resolve_references(
    query: str,
    documents: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Resolve ordinal paper references, or return a user-facing clarification."""

    normalized = query.casefold()
    mapping = {str(index + 1): dict(value) for index, value in enumerate(documents)}
    match = re.search(r"前([一二两三四五六七八九十\d]+)篇", query)
    count = _number(match.group(1)) if match else None
    if count is None and any(marker in normalized for marker in ("这四篇", "这几篇", "these papers", "the four papers")):
        count = 4
    is_reference_task = any(marker in normalized for marker in ("比较", "对比", "总结", "these papers", "前"))
    if is_reference_task and count is not None:
        if len(documents) < count:
            return [], mapping, f"我找不到可对应的 {count} 篇论文。请先列出论文，或明确提供要比较的论文名称/编号。"
        return [dict(value) for value in documents[:count]], mapping, None
    if is_reference_task and any(marker in normalized for marker in ("这四篇", "这几篇", "这篇", "刚才那几篇")) and not documents:
        return [], mapping, "当前会话没有可解析的论文指代。请先搜索或提供论文名称/编号。"
    return [dict(value) for value in documents[:count]] if count else [], mapping, None


class ReferenceResolver:
    """Named node service so reference resolution is independently testable."""

    def resolve(
        self, query: str, documents: list[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        return resolve_references(query, documents)


def _skill_for(task_type: ResearchTaskType) -> str:
    return {
        "timeline": "timeline_research",
        "compare": "paper_compare",
        "author_analysis": "author_analysis",
        "multi_paper_summary": "multi_paper_summary",
        "literature_search": "bilingual_search",
        "paper_summary": "paper_summary",
        "direct_qa": "direct_qa",
    }[task_type]


@dataclass
class LangGraphResearchRuntime:
    """Compile and execute the state graph using an in-memory development saver."""

    translation_runnable: Any
    research_tool: ResearchTool
    scope_reader: Callable[[ResearchAgentState], Mapping[str, Any]] | None = None
    tool_handlers: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]] | None = None
    action_decider: ActionDecider | None = None
    max_iterations: int = 4
    checkpointer: Any | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations 必须大于 0。")
        self.checkpointer = self.checkpointer or InMemorySaver()
        self._callback: Callable[..., Any] | None = None
        self.graph = self._compile()

    def _record(self, state: ResearchAgentState, node: str, **details: Any) -> list[dict[str, Any]]:
        trace = list(state.get("trace") or [])
        trace.append({"node": node, "iteration": int(state.get("iteration") or 0), **details})
        return trace

    def _translation(self, state: ResearchAgentState) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": state["original_query"]}
        if self._callback is not None:
            payload["progress_callback"] = self._callback
        result = self.translation_runnable.invoke(payload)
        translation = dict(result) if isinstance(result, Mapping) else {}
        return {
            "translated_query": str(translation.get("en_query") or ""),
            "translation": translation,
            "output_language": str(translation.get("output_language") or "en"),
            "trace": self._record(state, "translation_skill", status="succeeded"),
        }

    @staticmethod
    def _reference(state: ResearchAgentState) -> dict[str, Any]:
        selected, mapping, question = resolve_references(
            state["original_query"],
            list(state.get("last_search_results") or state.get("selected_documents") or []),
        )
        result: dict[str, Any] = {
            "selected_documents": selected,
            "reference_map": mapping,
            "pending_question": question,
            "current_task": state["original_query"],
        }
        result["trace"] = LangGraphResearchRuntime._static_trace(state, "reference_resolver", status="clarify" if question else "succeeded")
        return result

    def _scope(self, state: ResearchAgentState) -> dict[str, Any]:
        if self.scope_reader is None:
            return {"trace": self._record(state, "scope_read", status="skipped")}
        scope = dict(self.scope_reader(state))
        return {"scope": scope, "trace": self._record(state, "scope_read", status="succeeded")}

    def _classify_task(self, state: ResearchAgentState) -> dict[str, Any]:
        task_type = classify_research_task(state["original_query"])
        return {
            "task_type": task_type,
            "current_skill": _skill_for(task_type),
            "trace": self._record(state, "classify_task", task_type=task_type),
        }

    def _plan_research(self, state: ResearchAgentState) -> dict[str, Any]:
        task_type = state.get("task_type") or "direct_qa"
        plan = build_research_plan(task_type, state.get("original_query", ""))
        return {
            "research_plan": plan,
            "research_intent": str(plan.get("research_intent") or task_type),
            "keywords": [str(value) for value in plan.get("keywords") or []],
            "time_constraints": dict(plan.get("time_constraints") or {}),
            "expected_evidence": [str(value) for value in plan.get("expected_evidence") or []],
            "need_retrieval": bool(plan.get("semantic_retrieval", True)),
            "trace": self._record(state, "research_planner", plan=plan),
        }

    def _resolve_papers(self, state: ResearchAgentState) -> dict[str, Any]:
        """Resolve references/session papers, or list the scoped corpus when needed."""

        papers = [dict(value) for value in state.get("selected_documents") or []]
        if not papers:
            papers = [dict(value) for value in state.get("last_search_results") or []]
        task_type = state.get("task_type") or "direct_qa"
        calls = list(state.get("tool_calls") or [])
        listed = False
        if not papers and task_type in {"timeline", "multi_paper_summary", "author_analysis"}:
            handler = (self.tool_handlers or {}).get("workspace.list_documents")
            if handler is not None:
                arguments = {
                    "workspace_id": state["workspace_id"],
                    "scope_version": state["scope_version"],
                }
                result = handler(arguments, state)
                raw = result.get("documents") if isinstance(result, Mapping) else None
                papers = (
                    [dict(value) for value in raw if isinstance(value, Mapping)]
                    if isinstance(raw, list)
                    else []
                )
                calls.append(
                    {
                        "tool_name": "workspace.list_documents",
                        "arguments": arguments,
                        "status": "succeeded",
                    }
                )
                listed = True
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for paper in papers:
            identity = str(
                paper.get("document_id") or paper.get("paper_id") or paper.get("title") or ""
            )
            if identity and identity not in seen:
                seen.add(identity)
                unique.append(paper)
        return {
            "papers": unique,
            "selected_documents": unique or list(state.get("selected_documents") or []),
            "tool_calls": calls,
            "trace": self._record(
                state,
                "resolve_papers",
                paper_count=len(unique),
                source="workspace" if listed else "reference_or_session",
            ),
        }

    def _fetch_metadata(self, state: ResearchAgentState) -> dict[str, Any]:
        plan = state.get("research_plan") or {}
        if not plan.get("requires_metadata"):
            return {"trace": self._record(state, "fetch_metadata", status="skipped")}
        handler = (self.tool_handlers or {}).get("literature.resolve_publication_date")
        papers = list(state.get("papers") or state.get("selected_documents") or [])
        if handler is None or not papers:
            return {
                "timeline": sort_timeline(state.get("publication_metadata") or []),
                "trace": self._record(
                    state,
                    "fetch_metadata",
                    status="skipped",
                    reason="tool_unavailable" if handler is None else "no_resolved_papers",
                ),
            }
        resolved: list[dict[str, Any]] = []
        calls = list(state.get("tool_calls") or [])
        for paper in papers:
            title = str(paper.get("title") or "").strip()
            if not title:
                continue
            arguments = {"title": title}
            try:
                observation = handler(arguments, state)
                value = dict(observation) if isinstance(observation, Mapping) else {}
                tool_status = "succeeded"
            except Exception as error:
                # Metadata helps organise a research answer but cannot make
                # existing local evidence unavailable.  Preserve the failure
                # in state and continue into the normal RAG path.
                value = {
                    "query_title": title,
                    "matched_title": "",
                    "publication_year": None,
                    "status": "unavailable",
                    "failure_type": type(error).__name__,
                }
                tool_status = "failed"
            value.update(
                {
                    "document_id": paper.get("document_id"),
                    "paper_id": paper.get("paper_id"),
                    "title": title,
                }
            )
            resolved.append(value)
            calls.append(
                {
                    "tool_name": "literature.resolve_publication_date",
                    "arguments": arguments,
                    "status": tool_status,
                }
            )
        return {
            "publication_metadata": resolved,
            "timeline": sort_timeline(resolved) if state.get("task_type") == "timeline" else [],
            "tool_calls": calls,
            "trace": self._record(
                state,
                "fetch_metadata",
                status="succeeded",
                resolved_count=len(resolved),
            ),
        }

    @staticmethod
    def _static_trace(state: ResearchAgentState, node: str, **details: Any) -> list[dict[str, Any]]:
        return [*(state.get("trace") or []), {"node": node, "iteration": int(state.get("iteration") or 0), **details}]

    def _agent(self, state: ResearchAgentState) -> dict[str, Any]:
        at_limit = int(state.get("iteration") or 0) >= int(state.get("max_iterations") or self.max_iterations)
        structured_task = state.get("task_type") in {
            "timeline",
            "compare",
            "author_analysis",
            "multi_paper_summary",
        }
        action: dict[str, Any]
        if at_limit:
            action = {"type": "final"}
        elif state.get("pending_question") and not state.get("clarification_response"):
            action = {"type": "clarify", "question": state["pending_question"]}
        elif structured_task and state.get("observation") is None:
            # These plans have already deterministically selected their paper
            # scope and metadata.  Go straight to the canonical RAG tool,
            # rather than spending an additional LLM call deciding the same.
            action = {"type": "tool", "tool_name": "knowledge.retrieve", "arguments": {}}
        elif structured_task and state.get("observation") is not None:
            # The Harness owns generation and validation; the graph only needs
            # to hand its verified retrieval result to the final node.
            action = {"type": "final"}
        elif state.get("observation") is not None and str(state.get("last_tool") or "") != "knowledge.retrieve":
            # Metadata/document tools are preparatory observations, not final
            # research answers.  Continue into the canonical RAG Tool.
            action = {"type": "tool", "tool_name": "knowledge.retrieve", "arguments": {}}
        elif self.action_decider is not None:
            action = dict(self.action_decider(state))
        elif state.get("observation") is not None:
            last_tool = str(state.get("last_tool") or "")
            if last_tool == "document.get_outline":
                action = {
                    "type": "tool",
                    "tool_name": "document.read",
                    "arguments": {
                        "document_id": str((state.get("selected_documents") or [{}])[0].get("document_id") or ""),
                        "neighbor_count": 1,
                    },
                }
            elif last_tool in {"literature.search", "document.read"}:
                action = {
                    "type": "tool",
                    "tool_name": "knowledge.retrieve",
                    "arguments": {},
                }
            else:
                action = {"type": "final"}
        else:
            tool_name = "knowledge.retrieve"
            if state.get("current_skill") == "paper_summary" and state.get("selected_documents") and "document.get_outline" in (self.tool_handlers or {}):
                tool_name = "document.get_outline"
            elif state.get("current_skill") == "bilingual_search" and "literature.search" in (self.tool_handlers or {}):
                tool_name = "literature.search"
            arguments: dict[str, Any] = {
                "query": state["original_query"],
                "query_variants": [
                    value
                    for value in (
                        state.get("original_query"),
                        (state.get("translation") or {}).get("zh_query")
                        if isinstance(state.get("translation"), Mapping)
                        else None,
                        (state.get("translation") or {}).get("en_query")
                        if isinstance(state.get("translation"), Mapping)
                        else state.get("translated_query"),
                    )
                    if isinstance(value, str) and value.strip()
                ],
                "workspace_id": state["workspace_id"],
                "scope_version": state["scope_version"],
            }
            if tool_name == "document.get_outline":
                arguments = {"document_id": str((state.get("selected_documents") or [{}])[0].get("document_id") or "")}
            elif tool_name == "literature.search":
                arguments = {"query": str(state.get("translated_query") or state["original_query"]), "limit": 10}
            action = {
                "type": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
            }
        usage = action.pop("_usage", None)
        if action.get("type") == "tool" and action.get("tool_name") != "knowledge.retrieve" and action.get("tool_name") not in (self.tool_handlers or {}):
            action = {"type": "tool", "tool_name": "knowledge.retrieve", "arguments": {}}
        result: dict[str, Any] = {"action": action, "trace": self._record(state, "agent", action=action)}
        if isinstance(usage, Mapping):
            result["llm_usage"] = [*(state.get("llm_usage") or []), dict(usage)]
        return result

    @staticmethod
    def _route(state: ResearchAgentState) -> str:
        action = state.get("action") or {}
        return str(action.get("type") or "final")

    def _tool(self, state: ResearchAgentState) -> dict[str, Any]:
        action = dict(state.get("action") or {})
        name = str(action.get("tool_name") or "knowledge.retrieve")
        arguments = dict(action.get("arguments") or {})
        if name == "knowledge.retrieve":
            observation = dict(self.research_tool(state))
        else:
            handler = (self.tool_handlers or {}).get(name)
            if handler is None:
                raise KeyError(f"LangGraph Tool 未注册：{name}")
            observation = dict(handler(arguments, state))
        calls = list(state.get("tool_calls") or [])
        calls.append({"tool_name": name, "arguments": arguments, "status": "succeeded"})
        updates: dict[str, Any] = {
            "observation": observation,
            "retrieved_evidence": (
                list(observation.get("selected_evidence") or observation.get("results") or [])
                if name == "knowledge.retrieve"
                else list(state.get("retrieved_evidence") or [])
            ),
            "final_answer": None,
            "result": None,
            "last_tool": name,
            "iteration": int(state.get("iteration") or 0) + 1,
            "tool_calls": calls,
            "trace": self._record(state, "tool", tool_name=name, status="succeeded"),
        }
        if name == "literature.resolve_publication_date":
            updates["publication_metadata"] = [
                *(state.get("publication_metadata") or []), observation
            ]
            if state.get("task_type") == "timeline":
                updates["timeline"] = sort_timeline(updates["publication_metadata"])
        if name == "knowledge.retrieve":
            values = list(observation.get("selected_evidence") or observation.get("results") or [])
            updates["evidence"] = [dict(value) for value in values if isinstance(value, Mapping)]
            updates["need_more_search"] = not bool(values)
            updates["evidence_by_paper"] = group_evidence_by_paper(
                [value for value in values if isinstance(value, Mapping)]
            )
            hierarchical = observation.get("candidate_papers")
            updates["paper_candidates"] = [
                dict(value) for value in hierarchical if isinstance(value, Mapping)
            ] if isinstance(hierarchical, list) else []
            updates["candidate_paper_ids"] = [
                str(value) for value in observation.get("candidate_paper_ids", [])
                if isinstance(value, str) and value
            ]
            updates["hierarchical_retrieval"] = dict(observation)
            updates["coverage_report"] = dict(observation.get("coverage") or {})
            updates["provisional_evidence"] = [
                dict(value) for value in observation.get("provisional_evidence", [])
                if isinstance(value, Mapping)
            ]
            updates["fulltext_jobs"] = [
                dict(value) for value in observation.get("fulltext_jobs", [])
                if isinstance(value, Mapping)
            ]
        return updates

    def _clarify(self, state: ResearchAgentState) -> dict[str, Any]:
        response = interrupt({
            "type": "clarification",
            "question": state.get("pending_question") or "请补充论文范围。",
            "thread_id": state.get("workspace_id"),
        })
        text = str(response.get("text") if isinstance(response, Mapping) else response).strip()
        return {
            "clarification_response": text,
            "pending_question": None,
            "messages": [*(state.get("messages") or []), {"role": "user", "content": text}],
            "state_changes": [*(state.get("state_changes") or []), {"node": "clarification", "response": text}],
            "trace": self._record(state, "clarification", status="resumed"),
        }

    @staticmethod
    def _final(state: ResearchAgentState) -> dict[str, Any]:
        observation = state.get("observation")
        answer = observation if isinstance(observation, Mapping) else state.get("final_answer")
        return {
            "result": dict(answer) if isinstance(answer, Mapping) else None,
            "final_answer": str(answer.get("answer") or "") if isinstance(answer, Mapping) else None,
            "answer": str(answer.get("answer") or "") if isinstance(answer, Mapping) else None,
            "citations": list(answer.get("citations") or []) if isinstance(answer, Mapping) else [],
            "trace": LangGraphResearchRuntime._static_trace(state, "final", status="succeeded"),
        }

    def _compile(self) -> Any:
        builder = StateGraph(ResearchAgentState)
        builder.add_node("translation_skill", self._translation)
        builder.add_node("reference_resolver", self._reference)
        builder.add_node("classify_task", self._classify_task)
        builder.add_node("research_planner", self._plan_research)
        builder.add_node("scope_read", self._scope)
        builder.add_node("resolve_papers", self._resolve_papers)
        builder.add_node("fetch_metadata", self._fetch_metadata)
        builder.add_node("agent", self._agent)
        builder.add_node("tool", self._tool)
        builder.add_node("clarify", self._clarify)
        builder.add_node("final", self._final)
        builder.add_edge(START, "translation_skill")
        builder.add_edge("translation_skill", "reference_resolver")
        builder.add_edge("reference_resolver", "classify_task")
        builder.add_edge("classify_task", "research_planner")
        builder.add_edge("research_planner", "scope_read")
        builder.add_edge("scope_read", "resolve_papers")
        builder.add_edge("resolve_papers", "fetch_metadata")
        builder.add_edge("fetch_metadata", "agent")
        builder.add_conditional_edges("agent", self._route, {"tool": "tool", "clarify": "clarify", "final": "final"})
        builder.add_edge("tool", "agent")
        builder.add_edge("clarify", "agent")
        builder.add_edge("final", END)
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, state: ResearchAgentState, *, thread_id: str, progress_callback: Callable[..., Any] | None = None) -> ResearchAgentState:
        self._callback = progress_callback
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = dict(self.graph.invoke(state, config=config))
            return self._mark_interrupt(result, config)
        finally:
            self._callback = None

    def resume(self, thread_id: str, user_input: str, *, progress_callback: Callable[..., Any] | None = None) -> ResearchAgentState:
        if not user_input.strip():
            raise ValueError("resume 输入不能为空。")
        self._callback = progress_callback
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = dict(self.graph.invoke(
                Command(resume={"text": user_input.strip()}),
                config=config,
            ))
            return self._mark_interrupt(result, config)
        finally:
            self._callback = None

    def _mark_interrupt(self, result: dict[str, Any], config: Mapping[str, Any]) -> ResearchAgentState:
        snapshot = self.graph.get_state(config)
        interrupts = [
            item.value
            for task in snapshot.tasks
            for item in getattr(task, "interrupts", ())
        ]
        if interrupts:
            result["__interrupt__"] = interrupts
        return result  # type: ignore[return-value]


def is_interrupted(state: Mapping[str, Any]) -> bool:
    """Whether a graph invocation stopped at an interrupt boundary."""

    return bool(state.get("__interrupt__")) or bool(state.get("pending_question")) and not state.get("final_answer")
