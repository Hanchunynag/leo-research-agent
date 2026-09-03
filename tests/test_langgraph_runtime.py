from __future__ import annotations

import json
from typing import Any, Mapping

from app.langchain_agent.graph import LLMActionDecider, LangGraphResearchRuntime, resolve_references
from app.langchain_agent.planning import build_research_plan, classify_research_task


class _Translation:
    def invoke(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "original_query": value["query"],
            "zh_query": value["query"],
            "en_query": "compare the first four papers",
            "retrieval_queries": [value["query"], "compare the first four papers"],
            "output_language": "zh",
        }


def test_reference_resolver_maps_ordinal_papers() -> None:
    papers = [{"document_id": f"D{i}", "title": f"Paper {i}"} for i in range(1, 6)]
    selected, mapping, question = resolve_references("比较前四篇论文", papers)

    assert question is None
    assert [item["document_id"] for item in selected] == ["D1", "D2", "D3", "D4"]
    assert mapping["1"]["title"] == "Paper 1"


def test_reference_resolver_accepts_chinese_two() -> None:
    papers = [{"document_id": f"D{i}", "title": f"Paper {i}"} for i in range(1, 4)]

    selected, _, question = resolve_references("比较前两篇论文", papers)

    assert question is None
    assert [item["document_id"] for item in selected] == ["D1", "D2"]


def test_langgraph_interrupt_and_resume() -> None:
    observations: list[dict[str, Any]] = []

    def retrieve(state: Mapping[str, Any]) -> Mapping[str, Any]:
        observations.append(dict(state))
        return {"answerable": True, "answer": "已完成比较", "selected_evidence": []}

    runtime = LangGraphResearchRuntime(_Translation(), retrieve)
    initial = runtime.invoke(
        {
            "messages": [],
            "original_query": "比较这四篇论文",
            "workspace_id": "default",
            "scope_version": 1,
            "last_search_results": [],
            "iteration": 0,
            "max_iterations": 4,
        },
        thread_id="thread-clarify",
    )

    assert initial["pending_question"]
    assert initial["__interrupt__"]
    assert observations == []

    resumed = runtime.resume("thread-clarify", "Paper A、Paper B、Paper C、Paper D")

    assert resumed["final_answer"] == "已完成比较"
    assert resumed["result"]["answerable"] is True
    assert resumed["tool_calls"][0]["tool_name"] == "knowledge.retrieve"
    assert any(item["node"] == "clarification" for item in resumed["trace"])


def test_graph_records_llm_action_usage() -> None:
    class Provider:
        def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": json.dumps({"type": "tool", "tool_name": "knowledge.retrieve"})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

    runtime = LangGraphResearchRuntime(
        _Translation(),
        lambda _state: {"answerable": True, "answer": "ok"},
        action_decider=LLMActionDecider(Provider()),
    )
    result = runtime.invoke(
        {"original_query": "What is measured?", "workspace_id": "default", "scope_version": 1, "max_iterations": 1},
        thread_id="thread-usage",
    )

    assert result["llm_usage"] == [{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}]


def test_timeline_skill_resolves_titles_before_rag() -> None:
    calls: list[tuple[str, str]] = []

    def resolve_title(arguments: Mapping[str, Any], _state: Mapping[str, Any]) -> Mapping[str, Any]:
        title = str(arguments["title"])
        calls.append(("metadata", title))
        return {
            "query_title": title,
            "matched_title": title,
            "publication_year": 2021 if title == "Paper A" else 2024,
            "status": "resolved",
        }

    def retrieve(_state: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(("rag", ""))
        return {"answerable": True, "answer": "timeline answer"}

    runtime = LangGraphResearchRuntime(
        _Translation(),
        retrieve,
        tool_handlers={"literature.resolve_publication_date": resolve_title},
    )
    result = runtime.invoke(
        {
            "original_query": "这些论文的研究路线如何演进？",
            "workspace_id": "default",
            "scope_version": 1,
            "last_search_results": [
                {"document_id": "D1", "title": "Paper A"},
                {"document_id": "D2", "title": "Paper B"},
            ],
            "max_iterations": 5,
        },
        thread_id="thread-timeline",
    )

    assert calls == [("metadata", "Paper A"), ("metadata", "Paper B"), ("rag", "")]
    assert [item["publication_year"] for item in result["publication_metadata"]] == [2021, 2024]
    assert result["result"]["answer"] == "timeline answer"


def test_research_planner_separates_timeline_and_compare() -> None:
    assert classify_research_task("总结这个领域的研究发展路线") == "timeline"
    assert classify_research_task("比较 Paper A 和 Paper B") == "compare"
    assert build_research_plan("timeline")["order_by"] == "publication_year_ascending"
    assert build_research_plan("compare")["comparison_dimensions"] == [
        "method",
        "data_and_experiment",
        "contribution",
        "limitation",
    ]


def test_timeline_planner_lists_scope_resolves_metadata_and_sorts_before_rag() -> None:
    calls: list[tuple[str, str]] = []

    def list_documents(
        arguments: Mapping[str, Any], _state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        calls.append(("papers", str(arguments["workspace_id"])))
        return {
            "documents": [
                {"document_id": "D_late", "paper_id": "P_late", "title": "Late Paper"},
                {"document_id": "D_early", "paper_id": "P_early", "title": "Early Paper"},
            ]
        }

    def metadata(arguments: Mapping[str, Any], _state: Mapping[str, Any]) -> Mapping[str, Any]:
        title = str(arguments["title"])
        calls.append(("metadata", title))
        return {
            "query_title": title,
            "matched_title": title,
            "publication_year": 2023 if title == "Late Paper" else 2018,
            "status": "resolved",
        }

    def retrieve(_state: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(("rag", ""))
        return {
            "answerable": True,
            "answer": "timeline answer",
            "selected_evidence": [
                {"evidence_id": "E1", "document_id": "D_early"},
                {"evidence_id": "E2", "document_id": "D_late"},
            ],
        }

    runtime = LangGraphResearchRuntime(
        _Translation(),
        retrieve,
        tool_handlers={
            "workspace.list_documents": list_documents,
            "literature.resolve_publication_date": metadata,
        },
    )
    result = runtime.invoke(
        {
            "original_query": "总结该领域的发展路线",
            "workspace_id": "default",
            "scope_version": 1,
            "max_iterations": 4,
        },
        thread_id="thread-timeline-plan",
    )

    assert calls == [
        ("papers", "default"),
        ("metadata", "Late Paper"),
        ("metadata", "Early Paper"),
        ("rag", ""),
    ]
    assert result["task_type"] == "timeline"
    assert [item["document_id"] for item in result["timeline"]] == ["D_early", "D_late"]
    assert result["evidence_by_paper"] == {"D_early": ["E1"], "D_late": ["E2"]}


def test_compare_planner_keeps_metadata_and_evidence_grouped_by_paper() -> None:
    def metadata(arguments: Mapping[str, Any], _state: Mapping[str, Any]) -> Mapping[str, Any]:
        title = str(arguments["title"])
        return {"query_title": title, "matched_title": title, "publication_year": 2020}

    runtime = LangGraphResearchRuntime(
        _Translation(),
        lambda _state: {
            "answerable": True,
            "answer": "comparison answer",
            "selected_evidence": [
                {"evidence_id": "E_A", "document_id": "D_A"},
                {"evidence_id": "E_B", "document_id": "D_B"},
            ],
        },
        tool_handlers={"literature.resolve_publication_date": metadata},
    )
    result = runtime.invoke(
        {
            "original_query": "比较前两篇论文",
            "workspace_id": "default",
            "scope_version": 1,
            "last_search_results": [
                {"document_id": "D_A", "title": "Paper A"},
                {"document_id": "D_B", "title": "Paper B"},
            ],
            "max_iterations": 4,
        },
        thread_id="thread-compare-plan",
    )

    assert result["task_type"] == "compare"
    assert result["research_plan"]["output_structure"] == "comparison_matrix"
    assert len(result["publication_metadata"]) == 2
    assert result["evidence_by_paper"] == {"D_A": ["E_A"], "D_B": ["E_B"]}


def test_structured_task_skips_action_decider_before_and_after_retrieval() -> None:
    calls: list[Mapping[str, Any]] = []

    def decider(state: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(dict(state))
        return {"type": "final"}

    runtime = LangGraphResearchRuntime(
        _Translation(),
        lambda _state: {"answerable": True, "answer": "timeline answer"},
        action_decider=decider,
    )
    result = runtime.invoke(
        {
            "original_query": "总结该领域的发展路线",
            "workspace_id": "default",
            "scope_version": 1,
            "max_iterations": 4,
        },
        thread_id="thread-structured-no-controller",
    )

    assert calls == []
    assert result["result"]["answer"] == "timeline answer"
