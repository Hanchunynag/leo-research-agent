from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.agentic.models import AgenticAnswerDraft, AgenticClaim
from app.contracts import ResearchScope
from app.langchain_agent import build_langchain_agent_service
from app.langchain_agent.translation import BilingualQueryTranslator, build_translation_tool


@dataclass
class _Workspace:
    workspace_id: str = "default"
    scope_version: int = 1
    included_document_ids: tuple[str, ...] = ("D_1",)
    excluded_document_ids: tuple[str, ...] = ()


class _Workspaces:
    def ensure_default(self) -> _Workspace:
        return _Workspace()

    def require_scope(self, workspace_id: str, scope_version: int) -> ResearchScope:
        assert (workspace_id, scope_version) == ("default", 1)
        return ResearchScope(
            workspace_id="default",
            scope_version=1,
            included_document_ids=("D_1",),
        )


class _ChatProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def chat_completion(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.fail:
            raise RuntimeError("translation backend unavailable")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "original_query": "ignored by translator",
                                "zh_query": "哪些观测量用于估计低轨卫星星历和时钟误差？",
                                "en_query": "Which measurements estimate low-Earth-orbit satellite ephemeris and clock errors?",
                            }
                        )
                    }
                }
            ]
        }


class _ReasoningProvider:
    model_name = "fixture-llm"

    def __init__(self, chat: _ChatProvider) -> None:
        self.provider = chat
        self.answer_payloads: list[dict[str, Any]] = []

    def generate_answer(
        self, messages: list[dict[str, str]]
    ) -> tuple[AgenticAnswerDraft, dict[str, Any]]:
        payload = json.loads(messages[-1]["content"])
        self.answer_payloads.append(payload)
        first = payload["selected_evidence"][0]
        return (
            AgenticAnswerDraft(
                answerable=True,
                claims=[
                    AgenticClaim(
                        claim_id="C1",
                        text="Pseudorange measurements estimate LEO satellite ephemeris and clock errors.",
                        category="observation_type",
                        source_ids=[first["source_id"]],
                        evidence_ids=[first["evidence_id"]],
                    )
                ],
            ),
            {"usage": {"total_tokens": 20}},
        )


class _Knowledge:
    workspace_id = "default"
    last_diagnostics: dict[str, Any] = {
        "evidence_intelligence": {"conflicts": [["E_1", "E_2"]]}
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve_multi(self, queries: tuple[str, ...], **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append({"queries": tuple(queries), **kwargs})
        return {
            "results": [
                {
                    "evidence_id": "E_1",
                    "evidence_state": "selected",
                    "document_id": "D_1",
                    "chunk_id": "D_1_C_1",
                    "page_start": 1,
                    "page_end": 1,
                    "block_ids": ["B_1"],
                    "content": "Pseudorange measurements estimate LEO satellite ephemeris and clock errors.",
                    "evidence_grade": "primary",
                    "directness": "direct",
                },
                {
                    "evidence_id": "E_1",
                    "evidence_state": "selected",
                    "document_id": "D_1",
                    "chunk_id": "D_1_C_1",
                    "content": "duplicate",
                },
                {
                    "evidence_id": "E_2",
                    "evidence_state": "verified",
                    "document_id": "D_1",
                    "chunk_id": "D_1_C_2",
                    "content": "must not enter generation",
                },
            ]
        }


def _service(tmp_path: Path, *, fail_translation: bool = False) -> tuple[Any, _ChatProvider, _Knowledge]:
    chat = _ChatProvider(fail=fail_translation)
    knowledge = _Knowledge()
    return (
        build_langchain_agent_service(
            tmp_path,
            knowledge,
            _Workspaces(),
            _ReasoningProvider(chat),
        ),
        chat,
        knowledge,
    )


def test_langchain_agent_translates_then_uses_merged_bilingual_retrieval(
    tmp_path: Path,
) -> None:
    service, chat, knowledge = _service(tmp_path)
    progress: list[str] = []

    result = service.answer(
        "哪些观测量用于估计低轨卫星星历和时钟误差？",
        progress_callback=lambda stage, _message, _progress, *_details: progress.append(stage),
    )

    assert len(chat.calls) == 1
    assert knowledge.calls == [
        {
            "queries": (
                "哪些观测量用于估计低轨卫星星历和时钟误差？",
                "Which measurements estimate low-Earth-orbit satellite ephemeris and clock errors?",
            ),
            "limit": 10,
            "workspace_id": "default",
            "scope_version": 1,
        }
    ]
    assert result["answerable"] is True
    assert [value["evidence_id"] for value in result["selected_evidence"]] == ["E_1"]
    assert result["conflicts"] == []
    assert result["diagnostics"]["retrieval_mode"] == "langchain_bilingual_rag"
    assert result["diagnostics"]["harness"]["usage"]["llm_calls"] == 2
    assert progress == [
        "translation",
        "translation_completed",
        "retrieval_start",
        "scope_ready",
        "retrieved",
        "generation_start",
        "generated",
    ]
    assert result["diagnostics"]["langchain"]["skills"] == [
        "TranslationSkill",
        "ScopeReadSkill",
        "BilingualRetrievalSkill",
        "AnswerGenerationSkill",
        "ClaimValidationSkill",
    ]
    assert result["workflow_details"]["bilingual_query"]["llm_execution"]["tool"] == "translate_query"
    bilingual = result["workflow_details"]["bilingual_query"]
    assert bilingual["translation_status"] == "translated"
    assert bilingual["en_query"].startswith("Which measurements")
    assert bilingual["llm_execution"]["tool"] == "translate_query"
    assert bilingual["llm_execution"]["model"] == "fixture-llm"


def test_translation_failure_falls_back_to_original_query_without_refusal(
    tmp_path: Path,
) -> None:
    service, chat, knowledge = _service(tmp_path, fail_translation=True)

    result = service.answer("哪些观测量用于估计低轨卫星星历和时钟误差？")

    assert len(chat.calls) == 1
    assert knowledge.calls[0]["queries"] == ("哪些观测量用于估计低轨卫星星历和时钟误差？",)
    assert result["answerable"] is True
    bilingual = result["workflow_details"]["bilingual_query"]
    assert {
        key: bilingual[key]
        for key in (
            "original_query",
            "zh_query",
            "en_query",
            "retrieval_queries",
            "translation_status",
            "translation_failure_kind",
        )
    } == {
        "original_query": "哪些观测量用于估计低轨卫星星历和时钟误差？",
        "zh_query": "哪些观测量用于估计低轨卫星星历和时钟误差？",
        "en_query": "哪些观测量用于估计低轨卫星星历和时钟误差？",
        "retrieval_queries": ["哪些观测量用于估计低轨卫星星历和时钟误差？"],
        "translation_status": "fallback",
        "translation_failure_kind": "RuntimeError",
    }
    assert bilingual["llm_execution"]["tool"] == "translate_query"


def test_translation_tool_uses_original_query_and_parses_json_response() -> None:
    chat = _ChatProvider()
    tool = build_translation_tool(BilingualQueryTranslator(_ReasoningProvider(chat)))

    output = tool.invoke({"query": "What measurements estimate ephemeris error?"})

    assert chat.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "What measurements estimate ephemeris error?",
    }
    assert output["original_query"] == "What measurements estimate ephemeris error?"
    assert output["zh_query"].startswith("哪些观测量")


def test_compare_plan_reaches_synthesis_without_changing_retrieval_query(
    tmp_path: Path,
) -> None:
    chat = _ChatProvider()
    reasoning = _ReasoningProvider(chat)
    knowledge = _Knowledge()
    service = build_langchain_agent_service(
        tmp_path,
        knowledge,
        _Workspaces(),
        reasoning,
    )

    service.answer("比较 Paper A 和 Paper B")

    assert knowledge.calls[0]["queries"][0] == "比较 Paper A 和 Paper B"
    constraints = reasoning.answer_payloads[0]["scope_constraints"]
    assert any("比较维度" in value for value in constraints)
    assert any("对应论文" in value for value in constraints)


def test_timeline_plan_adds_bounded_title_evidence_probes(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    captured: dict[str, Any] = {}

    def answer(_query: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"answerable": True, "answer": "ok"}

    service.delegate.answer = answer  # type: ignore[method-assign]
    service._answer_options = {}  # noqa: SLF001
    service._graph_callback = None  # noqa: SLF001
    service._research_tool(  # noqa: SLF001
        {
            "original_query": "总结发展路线",
            "output_language": "zh",
            "translation": {"retrieval_queries": ["总结发展路线", "summarize evolution"]},
            "research_plan": {"task_type": "timeline", "per_paper_evidence": True},
            "papers": [
                {"document_id": "D_1", "title": "Paper A"},
                {"document_id": "D_2", "title": "Paper B"},
            ],
            "publication_metadata": [],
        }
    )

    assert captured["retrieval_queries"] == (
        "总结发展路线",
        "summarize evolution",
        "Paper A method contribution limitation",
        "Paper B method contribution limitation",
    )
    assert captured["target_document_ids"] == ("D_1", "D_2")
