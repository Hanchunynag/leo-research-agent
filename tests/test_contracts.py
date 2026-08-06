from __future__ import annotations

from typing import Any

import pytest

from app.contracts import (
    EvidenceRequest,
    KnowledgeEngine,
    LegacyAgentRuntimeAdapter,
    LegacyEvidenceMapper,
    LegacyKnowledgeEngineAdapter,
    ResearchHarness,
)


def request(**overrides: Any) -> EvidenceRequest:
    values: dict[str, Any] = {
        "request_id": "REQ-001",
        "workspace_id": "WS-LEO",
        "scope_version": 3,
        "query": "Which inertial measurements aid the state prediction?",
        "top_k": 2,
    }
    values.update(overrides)
    return EvidenceRequest(**values)


class FakeLegacyRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((query, kwargs))
        return {
            "retriever": "hybrid_rrf",
            "results": [
                {
                    "rank": 1,
                    "score": 0.25,
                    "retrieval_source": "hybrid_rrf",
                    "work_id": "W-1",
                    "document_id": "D-1",
                    "chunk_id": "C-1",
                    "section_id": "S-1",
                    "page_start": 4,
                    "page_end": 4,
                    "block_ids": ["B-1"],
                    "content": "Doppler and inertial measurements aid prediction.",
                },
                {
                    "rank": 2,
                    "score": 0.20,
                    "retrieval_source": "dense",
                    "work_id": "W-2",
                    "document_id": "D-2",
                    "chunk_id": "C-2",
                    "page_start": 8,
                    "page_end": 9,
                    "block_ids": ["B-2"],
                    "content": "A second candidate.",
                },
            ],
        }


def test_workspace_and_scope_are_required_by_evidence_request() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        request(workspace_id="")
    with pytest.raises(ValueError, match="scope_version"):
        request(scope_version=0)


def test_legacy_knowledge_adapter_maps_and_filters_without_mutating_runtime() -> None:
    runtime = FakeLegacyRuntime()
    adapter = LegacyKnowledgeEngineAdapter(runtime)

    candidates = adapter.retrieve(request(document_ids=("D-1",)))

    assert isinstance(adapter, KnowledgeEngine)
    assert [item.chunk_id for item in candidates] == ["C-1"]
    assert candidates[0].workspace_id == "WS-LEO"
    assert candidates[0].scope_version == 3
    assert runtime.calls[0][1]["document_id"] == "D-1"
    assert adapter.last_response is not None
    assert adapter.last_response["retriever"] == "hybrid_rrf"


def test_legacy_evidence_mapper_rejects_untraceable_candidate() -> None:
    mapper = LegacyEvidenceMapper()
    req = request()
    valid = mapper.candidate_from_mapping(
        req,
        {
            "chunk_id": "C-1",
            "work_id": "W-1",
            "document_id": "D-1",
            "page_start": 2,
            "page_end": 2,
            "block_ids": ["B-1"],
            "content": "Traceable evidence.",
        },
        fallback_rank=1,
    )
    invalid = mapper.candidate_from_mapping(
        req,
        {"chunk_id": "C-2", "content": "No canonical locator."},
        fallback_rank=2,
    )

    bundle = mapper.bundle(req, [valid, invalid])

    assert [item.chunk_id for item in bundle.evidence] == ["C-1"]
    assert bundle.rejected_candidate_ids == ("C-2",)
    assert bundle.evidence[0].verification_method == "legacy_locator_fields"


class FakeLegacyAgentService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return {
            "answerable": True,
            "answer": "The measurements are Doppler and inertial readings. [S1]",
            "diagnostics": {
                "metrics": {"latency_ms": 12.5},
                "prompt_cache": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        }


def test_legacy_agent_adapter_preserves_answer_and_exposes_harness_contract() -> None:
    service = FakeLegacyAgentService()
    adapter = LegacyAgentRuntimeAdapter(service)

    legacy = adapter.answer("legacy query", session_id="session-1")
    run = adapter.run(request(), state={"session_id": "session-2"})

    assert legacy["answerable"] is True
    assert service.calls[0] == {
        "query": "legacy query",
        "session_id": "session-1",
        "force_new_topic": False,
        "include_context": False,
    }
    assert isinstance(adapter, ResearchHarness)
    assert run.state == "completed"
    assert run.answerable is True
    assert run.elapsed_ms == 12.5
    assert run.llm_call_count is None
    assert run.token_usage == 120
    assert service.calls[1]["session_id"] == "session-2"
