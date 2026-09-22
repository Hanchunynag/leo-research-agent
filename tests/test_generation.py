from __future__ import annotations

from typing import Any

import pytest
import httpx

from app.context.assembly import assemble_context_bundle
from app.context.models import ContextBundle
from app.generation.models import AnswerClaim, AnswerDraft
from app.generation.openai_compatible import (
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleConfig,
)
from app.generation.refusal import EMPTY_CONTEXT_REFUSAL, INVALID_DRAFT_REFUSAL
from app.generation.service import GroundedAnswerService


def candidate() -> dict[str, Any]:
    return {
        "rank": 1,
        "score": 0.9,
        "retrieval_source": "hybrid_rrf",
        "chunk_id": "C_alpha",
        "work_id": "W_alpha",
        "document_id": "D_alpha",
        "paper_id": "P_alpha",
        "title": "Alpha Tracking Paper",
        "authors": ["Ada Researcher"],
        "year": 2026,
        "doi": None,
        "section_path": ["METHOD", "Tracking"],
        "page_start": 2,
        "page_end": 3,
        "block_ids": ["B_alpha"],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": "Alpha observations estimate ephemeris and clock errors.",
    }


def context_bundle() -> ContextBundle:
    return assemble_context_bundle(
        "Which observations estimate the errors?",
        "fast",
        [candidate()],
        token_budget=1000,
    )


class FakeAnswerProvider:
    model_name = "fixture/answer"

    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls = 0
        self.queries: list[str] = []

    def generate(self, query: str, context: ContextBundle) -> AnswerDraft:
        self.calls += 1
        self.queries.append(query)
        return self.draft


class UnusedRuntime:
    def build_context(self, query: str, **kwargs: Any) -> ContextBundle:
        return context_bundle()


def service_for(provider: FakeAnswerProvider) -> GroundedAnswerService:
    return GroundedAnswerService(UnusedRuntime(), provider)  # type: ignore[arg-type]


def test_valid_claims_are_rendered_with_traceable_citations() -> None:
    provider = FakeAnswerProvider(
        AnswerDraft(
            True,
            [AnswerClaim("C1", "The observations estimate both errors.", ["S1"])],
            provider_metadata={"response_model": "fixture-served-model"},
        )
    )

    result = service_for(provider).answer_from_context(context_bundle())

    assert result.answerable is True
    assert result.answer == "The observations estimate both errors. [S1]"
    assert result.validation.valid is True
    assert result.diagnostics["response_model"] == "fixture-served-model"
    assert result.citations[0].document_id == "D_alpha"
    assert result.citations[0].page_start == 2
    assert result.citations[0].block_ids == ["B_alpha"]


@pytest.mark.parametrize(
    ("claim", "expected_code"),
    [
        (AnswerClaim("C1", "Unsupported source.", ["S9"]), "unknown_source_id"),
        (AnswerClaim("C1", "No citation.", []), "claim_without_citation"),
    ],
)
def test_invalid_claim_citations_fail_closed(
    claim: AnswerClaim,
    expected_code: str,
) -> None:
    provider = FakeAnswerProvider(AnswerDraft(True, [claim]))

    result = service_for(provider).answer_from_context(context_bundle())

    assert result.answerable is False
    assert result.answer == ""
    assert result.claims == []
    assert result.citations == []
    assert result.refusal_reason == INVALID_DRAFT_REFUSAL
    assert expected_code in {issue.code for issue in result.validation.issues}


def test_provider_can_return_a_valid_explicit_refusal() -> None:
    provider = FakeAnswerProvider(
        AnswerDraft(False, [], "The evidence does not identify the sensor.")
    )

    result = service_for(provider).answer_from_context(context_bundle())

    assert result.answerable is False
    assert result.refusal_reason == "The evidence does not identify the sensor."
    assert result.validation.valid is True


def test_empty_context_is_rejected_before_provider_call() -> None:
    empty = ContextBundle(
        query="unanswerable",
        retrieval_mode="fast",
        evidence=[],
        context_text="",
        token_budget=100,
        token_count=0,
        diagnostics={},
    )
    provider = FakeAnswerProvider(
        AnswerDraft(True, [AnswerClaim("C1", "Should not run.", ["S1"])])
    )

    result = service_for(provider).answer_from_context(empty)

    assert result.answerable is False
    assert result.refusal_reason == EMPTY_CONTEXT_REFUSAL
    assert result.diagnostics["generation_skipped"] is True
    assert provider.calls == 0


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHTTPClient:
    def __init__(self, response: FakeHTTPResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, *, json: dict[str, Any]) -> FakeHTTPResponse:
        self.calls.append((url, json))
        return self.response


class RetryHTTPClient(FakeHTTPClient):
    def __init__(self, response: FakeHTTPResponse) -> None:
        super().__init__(response)
        self.failures = 2

    def post(self, url: str, *, json: dict[str, Any]) -> FakeHTTPResponse:
        self.calls.append((url, json))
        if self.failures:
            self.failures -= 1
            raise httpx.ConnectError("temporary transport failure", request=httpx.Request("POST", url))
        return self.response


def test_openai_compatible_provider_supports_raw_gateway_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(
        {
            "choices": [{"message": {"content": "{\"ok\":true}"}}],
        }
    )
    captured: dict[str, Any] = {}

    class CapturingHTTPClient(FakeHTTPClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(response)
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.generation.openai_compatible.httpx.Client",
        CapturingHTTPClient,
    )
    provider = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig(
            "https://gateway.example/v1",
            "qwen3.5:9b",
            api_key="test-gateway-token",
            auth_scheme="raw",
        )
    )

    provider.chat_completion([{"role": "user", "content": "ping"}])

    assert captured["headers"] == {"Authorization": "test-gateway-token"}


def test_openai_compatible_provider_retries_transport_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(
        {
            "choices": [{"message": {"content": '{"answerable":false,"claims":[],"refusal_reason":"retry"}'}}]
        }
    )
    client = RetryHTTPClient(response)
    monkeypatch.setattr("app.generation.openai_compatible.time.sleep", lambda _: None)
    provider = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig("http://127.0.0.1:11434", "local-model"),
        client=client,
    )

    draft = provider.generate("question", context_bundle())

    assert draft.answerable is False
    assert len(client.calls) == 3


def test_openai_compatible_provider_parses_structured_json_without_network() -> None:
    response = FakeHTTPResponse(
        {
            "model": "served-local-model",
            "usage": {
                "prompt_tokens": 321,
                "prompt_cache_hit_tokens": 256,
                "prompt_cache_miss_tokens": 64,
                "completion_tokens": 42,
                "total_tokens": 363,
                "ignored_nested_detail": {"cached_tokens": 10},
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"answerable":true,"claims":[{"claim_id":"C1",'
                            '"text":"Supported fact.","source_ids":["S1"]}],'
                            '"refusal_reason":null}\n```'
                        )
                    }
                }
            ]
        }
    )
    client = FakeHTTPClient(response)
    provider = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig(
            "http://127.0.0.1:11434",
            "local-model",
            prompt_layout="context_first",
        ),
        client=client,
    )

    draft = provider.generate("question", context_bundle())

    assert draft.claims[0].source_ids == ["S1"]
    assert draft.provider_metadata["response_model"] == "served-local-model"
    assert draft.provider_metadata["usage"] == {
        "prompt_tokens": 321,
        "prompt_cache_hit_tokens": 256,
        "prompt_cache_miss_tokens": 64,
        "completion_tokens": 42,
        "total_tokens": 363,
    }
    assert draft.provider_metadata["cache_diagnostics"] == {
        "hit_tokens": 256,
        "miss_tokens": 64,
        "eligible_prompt_tokens": 320,
        "hit_rate": 0.8,
    }
    prompt_diagnostics = draft.provider_metadata["prompt_diagnostics"]
    assert prompt_diagnostics["layout"] == "context_first"
    assert len(prompt_diagnostics["fingerprint"]) == 64
    second_draft = provider.generate("different question", context_bundle())
    second_prompt = second_draft.provider_metadata["prompt_diagnostics"]
    assert (
        second_prompt["stable_prefix_fingerprint"]
        == prompt_diagnostics["stable_prefix_fingerprint"]
    )
    assert second_prompt["fingerprint"] != prompt_diagnostics["fingerprint"]
    assert response.status_checked is True
    assert client.calls[0][0] == "http://127.0.0.1:11434/v1/chat/completions"
    request = client.calls[0][1]
    assert request["model"] == "local-model"
    assert request["response_format"] == {"type": "json_object"}
    assert "[S1]" in request["messages"][1]["content"]
    user_content = request["messages"][1]["content"]
    assert user_content.index("Evidence bundle:") < user_content.index("Question:")


def test_chat_completion_can_disable_gateway_hidden_reasoning() -> None:
    response = FakeHTTPResponse(
        {"choices": [{"message": {"content": "{}"}}]}
    )
    client = FakeHTTPClient(response)
    provider = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig("http://127.0.0.1:11434", "local-model"),
        client=client,
    )

    provider.chat_completion(
        [{"role": "user", "content": "Return JSON."}],
        max_tokens=128,
        reasoning_effort="none",
    )

    assert client.calls[0][1]["reasoning_effort"] == "none"
