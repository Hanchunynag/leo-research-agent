from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import main as cli
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

    def generate(self, query: str, context: ContextBundle) -> AnswerDraft:
        self.calls += 1
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


def test_openai_compatible_provider_parses_structured_json_without_network() -> None:
    response = FakeHTTPResponse(
        {
            "model": "served-local-model",
            "usage": {
                "prompt_tokens": 321,
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
        OpenAICompatibleConfig("http://127.0.0.1:11434", "local-model"),
        client=client,
    )

    draft = provider.generate("question", context_bundle())

    assert draft.claims[0].source_ids == ["S1"]
    assert draft.provider_metadata == {
        "response_model": "served-local-model",
        "usage": {
            "prompt_tokens": 321,
            "completion_tokens": 42,
            "total_tokens": 363,
        },
    }
    assert response.status_checked is True
    assert client.calls[0][0] == "http://127.0.0.1:11434/v1/chat/completions"
    request = client.calls[0][1]
    assert request["model"] == "local-model"
    assert "[S1]" in request["messages"][1]["content"]


def test_answer_cli_outputs_grounded_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    provider = FakeAnswerProvider(
        AnswerDraft(True, [AnswerClaim("C1", "CLI supported fact.", ["S1"])])
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "retrieval_runtime_from_args",
        lambda args, include_reranker: UnusedRuntime(),
    )
    monkeypatch.setattr(cli, "answer_provider_from_args", lambda args: provider)

    cli.main(
        [
            "answer",
            "Which observations?",
            "--llm-base-url",
            "http://127.0.0.1:11434",
            "--llm-model",
            "local-model",
            "--local-files-only",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "1.0"
    assert output["answerable"] is True
    assert output["answer"] == "CLI supported fact. [S1]"
    assert output["citations"][0]["source_id"] == "S1"
    assert "context" not in output
    assert "citations" not in output["validation"]


def test_answer_cli_can_include_full_context_for_debugging(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = FakeAnswerProvider(
        AnswerDraft(True, [AnswerClaim("C1", "Debug fact.", ["S1"])])
    )
    monkeypatch.setattr(
        cli,
        "retrieval_runtime_from_args",
        lambda args, include_reranker: UnusedRuntime(),
    )
    monkeypatch.setattr(cli, "answer_provider_from_args", lambda args: provider)

    cli.main(
        [
            "answer",
            "Debug question",
            "--llm-base-url",
            "http://127.0.0.1:11434",
            "--llm-model",
            "local-model",
            "--include-context",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["context"]["evidence_count"] == 1
    assert output["validation"]["citations"][0]["source_id"] == "S1"


def test_answer_provider_reads_local_dotenv_with_safe_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "LEO_LLM_BASE_URL=https://file.example/v1\n"
        "LEO_LLM_MODEL=file-model\n"
        "LEO_LLM_API_KEY=file-test-key\n"
        "LEO_LLM_TIMEOUT_SECONDS=45\n"
        "LEO_LLM_MAX_TOKENS=700\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("LEO_LLM_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("LEO_LLM_MODEL", "environment-model")
    args = cli.build_parser().parse_args(
        [
            "answer",
            "question",
            "--llm-model",
            "cli-model",
        ]
    )

    provider = cli.answer_provider_from_args(args)

    assert provider.endpoint == "https://environment.example/v1/chat/completions"
    assert provider.config.model == "cli-model"
    assert provider.config.api_key == "file-test-key"
    assert provider.config.timeout_seconds == 45
    assert provider.config.max_tokens == 700


def test_answer_provider_reports_missing_local_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    for name in ("LEO_LLM_BASE_URL", "LEO_LLM_MODEL", "LEO_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    args = cli.build_parser().parse_args(["answer", "question"])

    with pytest.raises(ValueError, match="复制 .env.example 为 .env"):
        cli.answer_provider_from_args(args)
