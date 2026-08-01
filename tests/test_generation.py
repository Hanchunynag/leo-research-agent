from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import main as cli
from app.context.assembly import assemble_context_bundle
from app.context.models import ContextBundle
from app.context.session import ContextSessionStore
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
    assert "[S1]" in request["messages"][1]["content"]
    user_content = request["messages"][1]["content"]
    assert user_content.index("Evidence bundle:") < user_content.index("Question:")


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
            "--context-session",
            "cache_experiment",
        ]
    )

    provider = cli.answer_provider_from_args(args)

    assert provider.endpoint == "https://environment.example/v1/chat/completions"
    assert provider.config.model == "cli-model"
    assert provider.config.api_key == "file-test-key"
    assert provider.config.timeout_seconds == 45
    assert provider.config.max_tokens == 700
    assert provider.config.prompt_layout == "context_first"


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


def test_context_session_round_trip_and_integrity_check(tmp_path: Path) -> None:
    store = ContextSessionStore(tmp_path)
    original = context_bundle()

    saved = store.save("leo_timing", original)
    loaded = store.load("leo_timing")

    assert loaded.context_hash == saved.context_hash
    assert loaded.context.context_text == original.context_text
    assert loaded.context.evidence[0].chunk_id == "C_alpha"

    path = store.path_for("leo_timing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["context"]["context_text"] += " tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="完整性指纹不匹配"):
        store.load("leo_timing")


def test_context_session_rejects_unsafe_identifier(tmp_path: Path) -> None:
    store = ContextSessionStore(tmp_path)

    with pytest.raises(ValueError, match="session ID"):
        store.path_for("../outside")


def test_context_session_loads_pre_agentic_evidence_schema(tmp_path: Path) -> None:
    store = ContextSessionStore(tmp_path)
    store.save("legacy", context_bundle())
    path = store.path_for("legacy")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload["context"]["evidence"][0]
    evidence.pop("evidence_id")
    evidence.pop("origin")
    serialized = json.dumps(
        payload["context"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["context_hash"] = hashlib.sha256(serialized.encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load("legacy")

    assert loaded.context.evidence[0].evidence_id is None
    assert loaded.context.evidence[0].origin == "newly_retrieved"


class CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def build_context(self, query: str, **kwargs: Any) -> ContextBundle:
        self.calls += 1
        return replace(context_bundle(), query=query)


def test_answer_cli_reuses_context_session_without_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = CountingRuntime()
    provider = FakeAnswerProvider(
        AnswerDraft(True, [AnswerClaim("C1", "Session fact.", ["S1"])])
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "retrieval_runtime_from_args",
        lambda args, include_reranker: runtime,
    )
    monkeypatch.setattr(cli, "answer_provider_from_args", lambda args: provider)

    cli.main(["answer", "initial question", "--context-session", "leo_session"])
    created = json.loads(capsys.readouterr().out)
    assert created["diagnostics"]["context_session"]["state"] == "created"
    assert runtime.calls == 1

    monkeypatch.setattr(
        cli,
        "retrieval_runtime_from_args",
        lambda args, include_reranker: pytest.fail("session reuse must skip retrieval"),
    )
    cli.main(["answer", "follow-up question", "--context-session", "leo_session"])
    reused = json.loads(capsys.readouterr().out)

    session_diagnostics = reused["diagnostics"]["context_session"]
    assert reused["query"] == "follow-up question"
    assert session_diagnostics["state"] == "reused"
    assert session_diagnostics["retrieval_skipped"] is True
    assert session_diagnostics["source_query"] == "initial question"
    assert provider.queries == ["initial question", "follow-up question"]
