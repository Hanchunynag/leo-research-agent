from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

import main as cli
from app.agentic.config import AgenticRAGConfig
from app.agentic.coverage import (
    deterministic_coverage,
    deterministic_repair,
    deterministic_semantic_validation,
)
from app.agentic.evaluation import aggregate_agentic_metrics
from app.agentic.models import (
    AgenticAnswerDraft,
    AgenticClaim,
    AgenticMetricsRecord,
    CoverageReport,
    QueryPlan,
    RoutingLLMDecision,
    SemanticClaimResult,
    SemanticValidationReport,
)
from app.agentic.planning import QueryPlanner
from app.agentic.prompting import build_topic_messages, compact_topic
from app.agentic.provider import OpenAIAgenticReasoningProvider
from app.agentic.reranking import DirectAnswerReranker
from app.agentic.routing import TopicRouter
from app.agentic.service import AgenticRAGService
from app.agentic.store import AgenticSessionStore, stable_json
from app.generation.security import redact_sensitive_text
from app.generation.settings import load_local_llm_settings


def candidate(
    chunk_id: str,
    content: str,
    *,
    section: str = "METHOD",
    rank: int = 1,
) -> dict[str, Any]:
    suffix = chunk_id.removeprefix("C_")
    return {
        "rank": rank,
        "score": 0.01,
        "retrieval_source": "hybrid_rrf",
        "chunk_id": chunk_id,
        "work_id": f"W_{suffix}",
        "document_id": f"D_{suffix}",
        "paper_id": f"P_{suffix}",
        "title": f"Paper {suffix}",
        "authors": ["Ada Researcher"],
        "year": 2026,
        "doi": None,
        "section_path": [section],
        "page_start": 2,
        "page_end": 2,
        "block_ids": [f"B_{suffix}"],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": content,
    }


EPHEMERIS = candidate(
    "C_ephemeris",
    "Carrier phase measurements were used to estimate the LEO ephemeris error.",
)
CLOCK = candidate(
    "C_clock",
    "Doppler frequency measurements were used to estimate relative clock drift.",
)
BACKGROUND = candidate(
    "C_background",
    "Introduction background: ephemeris error and clock error are major challenges.",
    section="INTRODUCTION",
)


class FakeEmbeddingProvider:
    model_name = "fixture/embedding"
    revision = "embedding-revision"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        lowered = query.lower()
        if "python" in lowered or "装饰器" in lowered:
            return [0.0, 1.0, 0.0]
        if "rrf" in lowered:
            return [0.4, 0.4, 0.2]
        return [1.0, 0.0, 0.0]


class FakeRerankerProvider:
    model_name = "fixture/reranker"
    revision = "reranker-revision"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [100.0 if "Introduction background" in value else 1.0 for value in documents]


class FakeRuntime:
    def __init__(self, *, staged: bool = True) -> None:
        self.embedding_provider = FakeEmbeddingProvider()
        self.reranker_provider = FakeRerankerProvider()
        self.calls = 0
        self.staged = staged

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        normalized = query.lower()
        if self.staged and self.calls <= 3:
            values = [dict(EPHEMERIS)]
        elif any(term in normalized for term in ("clock", "钟漂", "时钟", "多普勒")):
            values = [dict(CLOCK)]
        elif "背景" in normalized:
            values = [dict(BACKGROUND), dict(EPHEMERIS)]
        else:
            values = [dict(EPHEMERIS), dict(CLOCK)]
        for index, value in enumerate(values, 1):
            value["rank"] = index
        return {
            "retriever": "hybrid_rrf",
            "result_count": len(values),
            "results": values,
        }


class FakeReasoningProvider:
    model_name = "fixture/agentic-llm"

    def resolve_route(
        self,
        standalone_query: str,
        topic_summary: str,
        signals: dict[str, float],
    ) -> RoutingLLMDecision:
        return RoutingLLMDecision(
            relation="related_subtopic",
            confidence=0.6,
            reason="fixture ambiguity",
            context_dependent=False,
            standalone_query=standalone_query,
            reuse_previous_evidence=False,
            requires_new_retrieval=True,
        )

    def check_coverage(
        self,
        plan: QueryPlan,
        evidence: Sequence[dict[str, Any]],
    ) -> CoverageReport:
        return deterministic_coverage(plan, evidence)

    @staticmethod
    def _current_sources(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for message in reversed(messages):
            if message["content"].startswith("[EVENT:state_delta]"):
                payload = json.loads(message["content"].split("\n", 1)[1])
                if "current_sources" in payload:
                    return payload["current_sources"]
        return []

    def generate_answer(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[AgenticAnswerDraft, dict[str, Any]]:
        sources = self._current_sources(messages)
        claims = [
            AgenticClaim(
                claim_id=f"C{index}",
                text=(
                    "载波相位观测被用于估计星历误差。"
                    if "ephemeris" in source["chunk_id"]
                    else "多普勒频率观测被用于估计相对时钟漂移。"
                ),
                category="measurement",
                source_ids=[source["source_id"]],
                evidence_ids=[source["evidence_id"]],
            )
            for index, source in enumerate(sources, 1)
        ]
        return (
            AgenticAnswerDraft(answerable=True, claims=claims),
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 80,
                    "prompt_cache_hit_tokens": 800,
                    "prompt_cache_miss_tokens": 200,
                }
            },
        )

    def validate_semantic(
        self,
        query: str,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        evidence: Sequence[dict[str, Any]],
    ) -> SemanticValidationReport:
        return deterministic_semantic_validation(
            query,
            plan,
            draft,
            evidence,
            structural_valid=True,
        )

    def repair_answer(
        self,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        validation: SemanticValidationReport,
        evidence: Sequence[dict[str, Any]],
    ) -> AgenticAnswerDraft:
        return deterministic_repair(draft, validation)


def config(*, rounds: int = 2, reranker: bool = True) -> AgenticRAGConfig:
    return AgenticRAGConfig(
        candidate_limit=8,
        rerank_top_k=4,
        final_top_k=4,
        max_retrieval_rounds=rounds,
        reranker_enabled=reranker,
    )


def make_service(
    root: Path,
    *,
    staged: bool = True,
    rounds: int = 2,
) -> tuple[AgenticRAGService, FakeRuntime, AgenticSessionStore]:
    runtime = FakeRuntime(staged=staged)
    store = AgenticSessionStore(root)
    service = AgenticRAGService(
        runtime,  # type: ignore[arg-type]
        FakeReasoningProvider(),
        store,
        DirectAnswerReranker(runtime.reranker_provider),
        config(rounds=rounds),
    )
    return service, runtime, store


def topic_fixture() -> dict[str, Any]:
    return {
        "topic_id": "T001",
        "topic_summary": "低轨卫星星历与时钟误差估计中的观测量",
        "entities": ["低轨卫星", "星历", "时钟误差", "观测量"],
    }


def test_topic_router_same_related_and_new_topic() -> None:
    router = TopicRouter(FakeEmbeddingProvider(), config(), ambiguity_resolver=None)
    registry = [dict(CLOCK, evidence_id="E001")]

    same = router.route(
        "那为什么多普勒能够估计钟漂？",
        topic_fixture(),
        [CLOCK],
        registry,
    )
    related = router.route("RRF 中的 k=60 是什么意思？", topic_fixture(), [], registry)
    new = router.route("Python 装饰器是什么？", topic_fixture(), [], registry)

    assert same.relation == "same_topic"
    assert same.context_dependent is True
    assert "那个" not in same.standalone_query
    assert "相对时钟漂移" in same.standalone_query
    assert same.reuse_previous_evidence is True
    assert related.relation == "related_subtopic"
    assert new.relation == "new_topic"


def test_query_planner_excludes_predicted_ephemeris_from_measurements() -> None:
    plan = QueryPlanner().plan("哪些观测量用于估计星历和时钟误差？")

    assert plan.intent == "fact_list"
    assert plan.target_category == "measurement"
    assert {"input", "prior", "state"}.issubset(plan.excluded_categories)
    assert len(plan.subquestions) == 2
    assert any("预测星历" in value for value in plan.answer_constraints)


def test_direct_answer_reranker_beats_background_and_can_fallback() -> None:
    plan = QueryPlanner().plan("哪些观测量用于估计时钟误差？")
    reranker = DirectAnswerReranker(FakeRerankerProvider())

    result = reranker.rerank(
        "哪些观测量用于估计时钟误差？",
        [BACKGROUND, CLOCK],
        2,
        plan,
    )
    fallback = DirectAnswerReranker(None, enabled=False)
    fallback_result = fallback.rerank("query", [CLOCK, BACKGROUND], 2, plan)

    assert result[0]["chunk_id"] == "C_clock"
    assert result[0]["directness_grade"] == 3
    assert reranker.last_diagnostics["fallback_used"] is False
    assert fallback_result[0]["chunk_id"] == "C_clock"
    assert fallback.last_diagnostics["fallback_used"] is True


def test_coverage_triggers_bounded_second_retrieval_round(tmp_path: Path) -> None:
    service, runtime, _ = make_service(tmp_path, staged=True, rounds=2)

    result = service.answer("哪些观测量用于估计星历和时钟误差？", session_id="demo")

    assert result["answerable"] is True
    assert len(result["retrieval_rounds"]) == 2
    assert result["retrieval_rounds"][0]["coverage_status"] != "sufficient"
    assert result["retrieval_rounds"][1]["coverage_status"] == "sufficient"
    assert result["coverage"]["overall_sufficient"] is True
    assert runtime.calls == 4


def test_session_persists_events_and_reuses_stable_evidence(tmp_path: Path) -> None:
    first_service, _, first_store = make_service(tmp_path, staged=False)
    first = first_service.answer(
        "哪些观测量用于估计星历和时钟误差？",
        session_id="persistent",
    )
    evidence_before = first_store.list_evidence("persistent", "T001")
    event_count_before = len(first_store.list_events("persistent", "T001"))

    second_service, _, second_store = make_service(tmp_path, staged=False)
    second = second_service.answer(
        "那为什么多普勒能够估计钟漂？",
        session_id="persistent",
    )
    evidence_after = second_store.list_evidence("persistent", "T001")
    events_after = second_store.list_events("persistent", "T001")

    assert first["session"]["topic_id"] == "T001"
    assert second["session"]["relation"] == "same_topic"
    assert second["session"]["topic_id"] == "T001"
    assert [item["evidence_id"] for item in evidence_before] == ["E001", "E002"]
    assert [item["evidence_id"] for item in evidence_after] == ["E001", "E002"]
    assert len(events_after) > event_count_before
    assert [event["ordinal"] for event in events_after] == list(
        range(1, len(events_after) + 1)
    )
    evidence_events = [
        event for event in events_after if event["event_type"] == "evidence_added"
    ]
    assert len(evidence_events) == 2
    assert second["diagnostics"]["metrics"]["reused_evidence_count"] > 0


def test_semantic_validation_rejects_predicted_ephemeris_as_observable() -> None:
    plan = QueryPlanner().plan("使用了哪些观测量？")
    evidence = [
        dict(
            EPHEMERIS,
            evidence_id="E001",
            source_id="S1",
            directness_grade=2,
            content="Carrier phase measurements and predicted ephemerides were inputs.",
        )
    ]
    draft = AgenticAnswerDraft(
        answerable=True,
        claims=[
            AgenticClaim(
                claim_id="C1",
                text="预测星历是一种用于估计误差的观测量。",
                category="measurement",
                source_ids=["S1"],
                evidence_ids=["E001"],
            )
        ],
    )

    validation = deterministic_semantic_validation(
        "使用了哪些观测量？",
        plan,
        draft,
        evidence,
        structural_valid=True,
    )
    repaired = deterministic_repair(draft, validation)

    claim_result = validation.claim_results[0]
    assert validation.valid is False
    assert claim_result.category_correct is False
    assert claim_result.entailment != "entailed"
    assert claim_result.repair_action in {"rewrite", "drop"}
    assert repaired.answerable is False
    assert repaired.claims == []


def test_append_only_prompt_is_strict_prefix_and_serialization_is_stable(
    tmp_path: Path,
) -> None:
    store = AgenticSessionStore(tmp_path)
    store.create_session("topic", session_id="prefix")
    topic = store.create_topic(
        "prefix",
        relation="new_topic",
        topic_summary="LEO timing",
        user_goal="study timing",
        entities=["leo"],
    )
    topic_id = str(topic["topic_id"])
    store.append_event("prefix", topic_id, "user_query", {"query": "Q1"})
    store.append_event("prefix", topic_id, "query_analysis", {"plan": "P1"})
    round_one = build_topic_messages(store.list_events("prefix", topic_id))
    store.append_event("prefix", topic_id, "answer", {"answer": "A1"})
    round_two = build_topic_messages(store.list_events("prefix", topic_id))

    assert round_two[: len(round_one)] == round_one
    assert stable_json(round_one) == stable_json(round_one)


def test_compaction_appends_without_deleting_history(tmp_path: Path) -> None:
    store = AgenticSessionStore(tmp_path)
    store.create_session("topic", session_id="compact")
    topic = store.create_topic(
        "compact",
        relation="new_topic",
        topic_summary="LEO timing",
        user_goal="study timing",
        entities=["leo"],
    )
    topic_id = str(topic["topic_id"])
    for index in range(12):
        store.append_event(
            "compact",
            topic_id,
            "user_query" if index % 2 == 0 else "state_delta",
            {"index": index, "content": "evidence context " * 20},
        )
    before = store.list_events("compact", topic_id)
    report = compact_topic(store, "compact", topic_id, recent_event_count=4)
    after = store.list_events("compact", topic_id)

    assert len(after) == len(before) + 1
    assert after[:-1] == before
    assert report.compaction_ordinal == len(after)
    assert build_topic_messages(after)[1]["content"].startswith("[EVENT:compaction]")


def test_agentic_cli_recovers_session_and_redacts_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "sessions.sqlite3"
    secret = "test-secret-key-never-persist"
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "answer_provider_from_args", lambda args: object())

    def service_factory(args: Any, answer_provider: Any) -> AgenticRAGService:
        runtime = FakeRuntime(staged=False)
        return AgenticRAGService(
            runtime,  # type: ignore[arg-type]
            FakeReasoningProvider(),
            AgenticSessionStore(tmp_path, database),
            DirectAnswerReranker(runtime.reranker_provider),
            config(),
        )

    monkeypatch.setattr(cli, "agentic_service_from_args", service_factory)
    common = [
        "--retrieval-mode",
        "agentic",
        "--session-id",
        "cli_demo",
        "--session-db-path",
        str(database),
        "--llm-api-key",
        secret,
    ]
    cli.main(["answer", "哪些观测量用于估计星历和时钟误差？", *common])
    first = json.loads(capsys.readouterr().out)
    cli.main(["answer", "那为什么多普勒能够估计钟漂？", *common])
    second = json.loads(capsys.readouterr().out)

    assert first["session"]["session_created"] is True
    assert second["session"]["session_created"] is False
    assert second["session"]["relation"] == "same_topic"
    assert secret not in json.dumps(first, ensure_ascii=False)
    assert secret not in json.dumps(second, ensure_ascii=False)
    assert secret.encode() not in database.read_bytes()

    cli.main(
        [
            "session",
            "--session-db-path",
            str(database),
            "show",
            "cli_demo",
        ]
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["topics"][0]["evidence_count"] == 2


def test_semantic_retrieve_more_uses_remaining_round_then_regenerates(
    tmp_path: Path,
) -> None:
    class RetrieveMoreProvider(FakeReasoningProvider):
        def __init__(self) -> None:
            self.generation_count = 0
            self.validation_count = 0

        def generate_answer(
            self,
            messages: list[dict[str, str]],
        ) -> tuple[AgenticAnswerDraft, dict[str, Any]]:
            self.generation_count += 1
            return super().generate_answer(messages)

        def validate_semantic(
            self,
            query: str,
            plan: QueryPlan,
            draft: AgenticAnswerDraft,
            evidence: Sequence[dict[str, Any]],
        ) -> SemanticValidationReport:
            self.validation_count += 1
            if self.validation_count == 1:
                return SemanticValidationReport(
                    valid=False,
                    issues=["citation is indirect"],
                    structural_valid=True,
                    semantic_valid=False,
                    claim_results=[
                        SemanticClaimResult(
                            claim_id=claim.claim_id,
                            entailment="partially_entailed",
                            query_aligned=True,
                            category_correct=True,
                            citation_direct=False,
                            reason="需要更直接证据。",
                            repair_action="retrieve_more",
                            revised_claim=None,
                        )
                        for claim in draft.claims
                    ],
                    requires_retrieval=True,
                    followup_queries=["clock drift direct measurement evidence"],
                )
            return super().validate_semantic(query, plan, draft, evidence)

    runtime = FakeRuntime(staged=False)
    provider = RetrieveMoreProvider()
    service = AgenticRAGService(
        runtime,  # type: ignore[arg-type]
        provider,
        AgenticSessionStore(tmp_path),
        DirectAnswerReranker(runtime.reranker_provider),
        config(rounds=2),
    )

    result = service.answer("使用了哪些观测量？", session_id="semantic_more")

    assert result["answerable"] is True
    assert len(result["retrieval_rounds"]) == 2
    assert result["retrieval_rounds"][1]["queries"] == [
        "clock drift direct measurement evidence"
    ]
    assert provider.generation_count == 2
    assert provider.validation_count == 2


def test_first_compaction_event_becomes_new_prompt_prefix(tmp_path: Path) -> None:
    store = AgenticSessionStore(tmp_path)
    store.create_session("topic", session_id="first_compaction")
    topic = store.create_topic(
        "first_compaction",
        relation="new_topic",
        topic_summary="LEO timing",
        user_goal="study timing",
        entities=["leo"],
    )
    compact_topic(store, "first_compaction", str(topic["topic_id"]))

    messages = build_topic_messages(
        store.list_events("first_compaction", str(topic["topic_id"]))
    )

    assert len(messages) == 2
    assert messages[1]["content"].startswith("[EVENT:compaction]")


def test_agentic_provider_repairs_empty_choices_once() -> None:
    class EmptyThenValidProvider:
        model_name = "fixture/model"

        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(
            self,
            messages: list[dict[str, str]],
        ) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"choices": []}
            content = RoutingLLMDecision(
                relation="same_topic",
                confidence=0.9,
                reason="follow-up",
                context_dependent=True,
                standalone_query="standalone",
                reuse_previous_evidence=True,
                requires_new_retrieval=True,
            ).model_dump_json()
            return {"choices": [{"message": {"content": content}}]}

    inner = EmptyThenValidProvider()
    provider = OpenAIAgenticReasoningProvider(inner)  # type: ignore[arg-type]

    result, diagnostics = provider._complete(  # noqa: SLF001
        "router_test",
        [{"role": "user", "content": "route"}],
        RoutingLLMDecision,
    )

    assert result.relation == "same_topic"
    assert diagnostics["structure_repair_attempts"] == 1
    assert inner.calls == 2


def test_deepseek_api_key_alias_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ds-test-secret-value"
    monkeypatch.delenv("LEO_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    settings = load_local_llm_settings(tmp_path)

    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == secret
    assert secret not in redact_sensitive_text(
        f"Authorization: Bearer {secret}; api_key={secret}",
        known_secrets=(secret,),
    )


def test_agentic_metric_interface_aggregates_optional_quality_fields() -> None:
    metrics = AgenticMetricsRecord(
        retrieval_rounds=2,
        new_evidence_count=2,
        reused_evidence_count=1,
        coverage_sufficient=True,
        citation_count=2,
        entailed_claim_count=2,
        total_claim_count=2,
        retrieval_recall_at_k=0.8,
        reranker_recall_at_k=0.9,
        reciprocal_rank=1.0,
        ndcg_at_k=0.95,
        citation_precision=1.0,
        citation_recall=0.5,
        claim_entailment_accuracy=1.0,
        answerable_correct=True,
        first_turn=True,
        latency_ms=25.0,
        compaction_count=1,
    )

    result = aggregate_agentic_metrics([metrics])

    assert result["retrieval_recall_at_k"] == 0.8
    assert result["reranker_recall_at_k"] == 0.9
    assert result["citation_recall"] == 0.5
    assert result["answerable_accuracy"] == 1.0
    assert result["average_first_turn_latency_ms"] == 25.0
    assert result["compaction_count"] == 1


def test_changed_evidence_appends_new_content_with_stable_id(tmp_path: Path) -> None:
    store = AgenticSessionStore(tmp_path)
    store.create_session("topic", session_id="changed")
    topic = store.create_topic(
        "changed",
        relation="new_topic",
        topic_summary="LEO",
        user_goal="LEO",
        entities=["leo"],
    )
    topic_id = str(topic["topic_id"])
    first, first_created = store.register_evidence(
        "changed",
        topic_id,
        dict(EPHEMERIS),
        1,
    )
    changed = dict(EPHEMERIS, content="Updated direct paper evidence.")
    second, second_created = store.register_evidence(
        "changed",
        topic_id,
        changed,
        2,
    )

    assert first_created is True
    assert second_created is True
    assert first["evidence_id"] == second["evidence_id"] == "E001"
    assert first["content_hash"] != second["content_hash"]
    assert store.list_evidence("changed", topic_id)[0]["content"] == changed["content"]


def test_agentic_config_loads_dotenv_with_environment_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LEO_AGENTIC_CANDIDATE_LIMIT=30\nLEO_AGENTIC_RRF_K=55\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEO_AGENTIC_RRF_K", "61")

    loaded = AgenticRAGConfig.from_environment(env_file)

    assert loaded.candidate_limit == 30
    assert loaded.rrf_k == 61


def test_relative_session_database_is_resolved_from_project_root(
    tmp_path: Path,
) -> None:
    store = AgenticSessionStore(tmp_path, Path("private/sessions.sqlite3"))

    assert store.database_path == (tmp_path / "private" / "sessions.sqlite3")
    assert store.database_path.is_file()
