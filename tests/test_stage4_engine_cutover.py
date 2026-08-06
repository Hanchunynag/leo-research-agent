from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.contracts import CandidateEvidence, EvidenceRequest, IndexGeneration
from app.knowledge_engine import (
    EngineCutoverService,
    IndexGenerationRepository,
    KnowledgeServingConfigRepository,
    UnifiedKnowledgeService,
)
from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture


def _generation(generation_id: str, state: str = "pending") -> IndexGeneration:
    return IndexGeneration(
        generation_id=generation_id,
        corpus_version="corpus-1",
        scope_version=1,
        state=state,  # type: ignore[arg-type]
        document_count=1,
        chunk_count=1,
        workspace_id="default",
        index_profile_id="profile-pinned",
    )


def _active(generations: IndexGenerationRepository, generation_id: str) -> None:
    generations.create(_generation(generation_id))
    generations.transition(generation_id, "validating")
    generations.activate(generation_id)


def _accepted() -> dict[str, Any]:
    return {"passed": True, "official_cutover_approved": True, "failures": []}


def test_cutover_requires_machine_acceptance_and_keeps_legacy_on_failure(
    tmp_path: Path,
) -> None:
    generations = IndexGenerationRepository(tmp_path)
    _active(generations, "IG_accepted")
    repository = KnowledgeServingConfigRepository(tmp_path)
    service = EngineCutoverService(repository, generations)

    with pytest.raises(PermissionError, match="Acceptance 未通过"):
        service.switch_to_lightrag(
            "IG_accepted",
            {"passed": False, "official_cutover_approved": False},
            actor="test",
        )

    config = repository.load()
    assert config.official_engine == "legacy"
    assert config.official_generation_id is None
    assert repository.audit_records() == ()


def test_legacy_to_lightrag_cutover_pins_generation_and_writes_audit(
    tmp_path: Path,
) -> None:
    generations = IndexGenerationRepository(tmp_path)
    _active(generations, "IG_accepted")
    repository = KnowledgeServingConfigRepository(tmp_path)
    service = EngineCutoverService(repository, generations)

    config = service.switch_to_lightrag("IG_accepted", _accepted(), actor="operator")

    assert config.official_engine == "lightrag"
    assert config.official_generation_id == "IG_accepted"
    assert repository.load() == config
    record = repository.audit_records()[0]
    assert record["operation"] == "cutover"
    assert record["actor"] == "operator"
    assert record["details"]["profile_id"] == "profile-pinned"


@pytest.mark.parametrize("state", ["building", "failed"])
def test_building_or_failed_generation_is_rejected(
    tmp_path: Path, state: str
) -> None:
    generations = IndexGenerationRepository(tmp_path)
    generations.create(_generation("IG_bad"))
    if state == "failed":
        generations.transition("IG_bad", "failed")
    repository = KnowledgeServingConfigRepository(tmp_path)

    with pytest.raises(ValueError, match="不能用于服务"):
        EngineCutoverService(repository, generations).switch_to_lightrag(
            "IG_bad", _accepted(), actor="operator"
        )

    assert repository.load().official_engine == "legacy"


class _LegacyRuntime:
    embedding_provider = object()
    reranker_provider = None

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("LightRAG Official 时不应调用 Legacy retrieve")


class _PinnedEngine:
    serving_generation_id = "IG_pinned"

    def retrieve_candidates(
        self, request: EvidenceRequest
    ) -> tuple[CandidateEvidence, ...]:
        return (
            CandidateEvidence(
                "candidate-1",
                request.request_id,
                request.workspace_id,
                request.scope_version,
                "canonical text",
                1.0,
                "lightrag_chunk",
                work_id="W_001",
                document_id="D_001",
                chunk_id="D_001_c1",
            ),
        )


def test_query_uses_one_pinned_official_snapshot(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    pipeline = EvidenceIntelligencePipeline(corpus, workspaces)
    service = UnifiedKnowledgeService(
        _LegacyRuntime(),
        pipeline,
        official_engine_name="lightrag",
        official_engine=_PinnedEngine(),
        official_generation_id="IG_pinned",
    )

    result = service.retrieve("query", limit=5)

    assert result["retriever"] == "lightrag"
    assert result["results"][0]["chunk_id"] == "D_001_c1"
    assert service.last_diagnostics["official_engine"] == "lightrag"
    assert service.last_diagnostics["official_generation_id"] == "IG_pinned"


def test_generation_pin_mismatch_fails_closed(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    service = UnifiedKnowledgeService(
        _LegacyRuntime(),
        EvidenceIntelligencePipeline(corpus, workspaces),
        official_engine_name="lightrag",
        official_engine=_PinnedEngine(),
        official_generation_id="IG_other",
    )

    with pytest.raises(RuntimeError, match="Generation Pin 不一致"):
        service.retrieve("query", limit=5)

