from __future__ import annotations

from pathlib import Path
from typing import Any

from app.contracts import CandidateEvidence, EvidenceRequest
from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline
from app.knowledge_engine import (
    EngineCutoverService,
    IndexGenerationRepository,
    KnowledgeServingConfigRepository,
    UnifiedKnowledgeService,
)
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture
from tests.test_stage4_engine_cutover import _accepted, _active


def test_rollback_restores_previous_retired_generation(tmp_path: Path) -> None:
    generations = IndexGenerationRepository(tmp_path)
    _active(generations, "IG_previous")
    _active(generations, "IG_current")
    assert generations.get("IG_previous").state == "retired"  # type: ignore[union-attr]
    repository = KnowledgeServingConfigRepository(tmp_path)
    cutover = EngineCutoverService(repository, generations)
    cutover.switch_to_lightrag("IG_previous", _accepted(), actor="operator")
    cutover.switch_to_lightrag("IG_current", _accepted(), actor="operator")

    restored = cutover.rollback_previous(actor="operator", reason="smoke failure")

    assert restored.official_engine == "lightrag"
    assert restored.official_generation_id == "IG_previous"
    assert repository.audit_records()[-1]["operation"] == "rollback_previous"


class _LegacyRuntime:
    embedding_provider = object()
    reranker_provider = None

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "retriever": "legacy",
            "results": [
                {
                    "rank": 1,
                    "score": 1.0,
                    "retrieval_source": "legacy",
                    "work_id": "W_001",
                    "document_id": "D_001",
                    "chunk_id": "D_001_c1",
                    "content": "canonical text",
                }
            ],
        }


class _LightRAG:
    def __init__(self, generation_id: str) -> None:
        self.serving_generation_id = generation_id

    def retrieve_candidates(
        self, request: EvidenceRequest
    ) -> tuple[CandidateEvidence, ...]:
        return (
            CandidateEvidence(
                "light-1",
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


def _service(
    tmp_path: Path,
    repository: KnowledgeServingConfigRepository,
    runtime: _LegacyRuntime,
) -> UnifiedKnowledgeService:
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    config = repository.load()
    engine = (
        _LightRAG(config.official_generation_id)
        if config.official_generation_id
        else None
    )
    return UnifiedKnowledgeService(
        runtime,
        EvidenceIntelligencePipeline(corpus, workspaces),
        official_engine_name=config.official_engine,
        official_engine=engine,
        official_generation_id=config.official_generation_id,
    )


def test_lightrag_to_legacy_rollback_recovers_legacy_top_k(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    generations = IndexGenerationRepository(tmp_path)
    _active(generations, "IG_light")
    repository = KnowledgeServingConfigRepository(tmp_path)
    cutover = EngineCutoverService(repository, generations)
    runtime = _LegacyRuntime()
    before = _service(tmp_path, repository, runtime).retrieve("query", limit=5)
    cutover.switch_to_lightrag("IG_light", _accepted(), actor="operator")
    assert _service(tmp_path, repository, runtime).retrieve("query", limit=5)[
        "retriever"
    ] == "lightrag"

    cutover.switch_to_legacy(actor="operator", reason="rollback smoke")
    after = _service(tmp_path, repository, runtime).retrieve("query", limit=5)

    assert [value["chunk_id"] for value in after["results"]] == [
        value["chunk_id"] for value in before["results"]
    ]
    assert repository.load().official_engine == "legacy"

