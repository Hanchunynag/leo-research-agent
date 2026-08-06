from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.contracts import EvidenceRequest, IndexGeneration, IndexProfile, KnowledgeEngine
from app.corpus import CanonicalCorpusService
from app.knowledge_engine import IndexGenerationRepository, KnowledgeIndexService, LightRAGKnowledgeEngine
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture


@dataclass
class DeleteResult:
    status: str = "success"
    message: str = "deleted"


class FakeLightRAG:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    async def initialize_storages(self) -> None:
        self.state["initialized"] = self.state.get("initialized", 0) + 1

    async def finalize_storages(self) -> None:
        self.state["finalized"] = self.state.get("finalized", 0) + 1

    async def ainsert_custom_chunks(self, full_text: str, chunks: list[str], doc_id: str) -> None:
        self.state.setdefault("inserts", []).append((doc_id, list(chunks)))

    async def adelete_by_doc_id(self, document_id: str) -> DeleteResult:
        self.state.setdefault("deletes", []).append(document_id)
        return DeleteResult()

    async def aquery_data(self, query: str, param: Any) -> dict[str, Any]:
        self.state.setdefault("queries", []).append((query, param.mode))
        return self.state["query_result"]


def setup_engine(tmp_path: Path) -> tuple[LightRAGKnowledgeEngine, IndexGenerationRepository, dict[str, Any]]:
    write_fixture(tmp_path, "D_001")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    generations = IndexGenerationRepository(tmp_path)
    state: dict[str, Any] = {}
    engine = LightRAGKnowledgeEngine(
        tmp_path,
        corpus=corpus,
        workspaces=workspaces,
        generations=generations,
        client_factory=lambda path, workspace, profile: FakeLightRAG(state),
    )
    return engine, generations, state


def generation() -> IndexGeneration:
    return IndexGeneration(
        generation_id="IG_001",
        corpus_version="corpus-1",
        scope_version=1,
        state="pending",
        document_count=1,
        chunk_count=1,
        workspace_id="default",
        index_profile_id="profile-1",
    )


def test_index_generation_validates_before_activation_and_can_rollback(tmp_path: Path) -> None:
    engine, generations, _ = setup_engine(tmp_path)
    service = KnowledgeIndexService(engine, generations)
    profile = IndexProfile("profile-1")

    first, metrics = service.build_generation(
        engine.corpus.list_documents(),
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )
    second, _ = service.build_generation(
        engine.corpus.list_documents(),
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-2",
        profile=profile,
        activate=True,
    )

    assert first.state == "active"
    assert metrics["mapped_chunk_count"] == 1
    assert generations.get(first.generation_id).state == "retired"  # type: ignore[union-attr]
    assert generations.active("default") == second
    assert service.rollback("default", first.generation_id).generation_id == first.generation_id
    assert generations.get(second.generation_id).state == "retired"  # type: ignore[union-attr]


def test_incremental_update_and_delete_never_request_full_rebuild(tmp_path: Path) -> None:
    engine, generations, state = setup_engine(tmp_path)
    built = generations.create(generation())
    profile = IndexProfile("profile-1")
    documents = engine.corpus.list_documents()
    engine.index_documents(documents, generation=built, profile=profile)

    updated = engine.update_documents(documents, generation=built, profile=profile)
    deleted = engine.delete_documents(["D_001"], generation=built)

    assert updated["full_rebuild"] is False
    assert deleted["full_rebuild"] is False
    assert state["deletes"] == ["D_001", "D_001"]
    assert [value[0] for value in state["inserts"]] == ["D_001", "D_001"]


def test_lightrag_raw_results_are_mapped_to_canonical_candidate_and_scope(tmp_path: Path) -> None:
    engine, generations, state = setup_engine(tmp_path)
    built = generations.create(generation())
    profile = IndexProfile("profile-1")
    document = engine.corpus.list_documents()[0]
    engine.index_documents([document], generation=built, profile=profile)
    generations.transition("IG_001", "validating")
    generations.activate("IG_001")
    locator = engine.corpus.locate_document_chunks("D_001")[0]
    internal_id = engine._internal_chunk_id("D_001", locator.content)
    state["query_result"] = {
        "status": "success",
        "data": {
            "chunks": [{"chunk_id": internal_id, "content": "untrusted raw"}],
            "entities": [],
            "relationships": [],
        },
    }

    values = engine.retrieve_candidates(
        EvidenceRequest("REQ-1", "default", 1, "query", top_k=5)
    )

    assert isinstance(engine, KnowledgeEngine)
    assert len(values) == 1
    assert values[0].content == "canonical text"
    assert values[0].chunk_id == "D_001_c1"
    assert values[0].source_type == "lightrag_chunk"
    assert values[0].state == "retrieved"


def test_lightrag_candidate_pool_keeps_each_source_before_evidence_ranking(
    tmp_path: Path,
) -> None:
    engine, generations, state = setup_engine(tmp_path)
    built = generations.create(generation())
    document = engine.corpus.list_documents()[0]
    engine.index_documents([document], generation=built, profile=IndexProfile("profile-1"))
    generations.transition("IG_001", "validating")
    generations.activate("IG_001")
    locator = engine.corpus.locate_document_chunks("D_001")[0]
    internal_id = engine._internal_chunk_id("D_001", locator.content)
    state["query_result"] = {
        "status": "success",
        "data": {
            "chunks": [{"chunk_id": internal_id, "score": 0.2}],
            "entities": [{"source_id": internal_id, "entity_name": "A", "score": 99}],
            "relationships": [
                {
                    "source_id": internal_id,
                    "src_id": "A",
                    "tgt_id": "B",
                    "weight": 999,
                }
            ],
        },
    }

    values = engine.retrieve_candidates(
        EvidenceRequest("REQ-1", "default", 1, "query", top_k=1)
    )

    assert {value.retrieval_source for value in values} == {
        "lightrag_chunk",
        "lightrag_entity",
        "lightrag_relation",
    }
    assert {value.score for value in values} == {1.0}
    assert {value.metadata["raw_retrieval_score"] for value in values} == {
        0.2,
        99,
        999,
    }


def test_incremental_generation_forks_active_and_only_updates_changed_document(tmp_path: Path) -> None:
    engine, generations, state = setup_engine(tmp_path)
    service = KnowledgeIndexService(engine, generations)
    profile = IndexProfile("profile-1")
    documents = engine.corpus.list_documents()
    first, _ = service.build_generation(
        documents,
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )

    second, metrics = service.build_incremental_generation(
        operation="update",
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-2",
        profile=profile,
        documents=documents,
        activate=True,
    )

    assert metrics["full_rebuild"] is False
    assert metrics["incremental_operation"] == "update"
    assert second.previous_generation_id == first.generation_id
    assert generations.get(first.generation_id).state == "retired"  # type: ignore[union-attr]
    assert state["deletes"] == ["D_001"]
