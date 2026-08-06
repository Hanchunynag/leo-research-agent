from __future__ import annotations

from pathlib import Path

from app.contracts import IndexProfile
from app.knowledge_engine import IndexGenerationRepository, KnowledgeIndexService
from tests.test_stage2_corpus_workspace import write_fixture
from tests.test_stage4_incremental_indexing import MemoryIncrementalEngine, _document


def test_delete_fork_has_no_target_residual_and_preserves_old_generation(
    tmp_path: Path,
) -> None:
    write_fixture(tmp_path, "D_keep")
    write_fixture(tmp_path, "D_delete", "delete me")
    engine = MemoryIncrementalEngine()
    generations = IndexGenerationRepository(tmp_path)
    service = KnowledgeIndexService(engine, generations)
    profile = IndexProfile("profile-1")
    first, _ = service.build_generation(
        [_document("D_keep", "hash-keep"), _document("D_delete", "hash-delete")],
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )

    deleted, metrics = service.build_incremental_generation(
        operation="delete",
        workspace_id="default",
        scope_version=2,
        corpus_version="corpus-2",
        profile=profile,
        document_ids=["D_delete"],
        activate=True,
    )

    assert engine.mappings[deleted.generation_id] == {"D_keep"}
    assert "D_delete" not in engine.mappings[deleted.generation_id]
    assert engine.mappings[first.generation_id] == {"D_keep", "D_delete"}
    assert metrics["changed_document_ids"] == ["D_delete"]
    assert metrics["full_rebuild"] is False
    assert (tmp_path / "data/canonical/P_delete/paper.json").is_file()


def test_workspace_generation_delete_does_not_change_other_workspace(
    tmp_path: Path,
) -> None:
    engine = MemoryIncrementalEngine()
    generations = IndexGenerationRepository(tmp_path)
    service = KnowledgeIndexService(engine, generations)
    profile = IndexProfile("profile-1")
    documents = [_document("D_shared", "hash-shared")]
    service.build_generation(
        documents,
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )
    other, _ = service.build_generation(
        documents,
        workspace_id="other",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )

    service.build_incremental_generation(
        operation="delete",
        workspace_id="default",
        scope_version=2,
        corpus_version="corpus-2",
        profile=profile,
        document_ids=["D_shared"],
        activate=True,
    )

    assert engine.mappings[other.generation_id] == {"D_shared"}

