from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.contracts import (
    CandidateEvidence,
    Document,
    EvidenceRequest,
    IndexGeneration,
    IndexProfile,
)
from app.corpus import DocumentVersionRepository
from app.knowledge_engine import IndexGenerationRepository, KnowledgeIndexService


def _document(document_id: str, content_hash: str) -> Document:
    return Document(
        document_id=document_id,
        paper_id=document_id.replace("D_", "P_"),
        work_id="W_shared",
        source_sha256=content_hash,
        canonical_path=f"data/canonical/{document_id}/paper.json",
        chunk_ids=(f"{document_id}_c1",),
    )


class MemoryIncrementalEngine:
    def __init__(self) -> None:
        self.mappings: dict[str, set[str]] = {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]:
        ids = {value.document_id for value in documents}
        self.mappings[generation.generation_id] = ids
        self.calls.append(("index", tuple(sorted(ids))))
        return {
            "indexed_document_count": len(ids),
            "mapped_chunk_count": sum(len(value.chunk_ids) for value in documents),
        }

    def fork_generation(
        self, previous: IndexGeneration, target: IndexGeneration
    ) -> None:
        self.mappings[target.generation_id] = set(
            self.mappings[previous.generation_id]
        )

    def _load_mapping(
        self, generation: IndexGeneration
    ) -> Mapping[str, Mapping[str, str]]:
        return {
            f"internal:{document_id}": {
                "document_id": document_id,
                "chunk_id": f"{document_id}_c1",
            }
            for document_id in self.mappings.get(generation.generation_id, set())
        }

    def update_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]:
        ids = tuple(sorted(value.document_id for value in documents))
        self.calls.append(("update", ids))
        self.mappings[generation.generation_id].update(ids)
        return {
            "updated_document_count": len(ids),
            "mapped_chunk_count": len(self.mappings[generation.generation_id]),
            "full_rebuild": False,
        }

    def delete_documents(
        self,
        document_ids: Sequence[str],
        *,
        generation: IndexGeneration,
    ) -> Mapping[str, Any]:
        ids = tuple(sorted(document_ids))
        self.calls.append(("delete", ids))
        self.mappings[generation.generation_id].difference_update(ids)
        return {
            "deleted_document_count": len(ids),
            "mapped_chunk_count": len(self.mappings[generation.generation_id]),
            "full_rebuild": False,
        }

    def retrieve_candidates(
        self, request: EvidenceRequest
    ) -> Sequence[CandidateEvidence]:
        return ()

    def get_status(self) -> Mapping[str, Any]:
        return {}


def test_incremental_add_only_processes_new_document(tmp_path: Path) -> None:
    engine = MemoryIncrementalEngine()
    generations = IndexGenerationRepository(tmp_path)
    service = KnowledgeIndexService(engine, generations)
    profile = IndexProfile("profile-1")
    first, _ = service.build_generation(
        [_document("D_old", "hash-old")],
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-1",
        profile=profile,
        activate=True,
    )

    second, metrics = service.build_incremental_generation(
        operation="add",
        workspace_id="default",
        scope_version=1,
        corpus_version="corpus-2",
        profile=profile,
        documents=[_document("D_new", "hash-new")],
        activate=True,
    )

    assert second.previous_generation_id == first.generation_id
    assert engine.calls == [("index", ("D_old",)), ("update", ("D_new",))]
    assert metrics["changed_document_ids"] == ["D_new"]
    assert metrics["unrelated_history_reprocessed_count"] == 0
    assert metrics["full_document_extraction_count"] == 0
    assert metrics["full_rebuild"] is False
    assert engine.mappings[second.generation_id] == {"D_old", "D_new"}


def test_same_content_hash_and_extraction_profile_reuses_receipt(
    tmp_path: Path,
) -> None:
    versions = DocumentVersionRepository(tmp_path)
    first = versions.register(
        work_id="W_1",
        document_id="D_v1",
        content_hash="sha-1",
        extraction_profile="mineru-v1",
    )
    duplicate = versions.register(
        work_id="W_1",
        document_id="D_duplicate",
        content_hash="sha-1",
        extraction_profile="mineru-v1",
    )
    updated = versions.register(
        work_id="W_1",
        document_id="D_v2",
        content_hash="sha-2",
        extraction_profile="mineru-v1",
    )

    assert first.requires_extraction is True
    assert duplicate.requires_extraction is False
    assert duplicate.receipt == first.receipt
    assert updated.requires_extraction is True
    assert updated.receipt.work_id == first.receipt.work_id
    assert updated.receipt.version_number == 2

