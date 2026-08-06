from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.contracts import CandidateEvidence
from app.corpus import CanonicalCorpusService
from app.workspaces import WorkspaceRepository, WorkspaceService


def write_fixture(root: Path, document_id: str, content: str = "canonical text") -> None:
    paper_id = document_id.replace("D_", "P_", 1)
    sha = hashlib.sha256(content.encode()).hexdigest()
    canonical = root / "data" / "canonical" / paper_id / "paper.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "identity": {"document_id": document_id, "work_id": "W_001"},
                "source": {"sha256": sha},
                "metadata": {"title": "Fixture"},
                "blocks": [{"block_id": f"{paper_id}_b1", "page_number": 2, "text": content}],
            }
        ),
        encoding="utf-8",
    )
    structures = root / "data" / "knowledge" / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    (structures / f"{document_id}.structure.json").write_text(
        json.dumps({"sections": [{"section_id": f"{document_id}_s0001"}]}),
        encoding="utf-8",
    )
    chunks = root / "data" / "knowledge" / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    chunk = {
        "chunk_id": f"{document_id}_c1",
        "document_id": document_id,
        "work_id": "W_001",
        "section_id": f"{document_id}_s0001",
        "page_start": 2,
        "page_end": 2,
        "block_ids": [f"{paper_id}_b1"],
        "content": content,
    }
    (chunks / f"{document_id}.chunks.json").write_text(
        json.dumps({"chunks": [chunk]}), encoding="utf-8"
    )
    knowledge = root / "data" / "knowledge"
    chunks_jsonl = knowledge / "chunks.jsonl"
    existing = chunks_jsonl.read_text(encoding="utf-8") if chunks_jsonl.is_file() else ""
    chunks_jsonl.write_text(existing + json.dumps(chunk) + "\n", encoding="utf-8")


def candidate(document_id: str, *, scope_version: int = 1) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=f"E-{document_id}",
        request_id="REQ-1",
        workspace_id="default",
        scope_version=scope_version,
        content="evidence",
        score=0.8,
        retrieval_source="chunk",
        document_id=document_id,
    )


def test_canonical_repositories_preserve_ids_and_detect_duplicate_hash(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    corpus = CanonicalCorpusService(tmp_path)

    document = corpus.list_documents()[0]
    locator = corpus.locate_chunk("D_001_c1")

    assert document.document_id == "D_001"
    assert corpus.duplicate_for_hash(document.source_sha256) == document
    assert locator is not None
    assert locator.page_start == 2
    assert locator.block_ids == ("P_001_b1",)


def test_default_workspace_contains_all_existing_documents(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    write_fixture(tmp_path, "D_002", "second")
    service = WorkspaceService(tmp_path)

    workspace = service.ensure_default()

    assert workspace.workspace_id == "default"
    assert service.active_document_ids("default", 1) == {"D_001", "D_002"}


def test_scope_hard_filter_blocks_excluded_and_cross_workspace_candidates(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    write_fixture(tmp_path, "D_002", "second")
    repository = WorkspaceRepository(tmp_path)
    service = WorkspaceService(tmp_path, repository=repository)
    service.ensure_default()
    scope = service.replace_scope_documents("default", ["D_001"])

    allowed = candidate("D_001", scope_version=scope.scope_version)
    excluded = candidate("D_002", scope_version=scope.scope_version)
    stale = candidate("D_001", scope_version=1)

    assert service.filter_candidates("default", scope.scope_version, [allowed, excluded, stale]) == (allowed,)
    with pytest.raises(ValueError, match="尚不存在"):
        service.require_scope("default", 99)
