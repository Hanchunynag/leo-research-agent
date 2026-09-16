import json
from pathlib import Path
from typing import Any

from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.dense import build_dense_index
from app.indexing.paper import build_paper_bm25_index, write_paper_records
from app.indexing.paper_dense import build_paper_dense_index
from app.knowledge.corpus import corpus_summary, knowledge_index_readiness
from app.storage import write_jsonl_atomic


class _EmbeddingProvider:
    model_name = "fixture/bge-m3"
    revision = "fixture-revision-1"
    artifact_fingerprint = None
    normalized = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_corpus_summary_separates_metadata_only_and_indexed_papers(
    tmp_path: Path,
) -> None:
    knowledge = tmp_path / "data" / "knowledge"
    canonical = tmp_path / "data" / "canonical" / "P_indexed"
    structures = knowledge / "structures"
    canonical.mkdir(parents=True)
    structures.mkdir(parents=True)
    (canonical / "paper.json").write_text(
        json.dumps({"paper_id": "P_indexed"}), encoding="utf-8"
    )
    papers = [
        {"paper_id": "P_indexed", "title": "Indexed"},
        {"paper_id": "EXT_abstract", "title": "Abstract only"},
    ]
    write_jsonl_atomic(knowledge / "paper_records.jsonl", papers)
    chunks = [
        {
            "chunk_id": "C1",
            "paper_id": "P_indexed",
            "document_id": "D_indexed",
            "content": "evidence",
            "section_path": ["Results"],
            "parent_contexts": [],
            "overlap_context": None,
        }
    ]
    write_jsonl_atomic(knowledge / "chunks.jsonl", chunks)
    (structures / "D_indexed.structure.json").write_text(
        json.dumps({"sections": [{"section_id": "S1"}]}), encoding="utf-8"
    )
    write_bm25_index(tmp_path, build_bm25_index(chunks))

    summary = corpus_summary(tmp_path)

    assert summary.paper_count == 2
    assert summary.canonical_document_count == 1
    assert summary.indexed_paper_count == 1
    assert summary.indexed_document_count == 1
    assert summary.section_count == 1
    assert summary.chunk_count == 1
    assert summary.searchable_chunk_count == 1
    assert summary.bm25_index_count == 1
    assert summary.metadata_only_paper_ids == ["EXT_abstract"]


def test_knowledge_index_readiness_is_not_initialized_for_empty_corpus(
    tmp_path: Path,
) -> None:
    status = knowledge_index_readiness(tmp_path)

    assert status["status"] == "not_initialized"
    assert status["paper_bm25"]["status"] == "not_initialized"
    assert status["paper_dense"]["status"] == "not_initialized"
    assert status["content_bm25"]["status"] == "not_initialized"
    assert status["content_dense"]["status"] == "not_initialized"


def test_knowledge_index_readiness_reports_all_four_indexes_ready(
    tmp_path: Path,
) -> None:
    papers: list[dict[str, Any]] = [
        {
            "paper_id": "P001",
            "title": "Alpha paper",
            "abstract": "Alpha abstract",
            "authors": ["Researcher"],
            "publication_date": "2026-01-01",
            "venue": "LEO Journal",
            "keywords": ["alpha"],
        },
        {
            "paper_id": "P002",
            "title": "Beta paper",
            "abstract": "Beta abstract",
            "authors": ["Researcher"],
            "publication_date": "2026-01-02",
            "venue": "LEO Journal",
            "keywords": ["beta"],
        },
    ]
    chunks = [
        {
            "chunk_id": "C001",
            "paper_id": paper["paper_id"],
            "document_id": f"D_{paper['paper_id']}",
            "content": f"{paper['title']} evidence",
            "section_path": ["Results"],
            "parent_contexts": [],
            "overlap_context": None,
        }
        for paper in papers
    ]
    write_paper_records(tmp_path, papers)
    write_jsonl_atomic(tmp_path / "data" / "knowledge" / "chunks.jsonl", chunks)
    for paper in papers:
        canonical = tmp_path / "data" / "canonical" / paper["paper_id"]
        canonical.mkdir(parents=True)
        (canonical / "paper.json").write_text(
            json.dumps({"paper_id": paper["paper_id"]}), encoding="utf-8"
        )

    write_bm25_index(tmp_path, build_bm25_index(chunks))
    build_paper_bm25_index(tmp_path)
    provider = _EmbeddingProvider()
    build_dense_index(tmp_path, provider)
    build_paper_dense_index(tmp_path, provider)

    status = knowledge_index_readiness(tmp_path)

    assert status["status"] == "ready"
    assert status["index_consistent"] is True
    assert status["paper_bm25"]["indexed_count"] == 2
    assert status["paper_dense"]["indexed_count"] == 2
    assert status["content_bm25"]["indexed_count"] == 2
    assert status["content_dense"]["indexed_count"] == 2
    assert status["embedding_revision"] == "fixture-revision-1"
    assert status["manifest_status"]["status"] == "ready"
