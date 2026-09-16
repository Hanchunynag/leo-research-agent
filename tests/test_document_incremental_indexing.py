from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.indexing.dense import build_dense_index, dense_index_path
from app.indexing.paper_dense import (
    build_paper_dense_index,
    paper_dense_index_path,
)
from app.indexing.paper import write_paper_records
from app.indexing.paper import paper_retrieval_text
from app.storage import write_jsonl_atomic


class CountingEmbeddingProvider:
    model_name = "fixture/bge-m3"
    revision = "fixture-revision-1"
    artifact_fingerprint = None
    normalized = True

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in query.casefold() else [0.0, 1.0]


def _paper(paper_id: str, *, abstract: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": f"{paper_id} paper",
        "abstract": abstract,
        "authors": ["Researcher"],
        "year": 2026,
        "publication_date": "2026-01-01",
        "venue": "Journal of LEO Research",
        "keywords": ["orbit"],
        "doi": None,
        "work_id": f"W_{paper_id}",
        "document_id": f"D_{paper_id}",
    }


def _chunk(paper_id: str, chunk_id: str, content: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "work_id": f"W_{paper_id}",
        "document_id": f"D_{paper_id}",
        "section_id": f"D_{paper_id}_s0001",
        "title": f"{paper_id} paper",
        "authors": ["Researcher"],
        "year": 2026,
        "doi": None,
        "section_path": ["Results"],
        "content_zone": "main_body",
        "page_start": 1,
        "page_end": 1,
        "block_ids": [f"B_{chunk_id}"],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": content,
    }


def _write_projection(root: Path, papers: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    write_paper_records(root, papers)
    write_jsonl_atomic(root / "data" / "knowledge" / "chunks.jsonl", chunks)


def _collection_payloads(path: Path, collection: str) -> list[dict[str, Any]]:
    client = QdrantClient(path=str(path))
    try:
        points, _ = client.scroll(
            collection_name=collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        return [dict(point.payload or {}) for point in points]
    finally:
        client.close()


def test_document_level_incremental_indexing_only_embeds_new_changed_paper(tmp_path: Path) -> None:
    papers = [_paper("P001", abstract="alpha method"), _paper("P002", abstract="beta method")]
    chunks = [
        _chunk("P001", "C001", "alpha evidence"),
        _chunk("P002", "C001", "beta evidence"),
    ]
    _write_projection(tmp_path, papers, chunks)
    provider = CountingEmbeddingProvider()

    first_chunks = build_dense_index(tmp_path, provider)
    first_papers = build_paper_dense_index(tmp_path, provider)
    assert first_chunks.status == "built"
    assert first_papers.status == "built"
    assert [len(call) for call in provider.calls] == [2, 2]

    papers.append(_paper("P003", abstract="gamma method"))
    chunks.append(_chunk("P003", "C001", "gamma evidence"))
    _write_projection(tmp_path, papers, chunks)
    provider.calls.clear()

    added_chunks = build_dense_index(tmp_path, provider, changed_paper_ids={"P003"})
    added_papers = build_paper_dense_index(tmp_path, provider, changed_paper_ids={"P003"})
    assert added_chunks.status == "incremental"
    assert added_papers.status == "built"
    assert [len(call) for call in provider.calls] == [1, 3]
    assert all("P003" in text for text in provider.calls[0])
    assert {"P001", "P002", "P003"} <= {
        paper_id
        for text in provider.calls[1]
        for paper_id in ("P001", "P002", "P003")
        if paper_id in text
    }

    papers[1]["abstract"] = "beta method updated"
    chunks[1]["content"] = "beta evidence updated"
    _write_projection(tmp_path, papers, chunks)
    provider.calls.clear()
    changed_chunks = build_dense_index(tmp_path, provider, changed_paper_ids={"P002"})
    changed_papers = build_paper_dense_index(tmp_path, provider, changed_paper_ids={"P002"})
    assert changed_chunks.status == "incremental"
    assert changed_papers.status == "built"
    assert [len(call) for call in provider.calls] == [1, 3]
    assert all("P002" in text for text in provider.calls[0])
    assert {"P001", "P002", "P003"} <= {
        paper_id
        for text in provider.calls[1]
        for paper_id in ("P001", "P002", "P003")
        if paper_id in text
    }

    papers = [papers[1], papers[2]]
    chunks = [chunks[1], chunks[2]]
    _write_projection(tmp_path, papers, chunks)
    provider.calls.clear()
    deleted_chunks = build_dense_index(tmp_path, provider, changed_paper_ids={"P001"})
    deleted_papers = build_paper_dense_index(tmp_path, provider, changed_paper_ids={"P001"})
    assert deleted_chunks.status == "incremental"
    assert deleted_papers.status == "built"
    assert [len(call) for call in provider.calls] == [2]
    assert {"P002", "P003"} <= {
        paper_id
        for text in provider.calls[0]
        for paper_id in ("P002", "P003")
        if paper_id in text
    }

    chunk_payloads = _collection_payloads(dense_index_path(tmp_path), "leo_paper_chunks_dense")
    paper_payloads = _collection_payloads(paper_dense_index_path(tmp_path), "leo_papers_dense")
    assert {value["paper_id"] for value in chunk_payloads} == {"P002", "P003"}
    assert {value["paper_id"] for value in paper_payloads} == {"P002", "P003"}
    assert all(value["level"] == 2 for value in chunk_payloads)
    assert all(value["level"] == 1 and value["chunk_id"].endswith("_metadata") for value in paper_payloads)


def test_unchanged_document_does_not_call_embedding_provider(tmp_path: Path) -> None:
    papers = [_paper("P001", abstract="alpha method")]
    chunks = [_chunk("P001", "C001", "alpha evidence")]
    _write_projection(tmp_path, papers, chunks)
    provider = CountingEmbeddingProvider()
    build_dense_index(tmp_path, provider)
    build_paper_dense_index(tmp_path, provider)
    provider.calls.clear()

    chunk_report = build_dense_index(tmp_path, provider)
    paper_report = build_paper_dense_index(tmp_path, provider)
    assert chunk_report.status == "reused"
    assert paper_report.status == "reused"
    assert provider.calls == []


def test_paper_chunk_text_contains_the_complete_level_one_metadata() -> None:
    text = paper_retrieval_text(
        _paper("P001", abstract="alpha abstract")
    )
    assert "Title: P001 paper" in text
    assert "Authors: Researcher" in text
    assert "Publication date: 2026-01-01" in text
    assert "Venue: Journal of LEO Research" in text
    assert "Abstract: alpha abstract" in text
    assert "Keywords: orbit" in text
