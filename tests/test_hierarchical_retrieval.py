from __future__ import annotations

from pathlib import Path
from typing import Any

from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.paper import build_paper_bm25_index, write_paper_records
from app.indexing.paper_dense import build_paper_dense_index
from app.indexing.dense import build_dense_index
from app.retrieval.hierarchical import search_hierarchical_evidence
from app.storage import write_jsonl_atomic


class FakeEmbedding:
    model_name = "fixture/bge-m3"
    revision = "r1"
    normalized = True

    @staticmethod
    def vector(text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.vector(query)


class FakeReranker:
    model_name = "fixture/reranker"
    revision = "r1"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [2.0 if "alpha" in document.casefold() else 1.0 for document in documents]


def _chunk(chunk_id: str, paper_id: str, content: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "work_id": paper_id,
        "document_id": paper_id,
        "title": f"{paper_id} paper",
        "authors": [],
        "year": 2024,
        "section_path": ["Methods"],
        "page_start": 2,
        "page_end": 2,
        "content": content,
        "parent_contexts": [],
        "overlap_context": None,
        "content_types": ["paragraph"],
        "block_ids": [f"B_{chunk_id}"],
    }


def test_hierarchical_retrieval_scopes_chunk_stage_to_paper_candidates(tmp_path: Path) -> None:
    chunks = [_chunk("C_ALPHA", "P_ALPHA", "alpha method evidence"), _chunk("C_BETA", "P_BETA", "beta method evidence")]
    write_jsonl_atomic(tmp_path / "data" / "knowledge" / "chunks.jsonl", chunks)
    write_bm25_index(tmp_path, build_bm25_index(chunks))
    papers = [
        {"paper_id": "P_ALPHA", "title": "Alpha paper", "abstract": "alpha method", "authors": [], "year": 2024, "keywords": []},
        {"paper_id": "P_BETA", "title": "Beta paper", "abstract": "beta method", "authors": [], "year": 2024, "keywords": []},
    ]
    write_paper_records(tmp_path, papers)
    build_paper_bm25_index(tmp_path, papers)
    build_paper_dense_index(tmp_path, FakeEmbedding())
    build_dense_index(tmp_path, FakeEmbedding())

    result = search_hierarchical_evidence(
        tmp_path,
        FakeEmbedding(),
        "alpha method",
        reranker_provider=FakeReranker(),
        limit=2,
        paper_limit=1,
        max_chunks_per_work=2,
    )

    assert result["candidate_paper_ids"] == ["P_ALPHA"]
    assert [item["paper_id"] for item in result["results"]] == ["P_ALPHA"]
    assert result["chunk_retrieval"]["allowed_paper_ids"] == ["P_ALPHA"]
    assert result["results"][0]["section_path"] == ["Methods"]
