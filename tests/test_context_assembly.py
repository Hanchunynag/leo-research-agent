from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import main as cli
from app.context.assembly import assemble_context_bundle
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.dense import build_dense_index
from app.runtime.retrieval import RetrievalRuntime
from app.storage import write_jsonl_atomic


class FakeEmbeddingProvider:
    model_name = "fixture/dense"
    revision = "dense-revision"
    normalized = True

    def __init__(self) -> None:
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return self._vector(query)


class FakeRerankerProvider:
    model_name = "fixture/reranker"
    revision = "reranker-revision"
    max_length = 256

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        return [2.0 if "alpha" in document.lower() else 1.0 for document in documents]


def candidate(
    chunk_id: str,
    work_id: str,
    document_id: str,
    content: str,
    *,
    rank: int = 1,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": 0.9,
        "retrieval_source": "hybrid_rrf",
        "chunk_id": chunk_id,
        "work_id": work_id,
        "document_id": document_id,
        "paper_id": document_id.replace("D_", "P_"),
        "title": f"Paper {work_id}",
        "authors": ["Ada Researcher"],
        "year": 2026,
        "doi": None,
        "section_path": ["METHOD", "Tracking"],
        "content_zone": "main_body",
        "page_start": 2,
        "page_end": 2,
        "block_ids": [f"{chunk_id}_primary"],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": content,
    }


def prepare_store(root: Path) -> tuple[FakeEmbeddingProvider, FakeRerankerProvider]:
    chunks = [
        candidate("C_alpha", "W_alpha", "D_alpha", "alpha method evidence"),
        candidate("C_beta", "W_beta", "D_beta", "beta method evidence", rank=2),
    ]
    write_jsonl_atomic(root / "data" / "knowledge" / "chunks.jsonl", chunks)
    write_bm25_index(root, build_bm25_index(chunks))
    embedding = FakeEmbeddingProvider()
    build_dense_index(root, embedding)
    return embedding, FakeRerankerProvider()


def test_context_bundle_deduplicates_versions_and_preserves_source_boundaries() -> (
    None
):
    first = candidate("C1", "W1", "D1", "primary alpha evidence")
    first["parent_contexts"] = [
        {
            "work_id": "W1",
            "document_id": "D1",
            "section_path": ["METHOD"],
            "page_start": 1,
            "page_end": 1,
            "block_ids": ["B_parent"],
            "content": "shared parent bridge",
        }
    ]
    first["overlap_context"] = {
        "work_id": "W_foreign",
        "document_id": "D_foreign",
        "block_ids": ["B_foreign"],
        "content": "foreign context must not cross the boundary",
    }
    duplicate_chunk = dict(first)
    different_pdf_version = candidate(
        "C1_version",
        "W1",
        "D1_other",
        "different extraction from duplicate PDF version",
        rank=2,
    )
    duplicate_content = candidate(
        "C_duplicate",
        "W_other",
        "D_other",
        "primary alpha evidence",
        rank=3,
    )
    second = candidate("C2", "W2", "D2", "independent beta evidence", rank=4)
    second["parent_contexts"] = [
        {
            "work_id": "W2",
            "document_id": "D2",
            "block_ids": ["B_repeated_parent"],
            "content": "shared parent bridge",
        }
    ]

    bundle = assemble_context_bundle(
        "tracking evidence",
        "fast",
        [first, duplicate_chunk, different_pdf_version, duplicate_content, second],
        token_budget=1000,
    )

    assert [item.source_id for item in bundle.evidence] == ["S1", "S2"]
    assert [item.chunk_id for item in bundle.evidence] == ["C1", "C2"]
    assert bundle.evidence[0].block_ids == ["C1_primary", "B_parent"]
    assert bundle.evidence[0].page_start == 1
    assert "foreign context" not in bundle.context_text
    assert bundle.context_text.count("shared parent bridge") == 1
    assert "[S1]" in bundle.context_text and "Document ID: D1" in bundle.context_text
    reasons = bundle.diagnostics["skipped_candidate_reasons"]
    assert reasons == {
        "duplicate_chunk": 1,
        "duplicate_content": 1,
        "duplicate_work_document_version": 1,
    }
    assert bundle.diagnostics["omitted_context_count"] == 2


def test_context_bundle_truncates_only_within_one_evidence_boundary() -> None:
    long_content = " ".join(f"term{index}" for index in range(300))
    first = candidate("C_long", "W1", "D1", long_content)
    second = candidate("C_later", "W2", "D2", "later evidence", rank=2)

    bundle = assemble_context_bundle(
        "bounded context",
        "fast",
        [first, second],
        token_budget=120,
    )

    assert bundle.token_count <= 120
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].truncated is True
    assert bundle.evidence[0].content.endswith("…")
    assert "C_later" not in bundle.context_text


def test_retrieval_runtime_reuses_provider_instances_for_fast_and_accurate(
    tmp_path: Path,
) -> None:
    embedding, reranker = prepare_store(tmp_path)
    runtime = RetrievalRuntime(tmp_path, embedding, reranker)

    first_warmup = runtime.warmup()
    second_warmup = runtime.warmup()
    fast = runtime.build_context(
        "alpha method",
        mode="fast",
        token_budget=1000,
    )
    accurate = runtime.build_context(
        "alpha method",
        mode="accurate",
        token_budget=1000,
    )

    assert first_warmup["embedding"]["status"] == "warmed"
    assert first_warmup["reranker"]["status"] == "warmed"
    assert second_warmup["embedding"]["status"] == "already_warm"
    assert second_warmup["reranker"]["status"] == "already_warm"
    assert fast.retrieval_mode == "fast"
    assert fast.evidence[0].chunk_id == "C_alpha"
    assert fast.diagnostics["retrieval"]["retriever"] == "hybrid_rrf"
    assert accurate.retrieval_mode == "accurate"
    assert accurate.evidence[0].chunk_id == "C_alpha"
    assert (
        accurate.diagnostics["retrieval"]["retriever"]
        == "hybrid_rrf_reranked"
    )
    assert embedding.query_calls == 3
    assert reranker.calls == 2


def test_context_cli_outputs_stable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    embedding, reranker = prepare_store(tmp_path)
    runtime = RetrievalRuntime(tmp_path, embedding, reranker)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "retrieval_runtime_from_args",
        lambda args, include_reranker: runtime,
    )

    cli.main(
        [
            "context",
            "build",
            "alpha method",
            "--mode",
            "accurate",
            "--token-budget",
            "1000",
            "--local-files-only",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "1.0"
    assert output["retrieval_mode"] == "accurate"
    assert output["evidence"][0]["source_id"] == "S1"
    assert "Block IDs:" in output["context_text"]
