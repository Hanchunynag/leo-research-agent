from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import main as cli
from app.evaluation.retrieval import (
    evaluate_candidate_pool_oracle,
    evaluate_reranked,
)
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.dense import build_dense_index, dense_chunk_text
from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
from app.retrieval.reranked import (
    reranker_document_text,
    search_reranked_evidence,
)
from app.storage import write_jsonl_atomic


class FakeEmbeddingProvider:
    model_name = "fixture/dense"
    revision = "dense-revision"
    normalized = True

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "alpha" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


class FakeRerankerProvider:
    model_name = "fixture/reranker"
    revision = "reranker-revision"
    max_length = 256

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [2.0 if "alpha" in document.lower() else 1.0 for document in documents]


def chunk(
    chunk_id: str,
    work_id: str,
    document_id: str,
    block_id: str,
    content: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "work_id": work_id,
        "document_id": document_id,
        "paper_id": document_id.replace("D_", "P_"),
        "title": "Reranker Test Paper",
        "authors": [],
        "year": 2026,
        "doi": None,
        "section_path": ["METHOD"],
        "content_zone": "main_body",
        "page_start": 1,
        "page_end": 1,
        "block_ids": [block_id],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": content,
    }


def prepare_store(root: Path) -> tuple[list[dict[str, Any]], Path]:
    chunks = [
        chunk("C_alpha", "W_alpha", "D_alpha", "B_alpha", "alpha method"),
        chunk("C_beta", "W_beta", "D_beta", "B_beta", "beta method"),
    ]
    write_jsonl_atomic(root / "data" / "knowledge" / "chunks.jsonl", chunks)
    write_bm25_index(root, build_bm25_index(chunks))
    build_dense_index(root, FakeEmbeddingProvider())
    questions = root / "data" / "evaluation" / "questions.jsonl"
    questions.parent.mkdir(parents=True, exist_ok=True)
    questions.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "alpha method",
                "relevant_work_ids": ["W_alpha"],
                "relevant_document_ids": ["D_alpha"],
                "relevant_block_ids": ["B_alpha"],
                "question_type": "method",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return chunks, questions


def test_bge_reranker_adapter_returns_scalar_logits() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

        def predict(
            self,
            pairs: list[tuple[str, str]],
            **kwargs: Any,
        ) -> list[float]:
            self.calls.append((pairs, kwargs))
            return [0.25, -0.5]

    model = FakeModel()
    provider = BGERerankerProvider(
        BGERerankerConfig(batch_size=2, max_length=512, show_progress_bar=False),
        model=model,
    )

    assert provider.score(" query ", ["first", "second"]) == [0.25, -0.5]
    assert model.calls[0][0] == [("query", "first"), ("query", "second")]
    assert model.calls[0][1] == {
        "batch_size": 2,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    with pytest.raises(ValueError, match="query 不能为空"):
        provider.score(" ", ["first"])


def test_reranker_text_reuses_dense_source_aware_policy() -> None:
    value = chunk("C1", "W1", "D1", "B1", "primary content")
    value["parent_contexts"] = [{"section_path": ["PARENT"], "content": "bridge"}]

    assert reranker_document_text(value) == dense_chunk_text(value)
    assert "Parent section: PARENT" in reranker_document_text(value)


def test_reranked_search_preserves_rrf_provenance_and_records_timing(
    tmp_path: Path,
) -> None:
    prepare_store(tmp_path)

    result = search_reranked_evidence(
        tmp_path,
        FakeEmbeddingProvider(),
        FakeRerankerProvider(),
        "alpha method",
        limit=2,
        max_chunks_per_work=20,
    )

    assert result["retriever"] == "hybrid_rrf_reranked"
    assert result["candidate_count"] == 2
    assert result["results"][0]["chunk_id"] == "C_alpha"
    assert result["results"][0]["rrf_rank"] == 1
    assert result["results"][0]["score"] == 2.0
    assert result["results"][0]["reranker_score"] == 2.0
    assert result["results"][0]["source_ranks"] == {"bm25": 1, "dense": 1}
    assert result["timing"]["total_ms"] >= result["timing"]["reranking_ms"]


def test_oracle_and_reranker_use_same_qrels_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, questions = prepare_store(tmp_path)
    embedding = FakeEmbeddingProvider()
    reranker = FakeRerankerProvider()

    oracle = evaluate_candidate_pool_oracle(tmp_path, questions, embedding)
    assert oracle["union_top_n_each"]["mean_recall"] == 1.0
    assert oracle["rrf_top_n"]["mean_recall"] == 1.0

    report = evaluate_reranked(
        tmp_path,
        questions,
        embedding,
        reranker,
        k_values=(1, 2),
    )
    assert report["retriever"] == "hybrid_rrf_reranked"
    assert report["metrics"]["recall@1"] == 1.0
    diagnostics = report["per_question"][0]["reranking_diagnostics"]
    assert diagnostics["rrf_candidate_first_relevant_rank"] == 1
    assert diagnostics["reranked_candidate_first_relevant_rank"] == 1
    assert diagnostics["rank_delta"] == 0
    assert report["performance"]["total_pairs"] == 2

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "dense_provider_from_args", lambda args: embedding)
    monkeypatch.setattr(cli, "reranker_provider_from_args", lambda args: reranker)
    output = tmp_path / "data" / "evaluation" / "reranker.json"
    cli.main(
        [
            "evaluate",
            "retrieval",
            "--retriever",
            "reranker",
            "--questions",
            str(questions),
            "--output",
            str(output),
            "--k-values",
            "1,2",
            "--local-files-only",
        ]
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["retriever"] == "hybrid_rrf_reranked"
    assert output.is_file()
