from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import main as cli
from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
from app.evaluation.retrieval import evaluate_dense, evaluate_hybrid_rrf
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.dense import (
    build_dense_index,
    dense_chunk_text,
    load_dense_manifest,
)
from app.retrieval.dense import search_dense_evidence
from app.retrieval.hybrid import reciprocal_rank_fusion, search_hybrid_evidence
from app.storage import write_jsonl_atomic


class FakeEmbeddingProvider:
    model_name = "fixture/dense"
    revision = "0123456789abcdef"
    normalized = True

    def __init__(self) -> None:
        self.document_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


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
        "title": "Dense Test Paper",
        "authors": ["Ada Researcher"],
        "year": 2026,
        "doi": None,
        "section_path": ["RESULTS", "Orbit estimation"],
        "content_zone": "main_body",
        "page_start": 2,
        "page_end": 2,
        "block_ids": [block_id],
        "content_types": ["paragraph"],
        "parent_contexts": [],
        "overlap_context": None,
        "content": content,
    }


def write_chunks(root: Path, values: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(root / "data" / "knowledge" / "chunks.jsonl", values)


def write_questions(root: Path, block_id: str = "B_alpha") -> Path:
    path = root / "data" / "evaluation" / "questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "alpha evidence",
                "relevant_work_ids": ["W_alpha"],
                "relevant_document_ids": ["D_alpha"],
                "relevant_block_ids": [block_id],
                "question_type": "method",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_bge_m3_provider_uses_dense_sentence_transformer_output_only() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict[str, Any]]] = []

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            self.calls.append((texts, kwargs))
            return [[float(index), 2.0] for index, _ in enumerate(texts, 1)]

    model = FakeModel()
    provider = BGEM3EmbeddingProvider(
        BGEM3Config(batch_size=3, show_progress_bar=False),
        model=model,
    )

    assert provider.embed_documents(["first", "second"]) == [
        [1.0, 2.0],
        [2.0, 2.0],
    ]
    assert provider.embed_query(" query ") == [1.0, 2.0]
    assert model.calls[0][1] == {
        "batch_size": 3,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert model.calls[1][0] == ["query"]


def test_dense_text_preserves_explicit_context_and_source_boundaries() -> None:
    value = chunk("C1", "W1", "D1", "B1", "primary alpha")
    value["parent_contexts"] = [
        {
            "section_path": ["METHOD"],
            "content": "parent bridge",
            "block_ids": ["B_parent"],
        }
    ]
    value["overlap_context"] = {
        "content": "same section overlap",
        "block_ids": ["B_previous"],
    }

    text = dense_chunk_text(value)

    assert "Title: Dense Test Paper" in text
    assert "Section: RESULTS > Orbit estimation" in text
    assert "Parent section: METHOD\nparent bridge" in text
    assert "Previous context: same section overlap" in text
    assert text.endswith("Content: primary alpha")


def test_rrf_uses_ranks_instead_of_mixing_raw_score_scales() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [
                {"rank": 1, "score": 100.0, "chunk_id": "A"},
                {"rank": 2, "score": 50.0, "chunk_id": "B"},
            ],
            "dense": [
                {"rank": 1, "score": 0.8, "chunk_id": "B"},
                {"rank": 2, "score": 0.7, "chunk_id": "C"},
            ],
        },
        rrf_k=60,
        limit=3,
    )

    assert [item["chunk_id"] for item in fused] == ["B", "A", "C"]
    assert fused[0]["source_ranks"] == {"bm25": 2, "dense": 1}
    assert fused[0]["source_scores"] == {"bm25": 50.0, "dense": 0.8}
    assert fused[0]["retrieval_source"] == "hybrid_rrf"


def test_qdrant_manifest_reuse_search_filters_and_stale_rejection(
    tmp_path: Path,
) -> None:
    chunks = [
        chunk("C_alpha", "W_alpha", "D_alpha", "B_alpha", "alpha evidence"),
        chunk("C_beta", "W_beta", "D_beta", "B_beta", "beta evidence"),
    ]
    write_chunks(tmp_path, chunks)
    provider = FakeEmbeddingProvider()

    first = build_dense_index(tmp_path, provider)
    second = build_dense_index(tmp_path, provider)

    assert first.status == "built"
    assert first.vector_dimension == 3
    assert second.status == "reused"
    assert second.embedded_count == 0
    assert provider.document_calls == 1
    manifest = load_dense_manifest(tmp_path)
    assert manifest["model_revision"] == provider.revision
    assert manifest["vector_name"] == "dense"
    assert manifest["chunk_count"] == 2

    result = search_dense_evidence(tmp_path, provider, "alpha", limit=2)
    assert result["results"][0]["chunk_id"] == "C_alpha"
    assert result["results"][0]["retrieval_source"] == "dense"
    filtered = search_dense_evidence(
        tmp_path,
        provider,
        "alpha",
        document_id="D_beta",
    )
    assert [item["document_id"] for item in filtered["results"]] == ["D_beta"]

    chunks[0]["content"] = "changed after dense indexing"
    write_chunks(tmp_path, chunks)
    with pytest.raises(RuntimeError, match="manifest.*不一致"):
        search_dense_evidence(tmp_path, provider, "alpha")


def test_hybrid_search_fuses_local_bm25_and_dense(tmp_path: Path) -> None:
    values = [
        chunk("C_alpha", "W_alpha", "D_alpha", "B_alpha", "alpha evidence"),
        chunk("C_beta", "W_beta", "D_beta", "B_beta", "beta evidence"),
    ]
    write_chunks(tmp_path, values)
    write_bm25_index(tmp_path, build_bm25_index(values))
    provider = FakeEmbeddingProvider()
    build_dense_index(tmp_path, provider)

    result = search_hybrid_evidence(
        tmp_path,
        provider,
        "alpha evidence",
        limit=2,
        max_chunks_per_work=20,
    )

    assert result["retriever"] == "hybrid_rrf"
    assert result["candidate_limit_per_source"] == 20
    assert result["results"][0]["chunk_id"] == "C_alpha"
    assert result["results"][0]["source_ranks"] == {"bm25": 1, "dense": 1}


def test_dense_uses_same_evaluator_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = [
        chunk("C_alpha", "W_alpha", "D_alpha", "B_alpha", "alpha evidence"),
        chunk("C_beta", "W_beta", "D_beta", "B_beta", "beta evidence"),
    ]
    write_chunks(tmp_path, values)
    questions = write_questions(tmp_path)
    provider = FakeEmbeddingProvider()
    build_dense_index(tmp_path, provider)
    write_bm25_index(tmp_path, build_bm25_index(values))

    direct = evaluate_dense(tmp_path, questions, provider, k_values=(1, 2))
    assert direct["retriever"] == "dense"
    assert direct["metrics"]["recall@1"] == 1.0
    assert direct["metrics"]["mrr"] == 1.0

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "dense_provider_from_args", lambda args: provider)
    output = tmp_path / "data" / "evaluation" / "dense_report.json"
    cli.main(
        [
            "evaluate",
            "retrieval",
            "--retriever",
            "dense",
            "--questions",
            str(questions),
            "--output",
            str(output),
            "--k-values",
            "1,2",
            "--no-progress",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["retriever"] == "dense"
    assert report["metrics"]["recall@2"] == 1.0
    assert output.is_file()

    hybrid_direct = evaluate_hybrid_rrf(
        tmp_path,
        questions,
        provider,
        k_values=(1, 2),
    )
    assert hybrid_direct["retriever"] == "hybrid_rrf"
    assert hybrid_direct["metrics"]["recall@1"] == 1.0

    hybrid_output = tmp_path / "data" / "evaluation" / "rrf_report.json"
    cli.main(
        [
            "evaluate",
            "retrieval",
            "--retriever",
            "rrf",
            "--questions",
            str(questions),
            "--output",
            str(hybrid_output),
            "--k-values",
            "1,2",
            "--local-files-only",
        ]
    )
    hybrid_report = json.loads(capsys.readouterr().out)
    assert hybrid_report["retriever"] == "hybrid_rrf"
    assert hybrid_output.is_file()
