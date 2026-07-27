from __future__ import annotations

import json
from pathlib import Path

import pytest

import main as cli
from app.evaluation.retrieval import (
    RetrievalQuestion,
    evaluate_ranked_retriever,
    load_retrieval_questions,
    relevant_chunk_ids,
)
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.storage import write_jsonl_atomic


def chunk(chunk_id: str, block_id: str, content: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "work_id": "W_test",
        "document_id": "D_test",
        "paper_id": "P_test",
        "title": "Test Paper",
        "section_path": ["RESULTS"],
        "content_zone": "main_body",
        "page_start": 1,
        "page_end": 1,
        "block_ids": [block_id],
        "content_types": ["paragraph"],
        "content": content,
        "parent_contexts": [],
        "overlap_context": None,
    }


def question(question_id: str, block_id: str) -> RetrievalQuestion:
    return RetrievalQuestion(
        question_id=question_id,
        question=f"question {question_id}",
        relevant_work_ids=["W_test"],
        relevant_document_ids=["D_test"],
        relevant_block_ids=[block_id],
        question_type="method",
    )


def test_ranked_metrics_use_most_specific_block_qrels() -> None:
    chunks = [
        chunk("C1", "B1", "alpha"),
        chunk("C2", "B2", "beta"),
        chunk("C3", "B3", "gamma"),
    ]
    questions = [question("Q1", "B1"), question("Q2", "B3")]
    rankings = {
        "question Q1": ["C2", "C1", "C3"],
        "question Q2": ["C3", "C1", "C2"],
    }

    def retrieve(value: str, limit: int) -> list[dict[str, object]]:
        return [{"chunk_id": chunk_id} for chunk_id in rankings[value][:limit]]

    report = evaluate_ranked_retriever(
        questions,
        chunks,
        retrieve,
        retriever_name="fixture",
        k_values=(1, 3),
    )

    assert report["metrics"]["recall@1"] == 0.5
    assert report["metrics"]["recall@3"] == 1.0
    assert report["metrics"]["mrr"] == 0.75
    assert report["metrics"]["ndcg@3"] == pytest.approx(0.815465, abs=1e-6)
    assert report["per_question"][0]["target_level"] == "block"
    assert report["per_question"][0]["first_relevant_rank"] == 2


def test_parent_and_overlap_blocks_are_valid_qrels() -> None:
    value = chunk("C1", "B_primary", "primary")
    value["parent_contexts"] = [{"block_ids": ["B_parent"]}]
    value["overlap_context"] = {"block_ids": ["B_overlap"]}

    assert relevant_chunk_ids(question("Q1", "B_parent"), [value]) == {"C1"}
    assert relevant_chunk_ids(question("Q2", "B_overlap"), [value]) == {"C1"}


def test_question_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    record = {
        "question_id": "Q1",
        "question": "Evidence?",
        "relevant_work_ids": ["W_test"],
        "relevant_document_ids": [],
        "relevant_block_ids": [],
        "question_type": "method",
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="question_id 重复"):
        load_retrieval_questions(path)


def test_bm25_evaluation_cli_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks = [
        chunk("C1", "B1", "satellite clock ephemeris observation"),
        chunk("C2", "B2", "unrelated navigation text"),
    ]
    chunks_path = tmp_path / "data" / "knowledge" / "chunks.jsonl"
    write_jsonl_atomic(chunks_path, chunks)
    write_bm25_index(tmp_path, build_bm25_index(chunks))
    questions_path = tmp_path / "data" / "evaluation" / "questions.jsonl"
    questions_path.parent.mkdir(parents=True)
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "clock ephemeris observation",
                "relevant_work_ids": ["W_test"],
                "relevant_document_ids": ["D_test"],
                "relevant_block_ids": ["B1"],
                "question_type": "observation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "data" / "evaluation" / "baseline.json"
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    cli.main(
        [
            "evaluate",
            "retrieval",
            "--questions",
            str(questions_path),
            "--output",
            str(output),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["retriever"] == "bm25"
    assert report["metrics"]["recall@1"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
    assert output.is_file()
