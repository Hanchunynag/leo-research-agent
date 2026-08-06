from __future__ import annotations

from pathlib import Path

from app.evaluation.retrieval import load_retrieval_questions
from app.evaluation.stage4 import (
    load_relation_questions,
    relation_quality,
    retrieval_quality,
)
from app.retrieval.search import load_chunks


ROOT = Path(__file__).resolve().parents[1]


def test_stage4_retrieval_dataset_has_21_traceable_workspace_qrels() -> None:
    questions = load_retrieval_questions(
        ROOT / "data" / "evaluation" / "retrieval_questions.jsonl"
    )
    chunk_ids = {
        str(value["chunk_id"])
        for value in load_chunks(ROOT)
        if value.get("chunk_id")
    }
    assert len(questions) >= 21
    assert len({value.question_id for value in questions}) == len(questions)
    for value in questions:
        assert value.workspace_id == "default"
        assert value.relevant_document_ids
        assert value.relevant_chunk_ids
        assert set(value.relevant_chunk_ids) <= chunk_ids
        assert set(value.relevant_chunk_ids) <= set(value.acceptable_evidence_ids)
        assert isinstance(value.excluded_document_ids, list)
        assert "manually curated" in value.notes


def test_relation_dataset_is_explicitly_blocked_pending_human_qrels() -> None:
    questions = load_relation_questions(
        ROOT / "data" / "evaluation" / "relation_questions.jsonl"
    )
    assert len(questions) >= 8
    assert {value.question_type for value in questions} == {
        "method_to_result",
        "observation_to_observability",
        "error_propagation",
        "adaptive_robust_coupling",
        "conditional_conflict",
        "ephemeris_to_positioning",
        "cross_paper_support_counterexample",
        "cross_direction_workspace_relation",
    }
    assert all(
        value.annotation_status == "pending_human_confirmation"
        for value in questions
    )
    pending = relation_quality(
        questions[0],
        predicted_paths=[("A", "B")],
        selected_evidence_ids=["E1"],
    )
    assert pending["relation_path_precision"] is None
    assert pending["relation_path_coverage"] is None


def test_stage4_ranked_metrics_record_recall_mrr_and_ndcg() -> None:
    metrics = retrieval_quality(["C2", "C1", "C3"], {"C1"})
    assert metrics == {
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 0.5,
        "ndcg_at_10": 1 / 1.584962500721156,
    }
