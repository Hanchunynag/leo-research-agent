from __future__ import annotations

from pathlib import Path

from app.contracts import CandidateEvidence
from app.evaluation.acceptance import CutoverAcceptanceConfig, evaluate_cutover
from app.evaluation.shadow import compare_retrievals


ROOT = Path(__file__).resolve().parents[1]


def config() -> CutoverAcceptanceConfig:
    return CutoverAcceptanceConfig.load(
        ROOT / "tests" / "baselines" / "lightrag_cutover_acceptance.json"
    )


def passing_report() -> dict[str, object]:
    return {
        "dataset": {
            "retrieval_question_count": 21,
            "confirmed_relation_question_count": 8,
        },
        "quality": {
            "legacy_direct_ndcg_at_10": 0.8,
            "lightrag_direct_ndcg_at_10": 0.76,
            "relation_path_precision": 0.8,
            "legacy_relation_coverage": 0.7,
            "lightrag_relation_coverage": 0.7,
        },
        "safety": {
            "workspace_leakage_count": 0,
            "excluded_evidence_leakage_count": 0,
            "source_backfill_rate": 0.95,
        },
        "incremental": {
            "full_rebuild": False,
            "unrelated_reprocessed_document_count": 0,
        },
        "deletion": {"residual_result_count": 0},
        "rollback": {"legacy_top_k_restored": True},
        "diagnostics": {"all_failures_classified": True},
    }


def candidate(
    evidence_id: str,
    chunk_id: str,
    document_id: str,
    *,
    workspace_id: str = "default",
) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id,
        "REQ",
        workspace_id,
        1,
        "content",
        1.0,
        "fixture",
        work_id=document_id,
        document_id=document_id,
        chunk_id=chunk_id,
        metadata={"token_count": 4},
    )


def test_cutover_thresholds_are_centralized_and_complete() -> None:
    result = evaluate_cutover(config(), passing_report())
    assert result["passed"] is True
    assert result["official_cutover_approved"] is True
    assert all(result["checks"].values())


def test_missing_metrics_fail_closed_with_actionable_categories() -> None:
    result = evaluate_cutover(config(), {})
    assert result["passed"] is False
    assert result["official_cutover_approved"] is False
    assert {value["category"] for value in result["failures"]} >= {
        "dataset",
        "mapping",
        "reranking",
        "relation",
        "indexing",
        "deletion",
        "rollback",
        "diagnostics",
    }


def test_shadow_comparison_records_quality_leakage_and_distribution() -> None:
    official = [candidate("L1", "C1", "D1"), candidate("L2", "C2", "D2")]
    shadow = [
        candidate("S1", "C2", "D2"),
        candidate("S2", "C1", "D1"),
        candidate("S3", "CX", "DX"),
    ]
    result = compare_retrievals(
        "query",
        official,
        shadow,
        k=10,
        relevant_chunk_ids={"C1"},
        relevant_document_ids={"D1"},
        excluded_document_ids={"DX"},
    )
    assert result["official"]["mrr"] == 1.0
    assert result["shadow"]["mrr"] == 0.5
    assert result["shadow"]["document_hit_rate"] == 1.0
    assert result["shadow"]["excluded_evidence_leakage_count"] == 1
    assert result["shadow"]["selection_distribution"]["document_count"] == 3
