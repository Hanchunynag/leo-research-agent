from __future__ import annotations

from pathlib import Path

from app.evaluation.acceptance import CutoverAcceptanceConfig, evaluate_cutover
from app.evaluation.stage4 import load_relation_questions


ROOT = Path(__file__).resolve().parents[1]


def test_pending_relation_qrels_hard_block_official_cutover() -> None:
    questions = load_relation_questions(
        ROOT / "data" / "evaluation" / "relation_questions.jsonl"
    )
    confirmed = sum(value.annotation_status == "confirmed" for value in questions)
    config = CutoverAcceptanceConfig.load(
        ROOT / "tests" / "baselines" / "lightrag_cutover_acceptance.json"
    )
    result = evaluate_cutover(
        config,
        {
            "dataset": {
                "retrieval_question_count": 21,
                "confirmed_relation_question_count": confirmed,
            }
        },
    )
    assert confirmed == 0
    assert result["checks"]["relation_dataset_confirmed"] is False
    assert result["official_cutover_approved"] is False
