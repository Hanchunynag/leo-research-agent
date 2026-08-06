from __future__ import annotations

import json
from pathlib import Path


def test_stage2_shadow_acceptance_snapshot_is_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "tests" / "baselines" / "stage2_shadow_acceptance.json").read_text(
            encoding="utf-8"
        )
    )

    generation = payload["generation"]
    representative = payload["representative"]
    relationship = payload["relationship"]
    acceptance = payload["acceptance"]

    assert generation["state"] == "active"
    assert generation["document_count"] == 7
    assert generation["chunk_count"] == 135
    assert representative["relevant_chunk_id"] in representative["shadow_top_10"]
    assert representative["shadow_relevant_rank"] == 2
    assert representative["shadow_recall_at_10"] == 1.0
    assert relationship["graph_backfill_rate"] >= acceptance["graph_backfill_threshold"]
    assert relationship["cross_workspace_leakage_count"] == 0
    assert acceptance["passed"] is True
    assert acceptance["official_cutover_approved"] is False
