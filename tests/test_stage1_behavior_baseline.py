from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.graph.models import EvidenceCandidate
from app.graph.retrieval import GraphRetriever
from app.web.jobs import JobManager


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "baselines" / "stage1_behavior_baseline.json"


def test_stage1_baseline_has_all_required_behaviors_and_top_k_locators() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["corpus"]["chunk_count"] == 135
    inventory = {item["id"]: item for item in payload["behavior_inventory"]}
    assert set(inventory) == {
        "pdf_parse",
        "canonical_chunk",
        "bm25",
        "dense",
        "graph_relation",
        "rrf_reranker",
        "ordinary_qa",
        "relationship_qa",
        "citation_mapping",
        "add_paper",
        "delete_paper",
        "job_state_after_restart",
    }
    top_k = payload["retrieval_evaluation"]["representative_top_10"]
    for retriever in ("bm25", "dense", "rrf", "reranker"):
        assert len(top_k[retriever]) == 10
        assert all(chunk_id.startswith("D_") for chunk_id in top_k[retriever])
    assert top_k["relevant_document_ids"] == ["D_060e764f208c"]
    assert payload["usage"]["llm_call_count"] is None
    assert inventory["delete_paper"]["status"] == "unsupported_current_behavior"


class BranchOnlyGraphRetriever(GraphRetriever):
    def __init__(
        self,
        direct: list[EvidenceCandidate],
        paths: list[EvidenceCandidate],
    ) -> None:
        self._direct = direct
        self._paths = paths

    def direct_relation(
        self, entity_a: str, entity_b: str, epoch: int, limit: int = 20
    ) -> list[EvidenceCandidate]:
        return self._direct[:limit]

    def two_hop_paths(
        self, entity_a: str, entity_b: str, epoch: int, limit: int = 20
    ) -> list[EvidenceCandidate]:
        return self._paths[:limit]


def graph_candidate(candidate_type: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id="REL-1",
        candidate_type=candidate_type,  # type: ignore[arg-type]
        text="A sourced relationship.",
        source_chunk_keys=["CK-1"],
        source_claim_ids=["RC-1"],
    )


def test_graph_relationship_baseline_prefers_direct_then_disclaims_paths() -> None:
    direct = BranchOnlyGraphRetriever([graph_candidate("relation_claim")], [])
    path = BranchOnlyGraphRetriever([], [graph_candidate("graph_path")])
    missing = BranchOnlyGraphRetriever([], [])

    assert direct.relationship_search("A", "B", 1)["mode"] == "direct"
    inferred = path.relationship_search("A", "B", 1)
    assert inferred["mode"] == "inferred_path"
    assert "间接" not in inferred["disclaimer"]
    assert "没有直接" in inferred["disclaimer"]
    none = missing.relationship_search("A", "B", 1)
    assert none["mode"] == "none"
    assert none["candidates"] == []
    assert "没有检索到" in none["refusal_reason"]


def test_web_job_state_is_not_persisted_across_manager_restart() -> None:
    first = JobManager(max_workers=1)
    created = first.submit("answer", lambda emit: {"answerable": True})
    first.close()
    assert first.snapshot(created.job_id).status == "succeeded"

    restarted = JobManager(max_workers=1)
    try:
        with pytest.raises(KeyError, match="任务不存在"):
            restarted.snapshot(created.job_id)
    finally:
        restarted.close()
