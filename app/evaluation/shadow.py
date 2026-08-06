"""旧正式引擎与 LightRAG 影子召回的同查询比较。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.contracts import CandidateEvidence, VerifiedEvidence
from app.evaluation.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from app.evaluation.stage4 import selection_distribution
from app.storage import write_json_atomic


def compare_retrievals(
    query: str,
    official: Sequence[CandidateEvidence],
    shadow: Sequence[CandidateEvidence | VerifiedEvidence],
    *,
    k: int,
    relevant_chunk_ids: set[str] | None = None,
    relevant_document_ids: set[str] | None = None,
    excluded_document_ids: set[str] | None = None,
    official_elapsed_ms: float | None = None,
    shadow_elapsed_ms: float | None = None,
    shadow_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_ids = [value.chunk_id for value in official[:k] if value.chunk_id]
    shadow_ids = [value.chunk_id for value in shadow[:k] if value.chunk_id]
    qrels = relevant_chunk_ids or set()
    official_set = set(official_ids)
    shadow_set = set(shadow_ids)
    official_documents = [value.document_id for value in official[:k] if value.document_id]
    shadow_documents = [value.document_id for value in shadow[:k] if value.document_id]
    relation_values = [
        value
        for value in shadow[:k]
        if (
            value.source_type
            if isinstance(value, CandidateEvidence)
            else value.metadata.get("retrieval_source")
        )
        in {"lightrag_relation", "lightrag_entity"}
    ]
    leakage = [value.evidence_id for value in shadow if value.workspace_id != official[0].workspace_id] if official else []
    excluded = {
        value.evidence_id
        for value in shadow
        if value.document_id in (excluded_document_ids or set())
    }
    documents = relevant_document_ids or set()
    official_quality = {
        "recall_at_5": recall_at_k(official_ids, qrels, 5) if qrels else None,
        "recall_at_10": recall_at_k(official_ids, qrels, 10) if qrels else None,
        "mrr": reciprocal_rank(official_ids, qrels) if qrels else None,
        "ndcg_at_10": ndcg_at_k(official_ids, qrels, 10) if qrels else None,
        "document_hit_rate": len(set(official_documents) & documents) / len(documents)
        if documents
        else None,
        "chunk_hit_rate": len(set(official_ids) & qrels) / len(qrels) if qrels else None,
    }
    shadow_quality = {
        "recall_at_5": recall_at_k(shadow_ids, qrels, 5) if qrels else None,
        "recall_at_10": recall_at_k(shadow_ids, qrels, 10) if qrels else None,
        "mrr": reciprocal_rank(shadow_ids, qrels) if qrels else None,
        "ndcg_at_10": ndcg_at_k(shadow_ids, qrels, 10) if qrels else None,
        "document_hit_rate": len(set(shadow_documents) & documents) / len(documents)
        if documents
        else None,
        "chunk_hit_rate": len(set(shadow_ids) & qrels) / len(qrels) if qrels else None,
    }
    return {
        "schema_version": "1.0",
        "query": query,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "ground_truth_status": "provided" if relevant_chunk_ids is not None else "pending_annotation",
        "official": {
            "chunk_ids": official_ids,
            "recall_at_k": recall_at_k(official_ids, qrels, k) if qrels else None,
            "ndcg_at_k": ndcg_at_k(official_ids, qrels, k) if qrels else None,
            "query_elapsed_ms": official_elapsed_ms,
            "query_tokens": sum(int(value.metadata.get("token_count") or 0) for value in official[:k]),
            **official_quality,
        },
        "shadow": {
            "chunk_ids": shadow_ids,
            "recall_at_k": recall_at_k(shadow_ids, qrels, k) if qrels else None,
            "ndcg_at_k": ndcg_at_k(shadow_ids, qrels, k) if qrels else None,
            "query_elapsed_ms": shadow_elapsed_ms,
            "query_tokens": sum(int(value.metadata.get("token_count") or 0) for value in shadow[:k]),
            "relation_coverage": len(relation_values),
            "graph_backfill_rate": (shadow_diagnostics or {}).get("graph_backfill_rate"),
            "cross_workspace_leakage_count": len(leakage),
            "excluded_evidence_leakage_count": len(excluded),
            "excluded_evidence_ids": sorted(excluded),
            "selection_distribution": selection_distribution(
                [
                    {
                        "document_id": value.document_id,
                        "token_count": value.metadata.get("token_count", 0),
                    }
                    for value in shadow[:k]
                ]
            ),
            **shadow_quality,
        },
        "comparison": {
            "top_k_overlap": len(official_set & shadow_set) / max(1, k),
            "official_only": sorted(official_set - shadow_set),
            "shadow_only": sorted(shadow_set - official_set),
        },
    }


class ShadowReportStore:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.expanduser().resolve() / "data" / "evaluation" / "lightrag_shadow"

    def write(self, request_id: str, report: dict[str, Any]) -> Path:
        path = self.root / f"{request_id}.json"
        write_json_atomic(path, report)
        return path
