"""Agentic 检索、证据复用、语义引用与缓存指标接口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agentic.models import AgenticMetricsRecord


def aggregate_agentic_metrics(
    records: Sequence[AgenticMetricsRecord],
) -> dict[str, Any]:
    """聚合最小可评测指标；大型标注集可在此接口上扩展。"""

    if not records:
        return {"count": 0}
    count = len(records)
    total_evidence = sum(
        item.new_evidence_count + item.reused_evidence_count for item in records
    )
    cache_rates = [
        item.cache_hit_rate for item in records if item.cache_hit_rate is not None
    ]

    def mean_optional(field: str) -> float | None:
        values = [
            float(value)
            for item in records
            if (value := getattr(item, field)) is not None
        ]
        return sum(values) / len(values) if values else None

    first_latencies = [
        item.latency_ms
        for item in records
        if item.first_turn is True and item.latency_ms is not None
    ]
    followup_latencies = [
        item.latency_ms
        for item in records
        if item.first_turn is False and item.latency_ms is not None
    ]
    return {
        "count": count,
        "average_retrieval_rounds": sum(item.retrieval_rounds for item in records)
        / count,
        "average_new_evidence": sum(item.new_evidence_count for item in records) / count,
        "evidence_reuse_rate": (
            sum(item.reused_evidence_count for item in records) / total_evidence
            if total_evidence
            else 0.0
        ),
        "evidence_coverage_rate": sum(item.coverage_sufficient for item in records)
        / count,
        "citation_precision_proxy": (
            sum(item.entailed_claim_count for item in records)
            / max(1, sum(item.total_claim_count for item in records))
        ),
        "average_cache_hit_rate": (
            sum(cache_rates) / len(cache_rates) if cache_rates else None
        ),
        "average_latency_ms": (
            sum(item.latency_ms or 0.0 for item in records) / count
        ),
        "retrieval_recall_at_k": mean_optional("retrieval_recall_at_k"),
        "reranker_recall_at_k": mean_optional("reranker_recall_at_k"),
        "mean_reciprocal_rank": mean_optional("reciprocal_rank"),
        "mean_ndcg_at_k": mean_optional("ndcg_at_k"),
        "citation_precision": mean_optional("citation_precision"),
        "citation_recall": mean_optional("citation_recall"),
        "claim_entailment_accuracy": mean_optional(
            "claim_entailment_accuracy"
        ),
        "answerable_accuracy": mean_optional("answerable_correct"),
        "average_first_turn_latency_ms": (
            sum(first_latencies) / len(first_latencies)
            if first_latencies
            else None
        ),
        "average_followup_latency_ms": (
            sum(followup_latencies) / len(followup_latencies)
            if followup_latencies
            else None
        ),
        "compaction_count": sum(item.compaction_count for item in records),
    }
