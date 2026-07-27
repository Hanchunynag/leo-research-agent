"""BM25 与 Dense 候选的 Reciprocal Rank Fusion。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from app.embeddings.base import EmbeddingProvider
from app.retrieval.dense import search_dense_evidence
from app.retrieval.search import search_evidence


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_LIMIT = 20


def _positive_integer(value: int, field: str, maximum: int = 100) -> int:
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{field} 必须在 1 到 {maximum} 之间。")
    return value


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[dict[str, Any]]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """按来源名次融合，保留每个候选的来源排名和原始分数。"""

    fusion_k = _positive_integer(rrf_k, "rrf_k", 1000)
    output_limit = _positive_integer(limit, "limit")
    scores: defaultdict[str, float] = defaultdict(float)
    source_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    source_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    candidates: dict[str, dict[str, Any]] = {}
    for source, results in sorted(rankings.items()):
        seen: set[str] = set()
        for fallback_rank, result in enumerate(results, 1):
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank_value = result.get("rank")
            rank = (
                rank_value
                if isinstance(rank_value, int) and not isinstance(rank_value, bool)
                else fallback_rank
            )
            if rank < 1:
                rank = fallback_rank
            scores[chunk_id] += 1.0 / (fusion_k + rank)
            source_ranks[chunk_id][source] = rank
            raw_score = result.get("score")
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                source_scores[chunk_id][source] = float(raw_score)
            candidates.setdefault(chunk_id, dict(result))

    ordered_ids = sorted(
        candidates,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(source_ranks[chunk_id].values()),
            chunk_id,
        ),
    )[:output_limit]
    fused: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ordered_ids, 1):
        result = candidates[chunk_id]
        result.update(
            {
                "rank": rank,
                "score": round(scores[chunk_id], 9),
                "retrieval_source": "hybrid_rrf",
                "source_ranks": source_ranks[chunk_id],
                "source_scores": source_scores[chunk_id],
            }
        )
        fused.append(result)
    return fused


def search_hybrid_evidence(
    project_root: Path,
    provider: EmbeddingProvider,
    query: str,
    limit: int = 10,
    work_id: str | None = None,
    document_id: str | None = None,
    max_chunks_per_work: int = 2,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空。")
    output_limit = _positive_integer(limit, "limit")
    per_work = _positive_integer(max_chunks_per_work, "max_chunks_per_work", 20)
    candidates_per_source = _positive_integer(candidate_limit, "candidate_limit")
    fusion_k = _positive_integer(rrf_k, "rrf_k", 1000)
    bm25 = search_evidence(
        project_root=project_root,
        query=cleaned_query,
        limit=candidates_per_source,
        work_id=work_id,
        document_id=document_id,
        max_chunks_per_work=20,
    )
    dense = search_dense_evidence(
        project_root=project_root,
        provider=provider,
        query=cleaned_query,
        limit=candidates_per_source,
        work_id=work_id,
        document_id=document_id,
        max_chunks_per_work=20,
    )
    fused_candidates = reciprocal_rank_fusion(
        {
            "bm25": bm25.get("results", []),
            "dense": dense.get("results", []),
        },
        rrf_k=fusion_k,
        limit=candidates_per_source,
    )

    work_counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []
    for candidate in fused_candidates:
        result_work_id = str(candidate.get("work_id") or "")
        if work_counts[result_work_id] >= per_work:
            continue
        work_counts[result_work_id] += 1
        result = dict(candidate)
        result["rank"] = len(results) + 1
        results.append(result)
        if len(results) >= output_limit:
            break
    return {
        "query": cleaned_query,
        "retriever": "hybrid_rrf",
        "result_count": len(results),
        "candidate_limit_per_source": candidates_per_source,
        "rrf_k": fusion_k,
        "work_id_filter": work_id,
        "document_id_filter": document_id,
        "max_chunks_per_work": per_work,
        "results": results,
    }
