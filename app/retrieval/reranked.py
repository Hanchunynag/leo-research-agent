"""对冻结的 RRF 候选池执行 Cross-Encoder 精排。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from app.embeddings.base import EmbeddingProvider
from app.indexing.dense import dense_chunk_text
from app.reranking.base import RerankerProvider
from app.retrieval.hybrid import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_RRF_K,
    _positive_integer,
    search_hybrid_evidence,
)


RERANKER_TEXT_POLICY_VERSION = "1.0"


def reranker_document_text(candidate: dict[str, Any]) -> str:
    """沿用 Dense 的来源感知文本策略，避免创建第二套隐式 Chunk。"""

    return dense_chunk_text(candidate)


def search_reranked_evidence(
    project_root: Path,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
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

    total_started = perf_counter()
    candidate_started = perf_counter()
    hybrid = search_hybrid_evidence(
        project_root=project_root,
        provider=embedding_provider,
        query=cleaned_query,
        limit=candidates_per_source,
        work_id=work_id,
        document_id=document_id,
        max_chunks_per_work=20,
        candidate_limit=candidates_per_source,
        rrf_k=rrf_k,
    )
    candidate_retrieval_ms = (perf_counter() - candidate_started) * 1000
    raw_candidates = hybrid.get("results")
    candidates = (
        [value for value in raw_candidates if isinstance(value, dict)]
        if isinstance(raw_candidates, list)
        else []
    )
    documents = [reranker_document_text(candidate) for candidate in candidates]
    reranking_started = perf_counter()
    scores = reranker_provider.score(cleaned_query, documents)
    reranking_seconds = perf_counter() - reranking_started
    if len(scores) != len(candidates):
        raise RuntimeError("RerankerProvider 返回的分数数量与 RRF 候选不一致。")

    reranked: list[dict[str, Any]] = []
    for candidate, reranker_score in zip(candidates, scores, strict=True):
        result = dict(candidate)
        final_score = round(float(reranker_score), 6)
        result.update(
            {
                "rrf_rank": candidate.get("rank"),
                "rrf_score": candidate.get("score"),
                "score": final_score,
                "reranker_score": final_score,
                "retrieval_source": "hybrid_rrf_reranked",
            }
        )
        reranked.append(result)
    reranked.sort(
        key=lambda value: (
            -float(value["reranker_score"]),
            int(value.get("rrf_rank") or 10**9),
            str(value.get("chunk_id") or ""),
        )
    )

    work_counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []
    for candidate in reranked:
        result_work_id = str(candidate.get("work_id") or "")
        if work_counts[result_work_id] >= per_work:
            continue
        work_counts[result_work_id] += 1
        result = dict(candidate)
        result["rank"] = len(results) + 1
        results.append(result)
        if len(results) >= output_limit:
            break

    total_ms = (perf_counter() - total_started) * 1000
    reranking_ms = reranking_seconds * 1000
    return {
        "query": cleaned_query,
        "retriever": "hybrid_rrf_reranked",
        "result_count": len(results),
        "candidate_count": len(candidates),
        "candidate_limit_per_source": candidates_per_source,
        "rrf_k": hybrid.get("rrf_k"),
        "reranker_model": getattr(reranker_provider, "model_name", None),
        "reranker_revision": getattr(reranker_provider, "revision", None),
        "reranker_text_policy_version": RERANKER_TEXT_POLICY_VERSION,
        "work_id_filter": work_id,
        "document_id_filter": document_id,
        "max_chunks_per_work": per_work,
        "timing": {
            "candidate_retrieval_ms": round(candidate_retrieval_ms, 3),
            "reranking_ms": round(reranking_ms, 3),
            "total_ms": round(total_ms, 3),
            "pairs_per_second": round(
                len(candidates) / reranking_seconds if reranking_seconds else 0.0,
                3,
            ),
        },
        "results": results,
    }
