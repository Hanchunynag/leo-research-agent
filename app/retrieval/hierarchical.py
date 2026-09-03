"""Two-stage hierarchical retrieval built on the existing retrieval primitives."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from app.embeddings.base import EmbeddingProvider
from app.reranking.base import RerankerProvider
from app.retrieval.dense import search_dense_evidence
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.paper import search_papers_hybrid
from app.retrieval.reranked import reranker_document_text
from app.retrieval.search import search_evidence


def _positive(value: int, name: str, maximum: int = 100) -> int:
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{name} 必须在 1 到 {maximum} 之间。")
    return value


def _rerank(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: RerankerProvider | None,
    limit: int,
    max_chunks_per_work: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if reranker is None:
        ranked = candidates
    else:
        scores = reranker.score(query, [reranker_document_text(value) for value in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError("RerankerProvider 返回的分数数量与候选不一致。")
        ranked = []
        for candidate, score in zip(candidates, scores, strict=True):
            ranked.append({
                **candidate,
                "rrf_rank": candidate.get("rank"),
                "rrf_score": candidate.get("score"),
                "score": round(float(score), 6),
                "reranker_score": round(float(score), 6),
                "retrieval_source": "hierarchical_rrf_reranked",
            })
        ranked.sort(key=lambda value: (
            -float(value.get("reranker_score", 0.0)),
            int(value.get("rrf_rank") or 10**9),
            str(value.get("chunk_id") or ""),
        ))

    work_counts: defaultdict[str, int] = defaultdict(int)
    output: list[dict[str, Any]] = []
    for candidate in ranked:
        work_key = str(candidate.get("work_id") or candidate.get("paper_id") or "")
        if work_counts[work_key] >= max_chunks_per_work:
            continue
        work_counts[work_key] += 1
        output.append({**candidate, "rank": len(output) + 1})
        if len(output) >= limit:
            break
    return output


def search_hierarchical_evidence(
    project_root: Path,
    embedding_provider: EmbeddingProvider,
    query: str,
    *,
    reranker_provider: RerankerProvider | None = None,
    limit: int = 10,
    paper_limit: int = 10,
    paper_candidate_limit: int = 30,
    chunk_candidate_limit: int = 40,
    max_chunks_per_work: int = 2,
    rrf_k: int = 60,
    paper_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query 不能为空。")
    output_limit = _positive(limit, "limit")
    paper_output_limit = _positive(paper_limit, "paper_limit")
    paper_candidates = _positive(paper_candidate_limit, "paper_candidate_limit")
    chunk_candidates = _positive(chunk_candidate_limit, "chunk_candidate_limit")
    per_work = _positive(max_chunks_per_work, "max_chunks_per_work", 20)
    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于 0。")

    started = perf_counter()
    paper_stage = search_papers_hybrid(
        project_root,
        embedding_provider,
        cleaned,
        limit=paper_output_limit,
        candidate_limit=paper_candidates,
        rrf_k=rrf_k,
        filters=paper_filters,
    )
    candidate_papers = [value for value in paper_stage.get("results", []) if isinstance(value, dict)]
    candidate_paper_ids = [
        str(value.get("paper_id"))
        for value in candidate_papers
        if value.get("paper_id")
    ]
    if not candidate_paper_ids:
        return {
            "query": cleaned,
            "retriever": "hierarchical",
            "result_count": 0,
            "candidate_papers": [],
            "candidate_paper_ids": [],
            "results": [],
            "coverage": {"paper_count": 0, "evidence_count": 0},
        }

    # The paper_id restriction is applied inside both underlying indexes.  It
    # is intentionally not implemented as post-filtering after global Top-K.
    bm25 = search_evidence(
        project_root,
        cleaned,
        limit=chunk_candidates,
        max_chunks_per_work=20,
        paper_ids=candidate_paper_ids,
    )
    dense = search_dense_evidence(
        project_root,
        embedding_provider,
        cleaned,
        limit=chunk_candidates,
        max_chunks_per_work=20,
        paper_ids=candidate_paper_ids,
    )
    fused = reciprocal_rank_fusion(
        {"bm25": bm25.get("results", []), "dense": dense.get("results", [])},
        rrf_k=rrf_k,
        limit=chunk_candidates,
    )
    results = _rerank(cleaned, fused, reranker_provider, output_limit, per_work)
    return {
        "query": cleaned,
        "retriever": "hierarchical_rrf_reranked" if reranker_provider else "hierarchical_rrf",
        "result_count": len(results),
        "candidate_papers": candidate_papers,
        "candidate_paper_ids": candidate_paper_ids,
        "paper_retrieval": paper_stage,
        "chunk_retrieval": {
            "bm25_count": len(bm25.get("results", [])),
            "dense_count": len(dense.get("results", [])),
            "rrf_count": len(fused),
            "allowed_paper_ids": candidate_paper_ids,
        },
        "reranker_model": getattr(reranker_provider, "model_name", None),
        "rrf_k": rrf_k,
        "results": results,
        "coverage": {
            "paper_count": len({str(value.get("paper_id")) for value in results}),
            "evidence_count": len(results),
        },
        "timing": {"total_ms": round((perf_counter() - started) * 1000, 3)},
    }
