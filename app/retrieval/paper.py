"""Paper-level BM25, Dense and RRF retrieval for hierarchical RAG."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, cast

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.indexing.paper import paper_bm25_index_path, paper_retrieval_text, load_paper_records, papers_digest
from app.indexing.paper_dense import (
    PAPER_VECTOR_NAME,
    load_paper_dense_manifest,
    paper_dense_index_path,
)
from app.indexing.tokenization import normalize_search_text, tokenize


def _validate_limit(value: int, field: str, maximum: int = 100) -> int:
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{field} 必须在 1 到 {maximum} 之间。")
    return value


def _paper_matches_filters(paper: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    filters = filters or {}
    year = paper.get("year")
    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    if year_from is not None and isinstance(year, int) and year < int(year_from):
        return False
    if year_to is not None and isinstance(year, int) and year > int(year_to):
        return False
    allowed_ids = filters.get("paper_ids")
    if allowed_ids and str(paper.get("paper_id")) not in {str(value) for value in allowed_ids}:
        return False
    author = str(filters.get("author") or "").strip().casefold()
    if author and not any(author in str(value).casefold() for value in paper.get("authors") or []):
        return False
    keywords = {str(value).casefold() for value in paper.get("keywords") or []}
    requested_keywords = {str(value).casefold() for value in filters.get("keywords") or []}
    return not requested_keywords or bool(keywords & requested_keywords)


def _result(paper: dict[str, Any], rank: int, score: float, source: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": round(float(score), 6),
        "retrieval_source": source,
        "paper_id": paper.get("paper_id"),
        "work_id": paper.get("work_id"),
        "document_id": paper.get("document_id"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "keywords": paper.get("keywords") or [],
        "doi": paper.get("doi"),
        "status": paper.get("status"),
    }


def search_paper_bm25(
    project_root: Path,
    query: str,
    *,
    limit: int = 20,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query 不能为空。")
    output_limit = _validate_limit(limit, "limit")
    path = paper_bm25_index_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    documents = index.get("documents")
    postings = index.get("postings")
    if not isinstance(documents, list) or not isinstance(postings, dict):
        raise ValueError("Paper BM25 索引结构无效。")
    current_papers = load_paper_records(project_root)
    if index.get("papers_digest") != papers_digest(current_papers):
        raise RuntimeError("Paper BM25 索引与 Paper metadata 不一致，请重新构建索引。")
    query_tokens = list(dict.fromkeys(tokenize(cleaned)))
    count = int(index.get("document_count", len(documents)))
    average = float(index.get("average_document_length", 0.0)) or 1.0
    scores: defaultdict[int, float] = defaultdict(float)
    for term in query_tokens:
        raw = postings.get(term)
        if not isinstance(raw, list) or not raw:
            continue
        df = len(raw)
        idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
        for posting in raw:
            if not isinstance(posting, list) or len(posting) != 2:
                continue
            index_value, frequency = posting
            if not isinstance(index_value, int) or not isinstance(frequency, int) or index_value >= len(documents):
                continue
            paper = documents[index_value]
            if not isinstance(paper, dict) or not _paper_matches_filters(paper, filters):
                continue
            length = max(int(paper.get("length", 0)), 1)
            k1, b = 1.5, 0.75
            denominator = frequency + k1 * (1 - b + b * length / average)
            scores[index_value] += idf * frequency * (k1 + 1) / denominator
    normalized = normalize_search_text(cleaned)
    ranked = []
    for index_value, score in scores.items():
        paper = documents[index_value]
        if not isinstance(paper, dict):
            continue
        if normalized and normalized in normalize_search_text(paper_retrieval_text(paper)):
            score *= 1.2
        ranked.append((index_value, score))
    ranked.sort(key=lambda item: (-item[1], str(documents[item[0]].get("paper_id"))))
    results = [_result(documents[index], rank, score, "paper_bm25") for rank, (index, score) in enumerate(ranked[:output_limit], 1)]
    return {"query": cleaned, "retriever": "paper_bm25", "result_count": len(results), "results": results}


def _paper_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    filters = filters or {}
    must: list[models.FieldCondition | models.Filter] = []
    ids = [str(value) for value in filters.get("paper_ids") or [] if str(value)]
    if ids:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchAny(any=ids)))
    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    if year_from is not None:
        must.append(models.FieldCondition(key="year", range=models.Range(gte=int(year_from))))
    if year_to is not None:
        must.append(models.FieldCondition(key="year", range=models.Range(lte=int(year_to))))
    return models.Filter(must=cast(Any, must)) if must else None


def search_paper_dense(
    project_root: Path,
    provider: EmbeddingProvider,
    query: str,
    *,
    limit: int = 20,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query 不能为空。")
    output_limit = _validate_limit(limit, "limit")
    root = project_root.expanduser().resolve()
    manifest = load_paper_dense_manifest(root)
    papers = load_paper_records(root)
    if manifest.get("papers_digest") != papers_digest(papers):
        raise RuntimeError("Paper Dense manifest 与 Paper metadata 不一致，请重新构建索引。")
    if manifest.get("model_name") != getattr(provider, "model_name", None) or manifest.get("model_revision") != getattr(provider, "revision", None):
        raise RuntimeError("Paper Dense manifest 与当前 EmbeddingProvider 不一致。")
    vector = provider.embed_query(cleaned)
    if len(vector) != int(manifest.get("vector_dimension", 0)):
        raise RuntimeError("Paper 查询向量维度不一致。")
    client = QdrantClient(path=str(paper_dense_index_path(root)))
    try:
        response = client.query_points(
            collection_name=str(manifest.get("collection_name")),
            query=vector,
            using=PAPER_VECTOR_NAME,
            query_filter=_paper_filter(filters),
            limit=min(max(output_limit * 3, output_limit), 100),
            with_payload=True,
            with_vectors=False,
        )
    finally:
        client.close()
    results: list[dict[str, Any]] = []
    for rank, point in enumerate(response.points, 1):
        payload = point.payload or {}
        if not _paper_matches_filters(payload, filters):
            continue
        results.append(_result(payload, len(results) + 1, float(point.score), "paper_dense"))
        if len(results) >= output_limit:
            break
    return {"query": cleaned, "retriever": "paper_dense", "result_count": len(results), "results": results}


def _rrf(rankings: dict[str, Sequence[dict[str, Any]]], *, rrf_k: int, limit: int) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    candidates: dict[str, dict[str, Any]] = {}
    for source, ranking in sorted(rankings.items()):
        seen: set[str] = set()
        for fallback, value in enumerate(ranking, 1):
            paper_id = str(value.get("paper_id") or "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            rank = value.get("rank") if isinstance(value.get("rank"), int) else fallback
            scores[paper_id] += 1.0 / (rrf_k + max(rank, 1))
            ranks[paper_id][source] = max(rank, 1)
            candidates.setdefault(paper_id, dict(value))
    ordered = sorted(candidates, key=lambda item: (-scores[item], min(ranks[item].values()), item))[:limit]
    return [
        {
            **candidates[paper_id],
            "rank": rank,
            "score": round(scores[paper_id], 9),
            "rrf_score": round(scores[paper_id], 9),
            "retrieval_source": "paper_hybrid_rrf",
            "source_ranks": ranks[paper_id],
        }
        for rank, paper_id in enumerate(ordered, 1)
    ]


def search_papers_hybrid(
    project_root: Path,
    provider: EmbeddingProvider,
    query: str,
    *,
    limit: int = 10,
    candidate_limit: int = 30,
    rrf_k: int = 60,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_limit = _validate_limit(limit, "limit")
    per_source = _validate_limit(candidate_limit, "candidate_limit")
    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于 0。")
    bm25 = search_paper_bm25(project_root, query, limit=per_source, filters=filters)
    dense = search_paper_dense(project_root, provider, query, limit=per_source, filters=filters)
    results = _rrf({"bm25": bm25["results"], "dense": dense["results"]}, rrf_k=rrf_k, limit=output_limit)
    return {
        "query": query.strip(),
        "retriever": "paper_hybrid_rrf",
        "result_count": len(results),
        "candidate_limit_per_source": per_source,
        "rrf_k": rrf_k,
        "results": results,
        "branch_results": {"bm25": bm25["results"], "dense": dense["results"]},
    }
