"""Qdrant local 上的 BGE-M3 单向量 Dense Retrieval。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.indexing.bm25 import chunks_digest
from app.indexing.dense import (
    VECTOR_NAME,
    dense_index_path,
    load_dense_manifest,
)
from app.retrieval.search import load_chunks


def _validate_limit(value: int, field: str, maximum: int = 100) -> int:
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{field} 必须在 1 到 {maximum} 之间。")
    return value


def _query_filter(
    work_id: str | None,
    document_id: str | None,
) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if work_id:
        must.append(
            models.FieldCondition(
                key="work_id",
                match=models.MatchValue(value=work_id),
            )
        )
    if document_id:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )
    return models.Filter(must=cast(Any, must)) if must else None


def search_dense_evidence(
    project_root: Path,
    provider: EmbeddingProvider,
    query: str,
    limit: int = 10,
    work_id: str | None = None,
    document_id: str | None = None,
    max_chunks_per_work: int = 2,
) -> dict[str, Any]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空。")
    validated_limit = _validate_limit(limit, "limit")
    per_work = _validate_limit(max_chunks_per_work, "max_chunks_per_work", 20)
    root = project_root.expanduser().resolve()
    chunks = load_chunks(root)
    manifest = load_dense_manifest(root)
    current_digest = chunks_digest(chunks)
    if manifest.get("chunks_digest") != current_digest:
        raise RuntimeError("Dense manifest 与 chunks.jsonl 不一致，请重新构建索引。")
    model_name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    if (
        manifest.get("model_name") != model_name
        or manifest.get("model_revision") != revision
    ):
        raise RuntimeError("Dense manifest 与当前 EmbeddingProvider 不一致。")

    vector = provider.embed_query(cleaned_query)
    expected_dimension = int(manifest.get("vector_dimension", 0))
    if len(vector) != expected_dimension:
        raise RuntimeError("查询向量维度与 Dense manifest 不一致。")
    candidate_limit = min(max(validated_limit * 5, validated_limit), 100)
    client = QdrantClient(path=str(dense_index_path(root)))
    try:
        response = client.query_points(
            collection_name=str(manifest.get("collection_name")),
            query=vector,
            using=VECTOR_NAME,
            query_filter=_query_filter(work_id, document_id),
            limit=candidate_limit,
            with_payload=True,
            with_vectors=False,
        )
    finally:
        client.close()

    work_counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        result_work_id = str(payload.get("work_id") or "")
        if work_counts[result_work_id] >= per_work:
            continue
        work_counts[result_work_id] += 1
        results.append(
            {
                "rank": len(results) + 1,
                "score": round(float(point.score), 6),
                "retrieval_source": "dense",
                "chunk_id": payload.get("chunk_id"),
                "work_id": payload.get("work_id"),
                "document_id": payload.get("document_id"),
                "paper_id": payload.get("paper_id"),
                "title": payload.get("title"),
                "authors": payload.get("authors"),
                "year": payload.get("year"),
                "doi": payload.get("doi"),
                "section_path": payload.get("section_path"),
                "content_zone": payload.get("content_zone"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "block_ids": payload.get("block_ids"),
                "content_types": payload.get("content_types"),
                "parent_contexts": payload.get("parent_contexts") or [],
                "overlap_context": payload.get("overlap_context"),
                "content": payload.get("content"),
                "citation": (
                    f"{payload.get('document_id')} pp. "
                    f"{payload.get('page_start')}-{payload.get('page_end')}"
                ),
            }
        )
        if len(results) >= validated_limit:
            break
    return {
        "query": cleaned_query,
        "retriever": "dense",
        "model_name": model_name,
        "result_count": len(results),
        "work_id_filter": work_id,
        "document_id_filter": document_id,
        "max_chunks_per_work": per_work,
        "results": results,
    }
