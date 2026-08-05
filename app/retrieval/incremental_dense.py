"""Active-epoch chunk and community dense retrieval from versioned Qdrant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.index_registry.store import IndexRegistryStore
from app.indexing.incremental_dense import qdrant_index_path
from app.indexing.qdrant_collections import (
    CHUNKS_ALIAS, COMMUNITIES_ALIAS, VECTOR_NAME, versioned_collection_name,
)


def _epoch_filter(epoch: int, extra: list[models.Condition] | None = None) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="valid_from_epoch", range=models.Range(lte=epoch)),
              *(extra or [])],
        must_not=[models.FieldCondition(key="valid_to_epoch", range=models.Range(lte=epoch))],
    )


def _query_points(client: QdrantClient, collection: str, vector: list[float],
                  query_filter: models.Filter, limit: int) -> list[Any]:
    response = client.query_points(collection_name=collection,
        query=vector, using=VECTOR_NAME, query_filter=query_filter, limit=limit,
        with_payload=True, with_vectors=False)
    return list(response.points)


def search_incremental_dense(
    project_root: Path, provider: EmbeddingProvider, query: str, limit: int = 20,
    *, epoch: int | None = None, work_id: str | None = None,
    document_id: str | None = None, client: QdrantClient | None = None,
) -> dict[str, Any]:
    active = epoch or IndexRegistryStore(project_root).active_epoch()
    if active is None:
        raise RuntimeError("no active index epoch")
    model_name = str(getattr(provider, "model_name"))
    revision = getattr(provider, "revision", None)
    collection = versioned_collection_name(CHUNKS_ALIAS, model_name, revision)
    extra: list[models.Condition] = []
    if work_id:
        extra.append(models.FieldCondition(key="work_id", match=models.MatchValue(value=work_id)))
    if document_id:
        extra.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    owned = client is None
    qdrant = client or QdrantClient(path=str(qdrant_index_path(project_root)))
    try:
        points = _query_points(qdrant, collection, provider.embed_query(query),
                               _epoch_filter(active, extra), limit)
    finally:
        if owned:
            qdrant.close()
    results = []
    for rank, point in enumerate(points, 1):
        value = dict(point.payload or {})
        value.update({"rank": rank, "score": float(point.score),
                      "retrieval_source": "dense"})
        results.append(value)
    return {"query": query, "active_epoch": active, "collection_name": collection,
            "result_count": len(results), "results": results}


def search_community_dense(
    project_root: Path, provider: EmbeddingProvider, query: str, limit: int = 10,
    *, epoch: int | None = None, client: QdrantClient | None = None,
) -> dict[str, Any]:
    active = epoch or IndexRegistryStore(project_root).active_epoch()
    if active is None:
        raise RuntimeError("no active index epoch")
    collection = versioned_collection_name(COMMUNITIES_ALIAS,
        str(getattr(provider, "model_name")), getattr(provider, "revision", None))
    owned = client is None
    qdrant = client or QdrantClient(path=str(qdrant_index_path(project_root)))
    try:
        points = _query_points(qdrant, collection, provider.embed_query(query),
                               _epoch_filter(active), limit)
    finally:
        if owned:
            qdrant.close()
    return {"query": query, "active_epoch": active, "collection_name": collection,
            "results": [{**dict(point.payload or {}), "rank": rank,
                         "score": float(point.score), "retrieval_source": "community"}
                        for rank, point in enumerate(points, 1)]}
