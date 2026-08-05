"""Entity description vectors for type-aware resolution and local graph search."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.graph.ontology import EntityType
from app.indexing.incremental_dense import qdrant_index_path
from app.indexing.qdrant_collections import ENTITIES_ALIAS, VECTOR_NAME, versioned_collection_name


ENTITY_POINT_NAMESPACE = uuid.UUID("b163d286-bf20-4195-9ac1-82bbd53a92f0")


def entity_description_text(entity: dict[str, Any]) -> str:
    return "\n".join((f"Name: {entity.get('canonical_name') or ''}",
                      f"Type: {entity.get('entity_type') or ''}",
                      f"Aliases: {'; '.join(entity.get('aliases') or [])}",
                      f"Description: {entity.get('description') or ''}"))


def sync_entity_embeddings(project_root: Path, provider: EmbeddingProvider,
                           entities: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    if not entities:
        return {"embedded_count": 0, "upserted_count": 0}
    model_name = str(getattr(provider, "model_name"))
    revision = getattr(provider, "revision", None)
    collection = versioned_collection_name(ENTITIES_ALIAS, model_name, revision)
    texts = [entity_description_text(value) for value in entities]
    vectors = provider.embed_documents(texts)
    if len(vectors) != len(entities):
        raise RuntimeError("entity embedding count mismatch")
    client = QdrantClient(path=str(qdrant_index_path(project_root)))
    try:
        existing = {value.name for value in client.get_collections().collections}
        if collection not in existing:
            client.create_collection(collection_name=collection, vectors_config={
                VECTOR_NAME: models.VectorParams(size=len(vectors[0]),
                                                  distance=models.Distance.COSINE)
            })
        points: list[models.PointStruct] = []
        for entity, text, vector in zip(entities, texts, vectors, strict=True):
            description_hash = hashlib.sha256(text.encode()).hexdigest()
            point_id = str(uuid.uuid5(ENTITY_POINT_NAMESPACE,
                f"{entity['entity_id']}\x1f{description_hash}\x1f{model_name}\x1f{revision or ''}"))
            old, _ = client.scroll(collection_name=collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="entity_id", match=models.MatchValue(value=entity["entity_id"]))]),
                limit=64, with_payload=True, with_vectors=False)
            old_ids = [value.id for value in old if (value.payload or {}).get("valid_to_epoch") is None]
            if old_ids:
                client.set_payload(collection_name=collection,
                    payload={"valid_to_epoch": epoch}, points=old_ids, wait=True)
            points.append(models.PointStruct(id=point_id, vector={VECTOR_NAME: vector}, payload={
                **entity, "description_hash": description_hash,
                "valid_from_epoch": epoch, "valid_to_epoch": None,
            }))
        client.upsert(collection_name=collection, points=points, wait=True)
    finally:
        client.close()
    return {"collection_name": collection, "embedded_count": len(points),
            "upserted_count": len(points)}


def entity_vector_matcher(project_root: Path, provider: EmbeddingProvider,
                          active_epoch: int | None) -> Callable[[str, EntityType], tuple[str, float] | None]:
    def match(description: str, entity_type: EntityType) -> tuple[str, float] | None:
        if active_epoch is None:
            return None
        collection = versioned_collection_name(ENTITIES_ALIAS,
            str(getattr(provider, "model_name")), getattr(provider, "revision", None))
        client = QdrantClient(path=str(qdrant_index_path(project_root)))
        try:
            if collection not in {value.name for value in client.get_collections().collections}:
                return None
            response = client.query_points(collection_name=collection,
                query=provider.embed_query(description), using=VECTOR_NAME, limit=1,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="entity_type",
                        match=models.MatchValue(value=entity_type.value)),
                    models.FieldCondition(key="valid_from_epoch",
                        range=models.Range(lte=active_epoch)),
                ], must_not=[models.FieldCondition(key="valid_to_epoch",
                    range=models.Range(lte=active_epoch))]), with_payload=True)
            if not response.points:
                return None
            point = response.points[0]
            return str((point.payload or {}).get("entity_id")), float(point.score)
        finally:
            client.close()
    return match
