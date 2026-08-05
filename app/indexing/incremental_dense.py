"""Incremental BGE-M3 dense synchronization for Qdrant named vectors."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.indexing.dense import dense_chunk_text
from app.indexing.qdrant_collections import (
    CHUNKS_ALIAS, VECTOR_NAME, versioned_collection_name,
)


POINT_NAMESPACE = uuid.UUID("fcf0d93c-f055-4a7b-bd59-1eab4db9252d")


@dataclass(frozen=True)
class IncrementalDenseReport:
    collection_name: str
    added_count: int
    changed_count: int
    deleted_count: int
    unchanged_count: int
    embedded_count: int
    upserted_count: int
    invalidated_count: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qdrant_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data/index/qdrant_graphrag"


def point_id_for_version(
    chunk_key: str, dense_text_hash: str, model_name: str, revision: str | None
) -> str:
    value = "\x1f".join((chunk_key, dense_text_hash, model_name, revision or ""))
    return str(uuid.uuid5(POINT_NAMESPACE, value))


def _provider_metadata(provider: EmbeddingProvider) -> tuple[str, str | None]:
    name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    if not isinstance(name, str) or not name:
        raise ValueError("EmbeddingProvider must expose model_name")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("EmbeddingProvider revision must be str or None")
    return name, revision


def _ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    existing = {value.name for value in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        vector = vectors.get(VECTOR_NAME) if isinstance(vectors, dict) else None
        if vector is not None and int(vector.size) != dimension:
            raise RuntimeError("existing Qdrant collection vector dimension mismatch")
        return
    client.create_collection(
        collection_name=name,
        vectors_config={VECTOR_NAME: models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        )},
        metadata={"embedding_model_family": "BAAI/bge-m3", "vector_name": VECTOR_NAME},
    )


def _invalidate(client: QdrantClient, collection: str, chunk_key: str, epoch: int) -> int:
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key="chunk_key", match=models.MatchValue(value=chunk_key)
        )]),
        limit=256, with_payload=True, with_vectors=False,
    )
    ids = [point.id for point in points if (point.payload or {}).get("valid_to_epoch") is None]
    if ids:
        client.set_payload(collection_name=collection, payload={"valid_to_epoch": epoch},
                           points=ids, wait=True)
    return len(ids)


def _switch_alias(client: QdrantClient, alias: str, collection: str) -> None:
    try:
        aliases = client.get_aliases().aliases
        actions: list[Any] = []
        for value in aliases:
            if value.alias_name == alias and value.collection_name != collection:
                actions.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        if not any(value.alias_name == alias and value.collection_name == collection for value in aliases):
            actions.append(models.CreateAliasOperation(create_alias=models.CreateAlias(
                collection_name=collection, alias_name=alias
            )))
        if actions:
            client.update_collection_aliases(change_aliases_operations=actions)
    except AttributeError:
        # Some embedded Qdrant releases do not expose aliases; callers retain the
        # concrete versioned collection name from the report/registry.
        return


def sync_incremental_dense(
    project_root: Path, provider: EmbeddingProvider, epoch: int, diffs: list[Any],
    *, client: QdrantClient | None = None, alias: str = CHUNKS_ALIAS,
) -> IncrementalDenseReport:
    started = perf_counter()
    model_name, revision = _provider_metadata(provider)
    collection = versioned_collection_name(alias, model_name, revision)
    owned = client is None
    qdrant = client or QdrantClient(path=str(qdrant_index_path(project_root)))
    added = sum(item.kind == "added" for item in diffs)
    changed = sum(item.kind in {"dense_changed", "changed"} for item in diffs)
    deleted = sum(item.kind == "deleted" for item in diffs)
    unchanged = sum(item.kind in {"unchanged", "graph_changed"} for item in diffs)
    candidates = [item for item in diffs if item.current is not None and
                  (item.kind == "added" or item.dense_changed)]
    invalidated = upserted = 0
    try:
        # No changed dense texts means no model call and no collection mutation.
        if candidates:
            vectors = provider.embed_documents([dense_chunk_text(item.current) for item in candidates])
            if len(vectors) != len(candidates) or not vectors:
                raise RuntimeError("EmbeddingProvider returned unexpected vector count")
            dimension = len(vectors[0])
            if not dimension or any(len(value) != dimension for value in vectors):
                raise RuntimeError("embedding vector dimensions are inconsistent")
            _ensure_collection(qdrant, collection, dimension)
            points: list[models.PointStruct] = []
            for item, vector in zip(candidates, vectors, strict=True):
                chunk = item.current
                invalidated += _invalidate(qdrant, collection, item.chunk_key, epoch)
                payload = {
                    key: chunk.get(key) for key in (
                        "chunk_key", "chunk_id", "work_id", "document_id", "paper_id",
                        "section_id", "section_path", "page_start", "page_end", "block_ids",
                        "dense_text_hash", "title", "content", "content_zone",
                    )
                }
                payload.update({"neo4j_chunk_id": item.chunk_key,
                                "valid_from_epoch": epoch, "valid_to_epoch": None,
                                "embedding_model": model_name, "embedding_revision": revision})
                points.append(models.PointStruct(
                    id=point_id_for_version(item.chunk_key, chunk["dense_text_hash"],
                                            model_name, revision),
                    vector={VECTOR_NAME: vector}, payload=payload,
                ))
            qdrant.upsert(collection_name=collection, points=points, wait=True)
            upserted = len(points)
            _switch_alias(qdrant, alias, collection)
        # Deletes/graph-only changes never invoke embedding. Dense changes already
        # invalidate inside the candidate loop.
        existing = {value.name for value in qdrant.get_collections().collections}
        if collection in existing:
            for item in diffs:
                if item.kind == "deleted":
                    invalidated += _invalidate(qdrant, collection, item.chunk_key, epoch)
    finally:
        if owned:
            qdrant.close()
    return IncrementalDenseReport(
        collection, added, changed, deleted, unchanged, len(candidates), upserted,
        invalidated, round((perf_counter() - started) * 1000, 3)
    )
