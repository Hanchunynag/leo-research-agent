"""Named-vector collection definitions and model-version aliases."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


VECTOR_NAME = "dense"
CHUNKS_ALIAS = "leo_chunks_dense"
ENTITIES_ALIAS = "leo_entities_dense"
COMMUNITIES_ALIAS = "leo_community_reports_dense"


@dataclass(frozen=True)
class CollectionDefinition:
    alias: str
    vector_name: str = VECTOR_NAME
    distance: str = "Cosine"


CHUNK_COLLECTION = CollectionDefinition(CHUNKS_ALIAS)
ENTITY_COLLECTION = CollectionDefinition(ENTITIES_ALIAS)
COMMUNITY_COLLECTION = CollectionDefinition(COMMUNITIES_ALIAS)


def versioned_collection_name(alias: str, model_name: str, revision: str | None) -> str:
    fingerprint = hashlib.sha256(
        f"{model_name}\x1f{revision or 'default'}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{alias}_{fingerprint}"
