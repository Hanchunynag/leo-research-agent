"""Stable neo4j-graphrag Qdrant/Neo4j retriever adapter.

The project intentionally does not use the experimental KG Builder. Extraction and ontology
validation remain local; this adapter reuses the supported external retriever/result models.
"""

from __future__ import annotations

import hashlib
from typing import Any

from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from neo4j_graphrag.types import RetrieverResult
from qdrant_client import QdrantClient

from app.graph.models import EvidenceCandidate
from app.indexing.qdrant_collections import VECTOR_NAME


class Neo4jGraphRAGChunkRetriever:
    def __init__(self, driver: object, qdrant: QdrantClient, collection_name: str,
                 database: str = "neo4j") -> None:
        self.retriever = QdrantNeo4jRetriever(
            driver=driver, client=qdrant, collection_name=collection_name,
            id_property_neo4j="chunk_key", id_property_external="neo4j_chunk_id",
            using=VECTOR_NAME, node_label_neo4j="Chunk", neo4j_database=database,
            return_properties=["chunk_key", "chunk_id", "content", "work_id",
                               "document_id", "page_start", "page_end", "block_ids"],
        )

    def search(self, query_vector: list[float], top_k: int = 20) -> tuple[list[EvidenceCandidate], dict[str, Any]]:
        result: RetrieverResult = self.retriever.search(query_vector=query_vector, top_k=top_k)
        candidates: list[EvidenceCandidate] = []
        for item in result.items:
            content = item.content if isinstance(item.content, dict) else {"content": str(item.content)}
            chunk_key = str(content.get("chunk_key") or "")
            evidence_id = "CH_" + hashlib.sha256(chunk_key.encode()).hexdigest()[:20]
            candidates.append(EvidenceCandidate(
                evidence_id=evidence_id, candidate_type="chunk",
                text=str(content.get("content") or ""), source_chunk_keys=[chunk_key],
                routes=["dense"], directness_grade=1,
                metadata={**content, **(item.metadata or {})},
            ))
        return candidates, dict(result.metadata or {})
