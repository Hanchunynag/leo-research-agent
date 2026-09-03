"""Offline orchestration for the additive two-level index set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.chunking.builder import build_knowledge_base
from app.embeddings.base import EmbeddingProvider
from app.indexing.dense import build_dense_index
from app.indexing.paper_dense import build_paper_dense_index


def build_hierarchical_indexes(
    project_root: Path,
    embedding_provider: EmbeddingProvider,
    *,
    force: bool = False,
    maximum_tokens: int = 700,
    minimum_chunk_tokens: int = 80,
    overlap_tokens: int = 80,
) -> dict[str, Any]:
    """Build Paper BM25/Dense and Chunk BM25/Dense in one compatible flow.

    ``build_knowledge_base`` continues to own Canonical -> Structure -> Chunk
    and legacy Chunk BM25 construction.  This function only adds the Paper
    projection and Paper Dense index around that existing pipeline.
    """

    knowledge = build_knowledge_base(
        project_root,
        force=force,
        maximum_tokens=maximum_tokens,
        minimum_chunk_tokens=minimum_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunk_dense = build_dense_index(project_root, embedding_provider, force=force)
    paper_dense = build_paper_dense_index(project_root, embedding_provider, force=force)
    return {
        "knowledge": knowledge.to_dict(),
        "chunk_dense": chunk_dense.to_dict(),
        "paper_dense": paper_dense.to_dict(),
        "hierarchical": True,
    }
