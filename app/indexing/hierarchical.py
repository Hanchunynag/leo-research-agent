"""Offline orchestration for the additive two-level index set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.chunking.builder import build_knowledge_base
from app.embeddings.base import EmbeddingProvider
from app.indexing.dense import build_dense_index
from app.indexing.paper import load_paper_records, papers_digest
from app.indexing.paper_dense import build_paper_dense_index
from app.indexing.tokenization import TOKENIZER_VERSION


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
    and content-level Chunk BM25 construction.  This function only adds the Paper
    projection and Paper Dense index around that existing pipeline.
    """

    knowledge = build_knowledge_base(
        project_root,
        force=force,
        maximum_tokens=maximum_tokens,
        minimum_chunk_tokens=minimum_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    changed_paper_ids = set(knowledge.changed_paper_ids)
    chunk_dense = build_dense_index(
        project_root,
        embedding_provider,
        force=force,
        changed_paper_ids=changed_paper_ids,
    )
    paper_dense = build_paper_dense_index(
        project_root,
        embedding_provider,
        force=force,
        changed_paper_ids=changed_paper_ids,
    )
    epoch_id = f"HE_{papers_digest(load_paper_records(project_root))[:16]}_{chunk_dense.chunks_digest[:16]}"
    try:
        from app.persistence import build_knowledge_repository

        repository = build_knowledge_repository(project_root)
        if repository is not None:
            repository.record_index_epoch({
                "epoch_id": epoch_id,
                "index_kind": "hierarchical",
                "source_fingerprint": f"{paper_dense.papers_digest}:{chunk_dense.chunks_digest}",
                "embedding_model": chunk_dense.model_name,
                "embedding_revision": chunk_dense.model_revision,
                "tokenizer_version": TOKENIZER_VERSION,
                "chunker_version": knowledge.chunk_policy_version,
                "status": "active",
            })
            repository.close()
    except Exception:
        # A configured structured store is part of the production index
        # contract. Only an explicitly disabled/fallback-enabled local setup
        # may continue with the rebuildable JSON projections.
        from app.persistence.mysql import MySQLConfig

        config = MySQLConfig.from_environment(project_root)
        if config.enabled and not config.fallback_to_json:
            raise
    return {
        "knowledge": knowledge.to_dict(),
        "chunk_dense": chunk_dense.to_dict(),
        "paper_dense": paper_dense.to_dict(),
        "index_epoch": epoch_id,
        "hierarchical": True,
    }
