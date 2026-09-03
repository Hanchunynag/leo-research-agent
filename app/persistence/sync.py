"""Synchronize rebuildable Canonical/Chunk projections into MySQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.indexing.paper import load_paper_records
from app.persistence.mysql import MySQLKnowledgeRepository, build_knowledge_repository


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def sync_knowledge_to_mysql(
    project_root: Path,
    *,
    chunks: Sequence[Mapping[str, Any]],
    repository: MySQLKnowledgeRepository | None = None,
) -> dict[str, Any]:
    """Write facts only when MySQL is enabled; return an auditable summary."""
    owns_repository = repository is None
    repo = repository or build_knowledge_repository(project_root)
    if repo is None:
        return {"enabled": False, "paper_count": 0, "section_count": 0, "chunk_count": 0}
    root = project_root.expanduser().resolve()
    # Synchronization must read the current rebuildable projections, never a
    # stale copy of the same rows from the destination database.
    papers = load_paper_records(root, prefer_database=False)
    repo.upsert_papers(papers)
    canonical_root = root / "data" / "canonical"
    section_values: list[dict[str, Any]] = []
    for paper in papers:
        document_id = str(paper.get("document_id") or "")
        if not document_id:
            continue
        canonical_path = canonical_root / document_id / "paper.json"
        canonical = _object(canonical_path) if canonical_path.is_file() else {}
        source_value = canonical.get("source")
        source: Mapping[str, Any] = source_value if isinstance(source_value, Mapping) else {}
        canonical_metadata = canonical.get("metadata")
        if not isinstance(canonical_metadata, Mapping):
            canonical_metadata = {}
        repo.upsert_version({
            "document_id": document_id,
            "paper_id": paper.get("paper_id"),
            "source_sha256": source.get("sha256"),
            "canonical_path": f"data/canonical/{document_id}/paper.json",
            "pdf_path": paper.get("pdf_path"),
            "metadata": canonical_metadata,
        })
        structure_path = root / "data" / "knowledge" / "structures" / f"{document_id}.structure.json"
        structure = _object(structure_path) if structure_path.is_file() else {}
        for section in structure.get("sections") or []:
            if not isinstance(section, Mapping):
                continue
            section_values.append({
                **dict(section),
                "paper_id": paper.get("paper_id"),
                "document_id": document_id,
            })
    repo.upsert_sections(section_values)
    repo.upsert_chunks([dict(value) for value in chunks])
    repo.mark_missing_local_records(
        paper_ids=[str(value.get("paper_id") or "") for value in papers],
        chunk_ids=[str(value.get("chunk_id") or "") for value in chunks],
    )
    result = {
        "enabled": True,
        "paper_count": len(papers),
        "section_count": len(section_values),
        "chunk_count": len(chunks),
    }
    if owns_repository:
        repo.close()
    return result
