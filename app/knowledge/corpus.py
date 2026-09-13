"""Single source of truth for corpus and index statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.indexing.bm25 import chunks_digest
from app.indexing.paper import load_paper_records, papers_digest
from app.retrieval.search import load_chunks


CORPUS_SUMMARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CorpusSummary:
    """Explicitly separated metadata, canonical and searchable corpus counts."""

    schema_version: str
    corpus_revision: str
    canonical_document_count: int
    canonical_valid_count: int
    paper_count: int
    indexed_paper_count: int
    indexed_document_count: int
    section_count: int
    chunk_count: int
    searchable_chunk_count: int
    bm25_index_count: int
    dense_index_count: int
    paper_bm25_index_count: int
    paper_dense_index_count: int
    canonical_paper_ids: list[str]
    metadata_only_paper_ids: list[str]
    chunks_digest: str
    papers_digest: str
    index_consistent: bool
    consistency_issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    values: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _canonical_paper_ids(root: Path) -> tuple[list[str], int]:
    canonical_root = root / "data" / "canonical"
    ids: list[str] = []
    file_count = 0
    if canonical_root.is_dir():
        for path in sorted(canonical_root.glob("*/paper.json")):
            file_count += 1
            value = _json_object(path)
            paper_id = value.get("paper_id")
            if isinstance(paper_id, str) and paper_id.strip():
                ids.append(paper_id.strip())
    return sorted(set(ids)), file_count


def _index_count(
    path: Path,
    *,
    digest_field: str,
    digest: str,
    count_field: str,
    required_directory: Path | None = None,
) -> int:
    value = _json_object(path)
    if value.get(digest_field) != digest:
        return 0
    if required_directory is not None and not required_directory.is_dir():
        return 0
    count = value.get(count_field)
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def corpus_summary(project_root: Path) -> CorpusSummary:
    """Build all user-visible corpus statistics from one local corpus revision."""

    root = project_root.expanduser().resolve()
    # Use the same read boundary as production retrieval.  When structured
    # persistence is enabled this is the normalized projection plus its
    # citation metadata; otherwise it is the rebuildable JSONL projection.
    try:
        chunks = load_chunks(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        chunks = []
    try:
        papers = load_paper_records(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        papers = []
    canonical_ids, canonical_file_count = _canonical_paper_ids(root)
    canonical_set = set(canonical_ids)
    indexed_paper_ids = sorted(
        {
            str(value.get("paper_id"))
            for value in chunks
            if isinstance(value.get("paper_id"), str) and value.get("paper_id")
        }
    )
    indexed_document_ids = sorted(
        {
            str(value.get("document_id"))
            for value in chunks
            if isinstance(value.get("document_id"), str)
            and value.get("document_id")
        }
    )
    metadata_paper_ids = sorted(
        {
            str(value.get("paper_id"))
            for value in papers
            if isinstance(value.get("paper_id"), str) and value.get("paper_id")
        }
    )
    metadata_only = sorted(set(metadata_paper_ids) - set(indexed_paper_ids))
    digest = chunks_digest(chunks)
    metadata_digest = papers_digest(papers)
    corpus_revision = "CR_" + hashlib.sha256(
        f"{digest}:{metadata_digest}".encode("ascii")
    ).hexdigest()[:24]

    section_count = 0
    structure_root = root / "data" / "knowledge" / "structures"
    if structure_root.is_dir():
        for path in sorted(structure_root.glob("*.structure.json")):
            sections = _json_object(path).get("sections")
            if isinstance(sections, list):
                section_count += sum(isinstance(value, dict) for value in sections)

    chunk_count = len(chunks)
    searchable_chunk_count = sum(value.get("searchable", True) is not False for value in chunks)
    issues: list[str] = []
    if canonical_file_count != len(canonical_ids):
        issues.append("canonical_file_count differs from canonical_valid_count")
    if set(indexed_paper_ids) - canonical_set:
        issues.append("indexed paper IDs missing from canonical corpus")
    if not chunks:
        issues.append("chunk projection is empty")

    bm25_count = _index_count(
        root / "data" / "index" / "bm25.json",
        digest_field="chunks_digest",
        digest=digest,
        count_field="document_count",
    )
    dense_count = _index_count(
        root / "data" / "index" / "dense_manifest.json",
        digest_field="chunks_digest",
        digest=digest,
        count_field="chunk_count",
        required_directory=root / "data" / "index" / "qdrant_dense",
    )
    paper_bm25_count = _index_count(
        root / "data" / "index" / "paper_bm25.json",
        digest_field="papers_digest",
        digest=metadata_digest,
        count_field="document_count",
    )
    paper_dense_count = _index_count(
        root / "data" / "index" / "paper_dense_manifest.json",
        digest_field="papers_digest",
        digest=metadata_digest,
        count_field="paper_count",
        required_directory=root / "data" / "index" / "qdrant_papers_dense",
    )
    if bm25_count != chunk_count:
        issues.append("BM25 index does not cover the current chunk projection")
    if dense_count != chunk_count:
        issues.append("Dense index does not cover the current chunk projection")
    if paper_bm25_count != len(papers):
        issues.append("Paper BM25 index does not cover the current paper projection")
    if paper_dense_count != len(papers):
        issues.append("Paper Dense index does not cover the current paper projection")

    return CorpusSummary(
        schema_version=CORPUS_SUMMARY_SCHEMA_VERSION,
        corpus_revision=corpus_revision,
        canonical_document_count=len(canonical_ids),
        canonical_valid_count=len(canonical_ids),
        paper_count=len(papers),
        indexed_paper_count=len(indexed_paper_ids),
        indexed_document_count=len(indexed_document_ids),
        section_count=section_count,
        chunk_count=chunk_count,
        searchable_chunk_count=searchable_chunk_count,
        bm25_index_count=bm25_count,
        dense_index_count=dense_count,
        paper_bm25_index_count=paper_bm25_count,
        paper_dense_index_count=paper_dense_count,
        canonical_paper_ids=canonical_ids,
        metadata_only_paper_ids=metadata_only,
        chunks_digest=digest,
        papers_digest=metadata_digest,
        index_consistent=not issues,
        consistency_issues=issues,
    )


__all__ = ["CORPUS_SUMMARY_SCHEMA_VERSION", "CorpusSummary", "corpus_summary"]
