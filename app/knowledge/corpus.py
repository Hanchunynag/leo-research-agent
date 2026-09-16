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
KNOWLEDGE_INDEX_READINESS_SCHEMA_VERSION = "1.0"


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


def _index_projection_status(
    path: Path,
    *,
    digest_field: str,
    expected_digest: str,
    count_field: str,
    expected_count: int,
    required_directory: Path | None = None,
    require_paper_digests: bool = False,
) -> dict[str, Any]:
    """Describe one rebuildable index without touching or repairing it.

    Readiness is intentionally a projection over the persisted artifacts.  It
    must never call a builder, load an embedding model, or silently rebuild an
    index while an HTTP readiness endpoint or a status command is running.
    """

    payload: dict[str, Any] = {
        "status": "not_initialized",
        "path": str(path),
        "indexed_count": 0,
        "expected_count": expected_count,
        "digest": None,
        "expected_digest": expected_digest,
    }
    if expected_count == 0:
        payload["reason"] = "source_projection_empty"
        return payload
    if not path.is_file():
        payload["reason"] = "artifact_missing"
        return payload

    value = _json_object(path)
    if not value:
        payload.update({"status": "degraded", "reason": "artifact_invalid"})
        return payload

    raw_count = value.get(count_field)
    indexed_count = (
        raw_count
        if isinstance(raw_count, int) and not isinstance(raw_count, bool)
        else 0
    )
    digest = value.get(digest_field)
    payload.update(
        {
            "indexed_count": indexed_count,
            "digest": digest if isinstance(digest, str) else None,
        }
    )
    if digest != expected_digest:
        payload.update({"status": "stale", "reason": "source_digest_mismatch"})
        return payload
    if required_directory is not None and not required_directory.is_dir():
        payload.update({"status": "degraded", "reason": "collection_missing"})
        return payload
    if require_paper_digests and not isinstance(value.get("paper_digests"), dict):
        payload.update({"status": "degraded", "reason": "paper_digests_missing"})
        return payload
    if indexed_count != expected_count:
        payload.update({"status": "degraded", "reason": "coverage_mismatch"})
        return payload

    payload["status"] = "ready"
    return payload


def _dense_manifest_status(
    path: Path,
    *,
    schema_field: str,
    expected_schema: str,
    text_policy_field: str,
    expected_text_policy: str,
    expected_digest_field: str,
    expected_digest: str,
    required_directory: Path,
) -> dict[str, Any]:
    """Return safe model/provenance details for one Dense manifest."""

    payload: dict[str, Any] = {
        "status": "not_initialized",
        "path": str(path),
        "schema_version": None,
        "text_policy_version": None,
        "model_name": None,
        "embedding_revision": None,
        "artifact_fingerprint": None,
        "vector_dimension": None,
        "index_epoch": None,
    }
    if not path.is_file():
        return payload
    value = _json_object(path)
    if not value:
        payload.update({"status": "degraded", "reason": "manifest_invalid"})
        return payload

    payload.update(
        {
            "schema_version": value.get(schema_field),
            "text_policy_version": value.get(text_policy_field),
            "model_name": value.get("model_name"),
            "embedding_revision": value.get("model_revision"),
            "artifact_fingerprint": value.get("model_artifact_fingerprint"),
            "vector_dimension": value.get("vector_dimension"),
            "index_epoch": value.get("index_epoch"),
        }
    )
    if not required_directory.is_dir():
        payload.update({"status": "degraded", "reason": "collection_missing"})
    elif (
        value.get(schema_field) != expected_schema
        or value.get(text_policy_field) != expected_text_policy
        or value.get(expected_digest_field) != expected_digest
        or not isinstance(value.get("paper_digests"), dict)
    ):
        payload.update({"status": "stale", "reason": "manifest_provenance_mismatch"})
    else:
        payload["status"] = "ready"
    return payload


def knowledge_index_readiness(project_root: Path) -> dict[str, Any]:
    """Project persisted Paper/Content indexes into a production readiness view.

    Level-1 and Level-2 remain separate lifecycle projections here.  The
    helper only compares source digests, coverage and manifest provenance; it
    never invokes parsing, chunking, BM25, embedding, Qdrant upsert, or a full
    rebuild.  ``stale`` means the source projection changed after an index was
    built, while ``degraded`` means the artifact exists but is incomplete or
    structurally invalid.
    """

    root = project_root.expanduser().resolve()
    summary = corpus_summary(root)
    index_root = root / "data" / "index"

    content_bm25 = _index_projection_status(
        index_root / "bm25.json",
        digest_field="chunks_digest",
        expected_digest=summary.chunks_digest,
        count_field="document_count",
        expected_count=summary.chunk_count,
    )
    paper_bm25 = _index_projection_status(
        index_root / "paper_bm25.json",
        digest_field="papers_digest",
        expected_digest=summary.papers_digest,
        count_field="document_count",
        expected_count=summary.paper_count,
    )

    # Keep the policy/version constants next to the owning index
    # implementations.  Importing them here is read-only and avoids copying
    # the release contract into the readiness layer.
    from app.indexing.dense import (
        DENSE_INDEX_SCHEMA_VERSION,
        DENSE_TEXT_POLICY_VERSION,
    )
    from app.indexing.paper_dense import (
        PAPER_DENSE_SCHEMA_VERSION,
        PAPER_DENSE_TEXT_POLICY_VERSION,
    )

    content_dense_manifest = _dense_manifest_status(
        index_root / "dense_manifest.json",
        schema_field="dense_index_schema_version",
        expected_schema=DENSE_INDEX_SCHEMA_VERSION,
        text_policy_field="dense_text_policy_version",
        expected_text_policy=DENSE_TEXT_POLICY_VERSION,
        expected_digest_field="chunks_digest",
        expected_digest=summary.chunks_digest,
        required_directory=index_root / "qdrant_dense",
    )
    content_dense = _index_projection_status(
        index_root / "dense_manifest.json",
        digest_field="chunks_digest",
        expected_digest=summary.chunks_digest,
        count_field="chunk_count",
        expected_count=summary.chunk_count,
        required_directory=index_root / "qdrant_dense",
        require_paper_digests=True,
    )
    content_dense["manifest_status"] = content_dense_manifest["status"]

    paper_dense_manifest = _dense_manifest_status(
        index_root / "paper_dense_manifest.json",
        schema_field="paper_dense_schema_version",
        expected_schema=PAPER_DENSE_SCHEMA_VERSION,
        text_policy_field="paper_dense_text_policy_version",
        expected_text_policy=PAPER_DENSE_TEXT_POLICY_VERSION,
        expected_digest_field="papers_digest",
        expected_digest=summary.papers_digest,
        required_directory=index_root / "qdrant_papers_dense",
    )
    paper_dense = _index_projection_status(
        index_root / "paper_dense_manifest.json",
        digest_field="papers_digest",
        expected_digest=summary.papers_digest,
        count_field="paper_count",
        expected_count=summary.paper_count,
        required_directory=index_root / "qdrant_papers_dense",
        require_paper_digests=True,
    )
    paper_dense["manifest_status"] = paper_dense_manifest["status"]

    index_statuses = {
        "paper_bm25": paper_bm25["status"],
        "paper_dense": paper_dense["status"],
        "content_bm25": content_bm25["status"],
        "content_dense": content_dense["status"],
    }
    statuses = set(index_statuses.values())
    if summary.paper_count == 0 and summary.chunk_count == 0:
        overall_status = "not_initialized"
    elif statuses == {"ready"} and summary.index_consistent:
        overall_status = "ready"
    elif "stale" in statuses:
        overall_status = "stale"
    else:
        overall_status = "degraded"

    revisions = {
        value
        for value in (
            content_dense_manifest.get("embedding_revision"),
            paper_dense_manifest.get("embedding_revision"),
        )
        if isinstance(value, str) and value
    }
    models = {
        value
        for value in (
            content_dense_manifest.get("model_name"),
            paper_dense_manifest.get("model_name"),
        )
        if isinstance(value, str) and value
    }
    issues = list(summary.consistency_issues)
    if len(revisions) > 1:
        issues.append("Paper and Content Dense embedding revisions differ")
    if len(models) > 1:
        issues.append("Paper and Content Dense embedding models differ")
    manifest_status = {
        "status": (
            "ready"
            if content_dense_manifest["status"] == "ready"
            and paper_dense_manifest["status"] == "ready"
            else "stale"
            if "stale" in {
                content_dense_manifest["status"],
                paper_dense_manifest["status"],
            }
            else "degraded"
        ),
        "content_dense": content_dense_manifest,
        "paper_dense": paper_dense_manifest,
    }

    return {
        "schema_version": KNOWLEDGE_INDEX_READINESS_SCHEMA_VERSION,
        "status": overall_status,
        "paper_count": summary.paper_count,
        "canonical_document_count": summary.canonical_document_count,
        "chunk_count": summary.chunk_count,
        "searchable_chunk_count": summary.searchable_chunk_count,
        "corpus_revision": summary.corpus_revision,
        "papers_digest": summary.papers_digest,
        "chunks_digest": summary.chunks_digest,
        "embedding_model": next(iter(models), None),
        "embedding_revision": next(iter(revisions), None),
        "paper_bm25": paper_bm25,
        "paper_dense": paper_dense,
        "content_bm25": content_bm25,
        "content_dense": content_dense,
        "manifest_status": manifest_status,
        "index_consistent": summary.index_consistent and len(revisions) <= 1 and len(models) <= 1,
        "consistency_issues": issues,
    }


__all__ = [
    "CORPUS_SUMMARY_SCHEMA_VERSION",
    "KNOWLEDGE_INDEX_READINESS_SCHEMA_VERSION",
    "CorpusSummary",
    "corpus_summary",
    "knowledge_index_readiness",
]
