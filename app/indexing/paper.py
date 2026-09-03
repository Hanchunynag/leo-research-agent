"""Paper-level metadata projection and deterministic BM25 index.

This module is deliberately additive.  The existing chunk BM25 index remains the
source used by the legacy retrieval path; this index is consumed only by the
hierarchical retriever.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.indexing.tokenization import tokenize
from app.storage import write_json_atomic, write_jsonl_atomic


PAPER_BM25_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    keywords: list[str]
    doi: str | None = None
    work_id: str | None = None
    document_id: str | None = None
    status: str = "indexed"
    metadata_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperBM25BuildReport:
    status: str
    index_path: str
    paper_count: int
    document_count: int
    papers_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paper_bm25_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "index" / "paper_bm25.json"


def paper_records_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "knowledge" / "paper_records.jsonl"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _optional_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 9999 else None


def _canonical_metadata(project_root: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Best-effort enrichment from the existing canonical document."""

    canonical_path = value.get("canonical_path")
    if isinstance(canonical_path, str) and canonical_path:
        path = project_root / canonical_path
    else:
        paper_id = value.get("paper_id")
        path = project_root / "data" / "canonical" / str(paper_id) / "paper.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    metadata = document.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _record_from_mapping(project_root: Path, value: dict[str, Any]) -> PaperRecord | None:
    metadata = _canonical_metadata(project_root, value)
    paper_id = str(value.get("paper_id") or value.get("work_id") or "").strip()
    title = str(value.get("title") or metadata.get("title") or "").strip()
    if not paper_id or not title:
        return None
    authors = _string_list(value.get("authors")) or _string_list(metadata.get("authors"))
    keywords = _string_list(value.get("keywords")) or _string_list(metadata.get("keywords"))
    abstract = str(value.get("abstract") or metadata.get("abstract") or "").strip()
    document_id = str(value.get("document_id") or "").strip() or None
    work_id = str(value.get("work_id") or "").strip() or None
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        year=_optional_year(value.get("year", metadata.get("year"))),
        keywords=keywords,
        doi=str(value.get("doi") or metadata.get("doi") or "").strip() or None,
        work_id=work_id,
        document_id=document_id,
        status=str(value.get("status") or "indexed"),
        metadata_source=str(value.get("metadata_source") or "canonical"),
    )


def load_paper_records(
    project_root: Path,
    *,
    prefer_database: bool = True,
) -> list[dict[str, Any]]:
    """Load the paper projection, falling back to the existing catalog/chunks.

    The fallback keeps old corpora usable during migration.  It does not alter
    ``papers.jsonl`` or ``chunks.jsonl``.
    """

    root = project_root.expanduser().resolve()
    # MySQL is authoritative when explicitly enabled.  A disabled or
    # unavailable database falls back to the existing projections so old
    # corpora and offline tests remain usable.
    try:
        from app.persistence import build_knowledge_repository

        if prefer_database:
            repository = build_knowledge_repository(root)
            if repository is not None:
                try:
                    return repository.list_papers()
                finally:
                    repository.close()
    except Exception:
        # The configured repository owns fail-closed behavior for callers that
        # explicitly disable fallback.  This compatibility loader only falls
        # through when fallback is enabled or the database is not configured.
        from app.persistence.mysql import MySQLConfig

        if not MySQLConfig.from_environment(root).fallback_to_json:
            raise
    candidates: list[dict[str, Any]] = []
    projection = paper_records_path(root)
    if projection.is_file():
        for line in projection.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    candidates.append(value)
    if not candidates:
        catalog = root / "data" / "knowledge" / "papers.jsonl"
        if catalog.is_file():
            for line in catalog.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        candidates.append(value)
    if not candidates:
        chunks = root / "data" / "knowledge" / "chunks.jsonl"
        if chunks.is_file():
            seen: set[str] = set()
            for line in chunks.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    continue
                key = str(value.get("paper_id") or value.get("work_id") or "")
                if key and key not in seen:
                    seen.add(key)
                    candidates.append(value)
    records: dict[str, PaperRecord] = {}
    for value in candidates:
        record = _record_from_mapping(root, value)
        if record is not None and record.status not in {"deleted", "superseded"}:
            records[record.paper_id] = record
    return [record.to_dict() for record in sorted(records.values(), key=lambda item: item.paper_id)]


def write_paper_records(project_root: Path, records: Iterable[dict[str, Any]]) -> Path:
    output = paper_records_path(project_root)
    payload = [dict(value) for value in records]
    write_jsonl_atomic(output, payload)
    return output


def paper_retrieval_text(paper: dict[str, Any]) -> str:
    raw_authors = paper.get("authors")
    raw_keywords = paper.get("keywords")
    authors: list[Any] = list(raw_authors) if isinstance(raw_authors, list) else []
    keywords: list[Any] = list(raw_keywords) if isinstance(raw_keywords, list) else []
    return "\n".join(
        (
            f"Title: {paper.get('title') or ''}",
            f"Abstract: {paper.get('abstract') or ''}",
            f"Authors: {' '.join(str(value) for value in authors)}",
            f"Year: {paper.get('year') or ''}",
            f"Keywords: {' '.join(str(value) for value in keywords)}",
        )
    )


def papers_digest(papers: list[dict[str, Any]]) -> str:
    payload = [
        {key: value for key, value in paper.items() if key not in {"status"}}
        for paper in papers
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_paper_bm25_index(
    project_root: Path,
    papers: list[dict[str, Any]] | None = None,
    *,
    force: bool = False,
) -> PaperBM25BuildReport:
    root = project_root.expanduser().resolve()
    raw_values = papers if papers is not None else load_paper_records(root)
    normalized_values = [
        record.to_dict()
        for value in raw_values
        if isinstance(value, dict)
        for record in [_record_from_mapping(root, value)]
        if record is not None and record.status not in {"deleted", "superseded"}
    ]
    values = normalized_values
    write_paper_records(root, values)
    digest = papers_digest(values)
    output = paper_bm25_index_path(root)
    if not force and output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("paper_bm25_schema_version") == PAPER_BM25_SCHEMA_VERSION
                and existing.get("papers_digest") == digest
            ):
                return PaperBM25BuildReport("reused", str(output), len(values), int(existing.get("document_count", 0)), digest)
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

    postings: dict[str, list[list[int]]] = {}
    documents: list[dict[str, Any]] = []
    total_length = 0
    for paper_index, paper in enumerate(values):
        tokens = tokenize(paper_retrieval_text(paper))
        frequencies = Counter(tokens)
        total_length += len(tokens)
        documents.append({**paper, "length": len(tokens)})
        for term, frequency in frequencies.items():
            postings.setdefault(term, []).append([paper_index, frequency])
    index_payload = {
        "paper_bm25_schema_version": PAPER_BM25_SCHEMA_VERSION,
        "papers_digest": digest,
        "index_epoch": f"PA_{digest[:16]}",
        "tokenizer_version": "app.indexing.tokenization.v1",
        "document_count": len(documents),
        "average_document_length": total_length / len(documents) if documents else 0.0,
        "documents": documents,
        "postings": postings,
    }
    write_json_atomic(output, index_payload)
    return PaperBM25BuildReport("built", str(output), len(values), len(documents), digest)
