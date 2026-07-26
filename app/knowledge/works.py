"""从 canonical 文档重建逻辑论文级 ``works.jsonl``。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.knowledge.identity import DOCUMENT_ID_PATTERN, WORK_ID_PATTERN
from app.storage import write_jsonl_atomic


WORK_CATALOG_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class WorkCatalogRecord:
    work_catalog_schema_version: str
    work_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    preferred_document_id: str
    document_ids: list[str]
    paper_ids: list[str]
    canonical_paths: list[str]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkCatalogIssue:
    path: str
    error_type: str
    message: str


@dataclass(frozen=True)
class WorkCatalogBuildResult:
    catalog_path: Path
    records: list[WorkCatalogRecord]
    issues: list[WorkCatalogIssue]
    unresolved_paper_ids: list[str]


@dataclass(frozen=True)
class _WorkDocument:
    work_id: str
    document_id: str
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    canonical_path: str
    updated_at: str
    verification_status: str | None
    has_abstract: bool


def works_catalog_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "knowledge" / "works.jsonl"


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _work_document(
    document: dict[str, Any],
    canonical_file: Path,
    root: Path,
) -> _WorkDocument | None:
    paper_id = document.get("paper_id")
    identity = document.get("identity")
    metadata = document.get("metadata")
    if not isinstance(paper_id, str) or not isinstance(metadata, dict):
        raise ValueError("paper_id 和 metadata 必须有效。")
    if not isinstance(identity, dict) or identity.get("work_id") is None:
        return None
    work_id = identity.get("work_id")
    document_id = identity.get("document_id")
    if not isinstance(work_id, str) or WORK_ID_PATTERN.fullmatch(work_id) is None:
        raise ValueError("identity.work_id 格式无效。")
    if (
        not isinstance(document_id, str)
        or DOCUMENT_ID_PATTERN.fullmatch(document_id) is None
    ):
        raise ValueError("identity.document_id 格式无效。")
    title = metadata.get("title")
    authors = metadata.get("authors")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("metadata.title 不能为空。")
    if not isinstance(authors, list) or not all(
        isinstance(author, str) and author.strip() for author in authors
    ):
        raise ValueError("metadata.authors 必须是字符串数组。")
    year = metadata.get("year")
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise ValueError("metadata.year 必须是整数或 null。")
    verification = metadata.get("verification")
    verification_status = (
        _optional_string(verification.get("status"))
        if isinstance(verification, dict)
        else None
    )
    pipeline = document.get("pipeline")
    updated_at = (
        _optional_string(verification.get("verified_at"))
        if isinstance(verification, dict)
        else None
    ) or (
        _optional_string(pipeline.get("created_at"))
        if isinstance(pipeline, dict)
        else None
    )
    if updated_at is None:
        updated_at = "1970-01-01T00:00:00+00:00"
    return _WorkDocument(
        work_id=work_id,
        document_id=document_id,
        paper_id=paper_id,
        title=title.strip(),
        authors=[author.strip() for author in authors],
        year=year,
        doi=_optional_string(metadata.get("doi")),
        canonical_path=relative_path(canonical_file, root),
        updated_at=updated_at,
        verification_status=verification_status,
        has_abstract=bool(_optional_string(metadata.get("abstract"))),
    )


def _preferred_document(documents: list[_WorkDocument]) -> _WorkDocument:
    return sorted(
        documents,
        key=lambda document: (
            0 if document.verification_status == "verified" else 1,
            0 if document.has_abstract else 1,
            document.paper_id,
        ),
    )[0]


def scan_work_catalog(
    project_root: Path,
) -> tuple[list[WorkCatalogRecord], list[WorkCatalogIssue], list[str]]:
    root = project_root.expanduser().resolve()
    canonical_root = root / "data" / "canonical"
    files = (
        sorted(canonical_root.glob("*/paper.json")) if canonical_root.exists() else []
    )
    grouped: dict[str, list[_WorkDocument]] = {}
    issues: list[WorkCatalogIssue] = []
    unresolved: list[str] = []
    for canonical_file in files:
        try:
            payload = json.loads(canonical_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("paper.json 必须是 JSON 对象。")
            work_document = _work_document(payload, canonical_file, root)
            paper_id = payload.get("paper_id")
            if work_document is None:
                if isinstance(paper_id, str):
                    unresolved.append(paper_id)
                continue
            grouped.setdefault(work_document.work_id, []).append(work_document)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            issues.append(
                WorkCatalogIssue(
                    path=relative_path(canonical_file, root),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    records: list[WorkCatalogRecord] = []
    for work_id, documents in sorted(grouped.items()):
        preferred = _preferred_document(documents)
        ordered = sorted(documents, key=lambda document: document.document_id)
        records.append(
            WorkCatalogRecord(
                work_catalog_schema_version=WORK_CATALOG_SCHEMA_VERSION,
                work_id=work_id,
                title=preferred.title,
                authors=preferred.authors,
                year=preferred.year,
                doi=preferred.doi,
                preferred_document_id=preferred.document_id,
                document_ids=[document.document_id for document in ordered],
                paper_ids=[document.paper_id for document in ordered],
                canonical_paths=[document.canonical_path for document in ordered],
                updated_at=max(document.updated_at for document in documents),
            )
        )
    return records, issues, sorted(unresolved)


def rebuild_work_catalog(project_root: Path) -> WorkCatalogBuildResult:
    records, issues, unresolved = scan_work_catalog(project_root)
    output = works_catalog_path(project_root)
    write_jsonl_atomic(output, (record.to_dict() for record in records))
    return WorkCatalogBuildResult(
        catalog_path=output,
        records=records,
        issues=issues,
        unresolved_paper_ids=unresolved,
    )
