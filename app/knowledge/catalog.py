"""从单篇 ``paper.json`` 重建全局 ``papers.jsonl`` 目录。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.knowledge.identity import (
    DOCUMENT_ID_PATTERN,
    WORK_ID_PATTERN,
    document_id_from_sha256,
)
from app.parsing.pipeline import find_mineru_artifacts
from app.storage import write_jsonl_atomic


CATALOG_SCHEMA_VERSION = "1.1"
PAPER_ID_PATTERN = re.compile(r"^P_[0-9a-f]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CatalogValidationError(ValueError):
    """单篇 canonical 文档不能生成可靠目录记录。"""


@dataclass(frozen=True)
class CatalogIssue:
    """目录扫描期间发现、但不会阻塞其他论文的问题。"""

    path: str
    error_type: str
    message: str


@dataclass(frozen=True)
class PaperCatalogRecord:
    """一篇论文在全库中的轻量记录。"""

    catalog_schema_version: str
    paper_id: str
    document_id: str
    work_id: str | None
    sha256: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    page_count: int
    canonical_path: str
    schema_version: str
    parser_name: str | None
    parser_version: str | None
    parse_status: str
    quality_issue_count: int
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogBuildResult:
    """一次目录重建的结果。"""

    catalog_path: Path
    records: list[PaperCatalogRecord]
    issues: list[CatalogIssue]
    work_catalog_path: Path | None = None
    work_record_count: int = 0
    unresolved_work_count: int = 0
    work_issue_count: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "catalog_path": str(self.catalog_path),
            "record_count": len(self.records),
            "issue_count": len(self.issues),
            "issues": [asdict(issue) for issue in self.issues],
            "work_catalog_path": (
                str(self.work_catalog_path) if self.work_catalog_path else None
            ),
            "work_record_count": self.work_record_count,
            "unresolved_work_count": self.unresolved_work_count,
            "work_issue_count": self.work_issue_count,
        }


@dataclass(frozen=True)
class CatalogLoadResult:
    """读取现有 JSONL 目录的结果。"""

    catalog_path: Path
    records: list[PaperCatalogRecord]
    issues: list[CatalogIssue]


@dataclass(frozen=True)
class LibraryStatus:
    """原始文件、解析产物、canonical 和目录之间的一致性状态。"""

    catalog_path: str
    catalog_exists: bool
    catalog_consistent: bool
    raw_paper_count: int
    parsed_paper_count: int
    canonical_file_count: int
    canonical_valid_count: int
    catalog_record_count: int
    quality_issue_count: int
    unparsed_paper_ids: list[str]
    missing_canonical_paper_ids: list[str]
    missing_raw_paper_ids: list[str]
    catalog_missing_paper_ids: list[str]
    catalog_orphan_paper_ids: list[str]
    catalog_stale_paper_ids: list[str]
    canonical_issues: list[CatalogIssue]
    catalog_issues: list[CatalogIssue]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_issues"] = [asdict(issue) for issue in self.canonical_issues]
        payload["catalog_issues"] = [asdict(issue) for issue in self.catalog_issues]
        return payload


def catalog_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "knowledge" / "papers.jsonl"


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()

    try:
        return resolved.relative_to(project_root.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogValidationError(f"{field} 必须是 JSON 对象。")

    return value


def require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(f"{field} 必须是非空字符串。")

    return value.strip()


def optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise CatalogValidationError(f"{field} 必须是字符串或 null。")

    stripped = value.strip()
    return stripped or None


def optional_year(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise CatalogValidationError("metadata.year 必须是整数或 null。")

    try:
        year = int(value)
    except (TypeError, ValueError) as error:
        raise CatalogValidationError("metadata.year 必须是整数或 null。") from error

    if not 1000 <= year <= 9999:
        raise CatalogValidationError("metadata.year 必须是四位年份或 null。")

    return year


def quality_issue_count(pipeline: dict[str, Any]) -> int:
    adapter_report = pipeline.get("adapter_report")

    if not isinstance(adapter_report, dict):
        return 0

    issue_counts = adapter_report.get("quality_issue_counts")
    total = 0

    if isinstance(issue_counts, dict):
        for value in issue_counts.values():
            if isinstance(value, int) and not isinstance(value, bool):
                total += max(value, 0)

    missing_assets = adapter_report.get("missing_asset_count")
    if isinstance(missing_assets, int) and not isinstance(
        missing_assets,
        bool,
    ):
        total += max(missing_assets, 0)

    return total


def canonical_updated_at(
    document: dict[str, Any],
    canonical_file: Path,
) -> str:
    pipeline = document.get("pipeline")

    if isinstance(pipeline, dict):
        created_at = pipeline.get("created_at")
        if isinstance(created_at, str) and created_at.strip():
            return created_at.strip()

    modified = datetime.fromtimestamp(
        canonical_file.stat().st_mtime,
        tz=timezone.utc,
    )
    return modified.isoformat()


def record_from_document(
    document: dict[str, Any],
    canonical_file: Path,
    project_root: Path,
) -> PaperCatalogRecord:
    """校验单篇 canonical 文档并生成目录记录。"""

    paper_id = require_string(document.get("paper_id"), "paper_id")
    if PAPER_ID_PATTERN.fullmatch(paper_id) is None:
        raise CatalogValidationError("paper_id 必须符合 P_ + 12 位小写十六进制格式。")

    if canonical_file.parent.name != paper_id:
        raise CatalogValidationError("paper_id 与 canonical 目录名不一致。")

    source = require_mapping(document.get("source"), "source")
    sha256 = require_string(source.get("sha256"), "source.sha256")
    if SHA256_PATTERN.fullmatch(sha256) is None:
        raise CatalogValidationError("source.sha256 必须是 64 位小写十六进制字符串。")

    metadata = require_mapping(document.get("metadata"), "metadata")
    title = require_string(metadata.get("title"), "metadata.title")
    authors_value = metadata.get("authors", [])

    if not isinstance(authors_value, list) or not all(
        isinstance(author, str) and author.strip() for author in authors_value
    ):
        raise CatalogValidationError(
            "metadata.authors 必须是字符串数组，且元素不得为空。"
        )

    authors = [author.strip() for author in authors_value]
    year = optional_year(metadata.get("year"))
    doi = optional_string(metadata.get("doi"), "metadata.doi")

    identity = document.get("identity")
    expected_document_id = document_id_from_sha256(sha256)
    document_id = expected_document_id
    work_id: str | None = None
    if identity is not None:
        identity_mapping = require_mapping(identity, "identity")
        document_id = require_string(
            identity_mapping.get("document_id"),
            "identity.document_id",
        )
        work_id = optional_string(
            identity_mapping.get("work_id"),
            "identity.work_id",
        )
    if (
        DOCUMENT_ID_PATTERN.fullmatch(document_id) is None
        or document_id != expected_document_id
    ):
        raise CatalogValidationError("identity.document_id 必须与 source.sha256 一致。")
    if work_id is not None and WORK_ID_PATTERN.fullmatch(work_id) is None:
        raise CatalogValidationError("identity.work_id 格式无效。")

    page_count = document.get("page_count")
    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count < 1
    ):
        raise CatalogValidationError("page_count 必须是正整数。")

    schema_version = require_string(
        document.get("schema_version"),
        "schema_version",
    )
    parser = require_mapping(document.get("parser"), "parser")
    parser_name = optional_string(parser.get("name"), "parser.name")
    parser_version = optional_string(
        parser.get("version"),
        "parser.version",
    )
    pipeline = require_mapping(document.get("pipeline"), "pipeline")

    return PaperCatalogRecord(
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        paper_id=paper_id,
        document_id=document_id,
        work_id=work_id,
        sha256=sha256,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        page_count=page_count,
        canonical_path=project_relative(canonical_file, project_root),
        schema_version=schema_version,
        parser_name=parser_name,
        parser_version=parser_version,
        parse_status="success",
        quality_issue_count=quality_issue_count(pipeline),
        updated_at=canonical_updated_at(document, canonical_file),
    )


def scan_canonical_documents(
    project_root: Path,
) -> tuple[list[PaperCatalogRecord], list[CatalogIssue], int]:
    root = project_root.expanduser().resolve()
    canonical_root = root / "data" / "canonical"
    files = (
        sorted(canonical_root.glob("*/paper.json")) if canonical_root.exists() else []
    )
    records_by_id: dict[str, PaperCatalogRecord] = {}
    issues: list[CatalogIssue] = []

    for canonical_file in files:
        relative_path = project_relative(canonical_file, root)

        try:
            value = json.loads(canonical_file.read_text(encoding="utf-8"))
            document = require_mapping(value, "paper.json")
            record = record_from_document(
                document=document,
                canonical_file=canonical_file,
                project_root=root,
            )

            if record.paper_id in records_by_id:
                raise CatalogValidationError(
                    f"目录中存在重复 paper_id：{record.paper_id}"
                )

            records_by_id[record.paper_id] = record
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            CatalogValidationError,
        ) as error:
            issues.append(
                CatalogIssue(
                    path=relative_path,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    records = sorted(
        records_by_id.values(),
        key=lambda record: record.paper_id,
    )
    return records, issues, len(files)


def rebuild_catalog(project_root: Path) -> CatalogBuildResult:
    """扫描全部 canonical 文档并原子重写 papers.jsonl。"""

    root = project_root.expanduser().resolve()
    records, issues, _ = scan_canonical_documents(root)
    output = catalog_path(root)
    write_jsonl_atomic(
        output,
        (record.to_dict() for record in records),
    )
    from app.knowledge.works import rebuild_work_catalog

    work_result = rebuild_work_catalog(root)
    return CatalogBuildResult(
        catalog_path=output,
        records=records,
        issues=issues,
        work_catalog_path=work_result.catalog_path,
        work_record_count=len(work_result.records),
        unresolved_work_count=len(work_result.unresolved_paper_ids),
        work_issue_count=len(work_result.issues),
    )


def record_from_catalog_payload(
    payload: dict[str, Any],
) -> PaperCatalogRecord:
    """校验 JSONL 中的一条目录记录。"""

    try:
        record = PaperCatalogRecord(**payload)
    except TypeError as error:
        raise CatalogValidationError(
            f"目录记录字段不完整或包含未知字段：{error}"
        ) from error

    if record.catalog_schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogValidationError(
            f"不支持的 catalog_schema_version：{record.catalog_schema_version}"
        )

    if (
        not isinstance(record.paper_id, str)
        or PAPER_ID_PATTERN.fullmatch(record.paper_id) is None
    ):
        raise CatalogValidationError("目录记录中的 paper_id 格式无效。")

    if (
        not isinstance(record.sha256, str)
        or SHA256_PATTERN.fullmatch(record.sha256) is None
    ):
        raise CatalogValidationError("目录记录中的 sha256 格式无效。")

    if (
        not isinstance(record.document_id, str)
        or DOCUMENT_ID_PATTERN.fullmatch(record.document_id) is None
        or record.document_id != document_id_from_sha256(record.sha256)
    ):
        raise CatalogValidationError("目录记录中的 document_id 格式无效。")

    if record.work_id is not None and (
        not isinstance(record.work_id, str)
        or WORK_ID_PATTERN.fullmatch(record.work_id) is None
    ):
        raise CatalogValidationError("目录记录中的 work_id 格式无效。")

    if not isinstance(record.title, str) or not record.title.strip():
        raise CatalogValidationError("目录记录中的 title 不能为空。")

    if record.year is not None and (
        not isinstance(record.year, int)
        or isinstance(record.year, bool)
        or not 1000 <= record.year <= 9999
    ):
        raise CatalogValidationError("目录记录中的 year 必须是四位整数或 null。")

    for value, field in (
        (record.doi, "doi"),
        (record.parser_name, "parser_name"),
        (record.parser_version, "parser_version"),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise CatalogValidationError(
                f"目录记录中的 {field} 必须是非空字符串或 null。"
            )

    if not isinstance(record.authors, list) or not all(
        isinstance(author, str) and author.strip() for author in record.authors
    ):
        raise CatalogValidationError("目录记录中的 authors 必须是字符串数组。")

    if (
        not isinstance(record.page_count, int)
        or isinstance(record.page_count, bool)
        or record.page_count < 1
    ):
        raise CatalogValidationError("目录记录中的 page_count 必须是正整数。")

    if (
        not isinstance(record.quality_issue_count, int)
        or isinstance(record.quality_issue_count, bool)
        or record.quality_issue_count < 0
    ):
        raise CatalogValidationError(
            "目录记录中的 quality_issue_count 必须是非负整数。"
        )

    if record.parse_status != "success":
        raise CatalogValidationError("目录记录中的 parse_status 必须是 success。")

    for value, field in (
        (record.canonical_path, "canonical_path"),
        (record.schema_version, "schema_version"),
        (record.updated_at, "updated_at"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CatalogValidationError(f"目录记录中的 {field} 不能为空。")

    return record


def load_catalog(project_root: Path) -> CatalogLoadResult:
    """读取现有 papers.jsonl，并逐行报告损坏记录。"""

    output = catalog_path(project_root)
    if not output.exists():
        raise FileNotFoundError(f"论文目录不存在：{output}。请先运行 library rebuild。")

    records_by_id: dict[str, PaperCatalogRecord] = {}
    issues: list[CatalogIssue] = []

    try:
        content = output.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return CatalogLoadResult(
            catalog_path=output,
            records=[],
            issues=[
                CatalogIssue(
                    path=str(output),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            ],
        )

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        path_label = f"{output}:{line_number}"
        try:
            value = json.loads(line)
            payload = require_mapping(value, "catalog record")
            record = record_from_catalog_payload(payload)
            if record.paper_id in records_by_id:
                raise CatalogValidationError(
                    f"目录中存在重复 paper_id：{record.paper_id}"
                )
            records_by_id[record.paper_id] = record
        except (json.JSONDecodeError, CatalogValidationError) as error:
            issues.append(
                CatalogIssue(
                    path=path_label,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    records = sorted(
        records_by_id.values(),
        key=lambda record: record.paper_id,
    )
    return CatalogLoadResult(
        catalog_path=output,
        records=records,
        issues=issues,
    )


def library_status(project_root: Path) -> LibraryStatus:
    """比较本地论文库各阶段的文件数量和 paper_id。"""

    root = project_root.expanduser().resolve()
    raw_root = root / "data" / "raw"
    parsed_root = root / "data" / "parsed"

    raw_ids = (
        {
            directory.name
            for directory in raw_root.iterdir()
            if directory.is_dir()
            and PAPER_ID_PATTERN.fullmatch(directory.name)
            and any(
                path.is_file() and path.suffix.lower() == ".pdf"
                for path in directory.iterdir()
            )
        }
        if raw_root.exists()
        else set()
    )

    parsed_ids = (
        {
            directory.name
            for directory in parsed_root.iterdir()
            if directory.is_dir()
            and PAPER_ID_PATTERN.fullmatch(directory.name)
            and find_mineru_artifacts(directory / "mineru") is not None
        }
        if parsed_root.exists()
        else set()
    )

    canonical_records, canonical_issues, canonical_file_count = (
        scan_canonical_documents(root)
    )
    canonical_ids = {record.paper_id for record in canonical_records}

    output = catalog_path(root)
    if output.exists():
        loaded = load_catalog(root)
        catalog_records = loaded.records
        catalog_issues = loaded.issues
    else:
        catalog_records = []
        catalog_issues = []

    catalog_ids = {record.paper_id for record in catalog_records}
    canonical_by_id = {record.paper_id: record for record in canonical_records}
    catalog_by_id = {record.paper_id: record for record in catalog_records}
    missing_canonical = raw_ids - canonical_ids
    missing_raw = canonical_ids - raw_ids
    catalog_missing = canonical_ids - catalog_ids
    catalog_orphan = catalog_ids - canonical_ids
    catalog_stale = {
        paper_id
        for paper_id in canonical_ids & catalog_ids
        if canonical_by_id[paper_id].to_dict() != catalog_by_id[paper_id].to_dict()
    }
    unparsed = raw_ids - parsed_ids
    consistent = (
        output.exists()
        and not canonical_issues
        and not catalog_issues
        and not unparsed
        and not missing_canonical
        and not missing_raw
        and not catalog_missing
        and not catalog_orphan
        and not catalog_stale
    )

    return LibraryStatus(
        catalog_path=str(output),
        catalog_exists=output.exists(),
        catalog_consistent=consistent,
        raw_paper_count=len(raw_ids),
        parsed_paper_count=len(parsed_ids),
        canonical_file_count=canonical_file_count,
        canonical_valid_count=len(canonical_records),
        catalog_record_count=len(catalog_records),
        quality_issue_count=sum(
            record.quality_issue_count for record in canonical_records
        ),
        unparsed_paper_ids=sorted(unparsed),
        missing_canonical_paper_ids=sorted(missing_canonical),
        missing_raw_paper_ids=sorted(missing_raw),
        catalog_missing_paper_ids=sorted(catalog_missing),
        catalog_orphan_paper_ids=sorted(catalog_orphan),
        catalog_stale_paper_ids=sorted(catalog_stale),
        canonical_issues=canonical_issues,
        catalog_issues=catalog_issues,
    )
