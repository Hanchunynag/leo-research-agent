"""容错的多 PDF 批量解析入口。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from app.knowledge.catalog import CatalogIssue, rebuild_catalog
from app.parsing.pipeline import (
    PaperParseConfig,
    PaperProgressCallback,
    PaperParseStage,
    parse_paper,
)
from app.storage import write_json_atomic


BatchItemStatus = Literal["success", "reused", "failed"]
BatchProgressStage = PaperParseStage | Literal["completed"]
BatchProgressCallback = Callable[
    [int, int, Path, BatchProgressStage],
    None,
]


@dataclass(frozen=True)
class BatchItemResult:
    """批处理中单个 PDF 的结果。"""

    input_path: str
    status: BatchItemStatus
    paper_id: str | None
    paper_json: str | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class BatchParseReport:
    """一次批量解析的可持久化报告。"""

    run_id: str
    input_source: str
    recursive: bool
    started_at: str
    finished_at: str
    total_count: int
    success_count: int
    reused_count: int
    failed_count: int
    catalog_record_count: int
    catalog_issues: list[CatalogIssue]
    report_path: Path
    items: list[BatchItemResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input_source": self.input_source,
            "recursive": self.recursive,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "reused_count": self.reused_count,
            "failed_count": self.failed_count,
            "catalog_record_count": self.catalog_record_count,
            "catalog_issue_count": len(self.catalog_issues),
            "catalog_issues": [asdict(issue) for issue in self.catalog_issues],
            "report_path": str(self.report_path),
            "items": [asdict(item) for item in self.items],
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def discover_pdfs(
    input_directory: Path,
    recursive: bool,
) -> list[Path]:
    """稳定排序并去重地发现 PDF，扩展名大小写不敏感。"""

    directory = input_directory.expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(directory)

    if not directory.is_dir():
        raise ValueError(f"批量输入路径不是目录：{directory}")

    candidates = directory.rglob("*") if recursive else directory.iterdir()
    unique_paths = {
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() == ".pdf"
    }
    return sorted(unique_paths, key=lambda path: path.as_posix().casefold())


def batch_parse_directory(
    input_directory: Path,
    config: PaperParseConfig,
    recursive: bool = False,
    progress_callback: BatchProgressCallback | None = None,
) -> BatchParseReport:
    """发现目录中的 PDF，然后交给统一多文件批处理。"""

    directory = input_directory.expanduser().resolve()
    pdfs = discover_pdfs(directory, recursive=recursive)
    return batch_parse_files(
        input_paths=pdfs,
        config=config,
        input_source=str(directory),
        recursive=recursive,
        progress_callback=progress_callback,
    )


def batch_parse_files(
    input_paths: Iterable[Path],
    config: PaperParseConfig,
    input_source: str = "file-list",
    recursive: bool = False,
    progress_callback: BatchProgressCallback | None = None,
) -> BatchParseReport:
    """逐篇运行 parse_paper；单篇失败不会终止后续论文。"""

    normalized_config = config.normalized()
    pdfs = sorted(
        {path.expanduser().resolve() for path in input_paths},
        key=lambda path: path.as_posix().casefold(),
    )
    started = utc_now()
    items: list[BatchItemResult] = []

    for index, pdf in enumerate(pdfs, start=1):
        paper_progress_callback: PaperProgressCallback | None = None

        if progress_callback is not None:

            def report_paper_stage(stage: PaperParseStage) -> None:
                assert progress_callback is not None
                progress_callback(
                    index - 1,
                    len(pdfs),
                    pdf,
                    stage,
                )

            paper_progress_callback = report_paper_stage

        try:
            result = parse_paper(
                input_path=pdf,
                config=normalized_config,
                progress_callback=paper_progress_callback,
            )
            status: BatchItemStatus = "reused" if result.mineru_reused else "success"
            items.append(
                BatchItemResult(
                    input_path=str(pdf),
                    status=status,
                    paper_id=result.paper_id,
                    paper_json=str(result.paper_json),
                    error_type=None,
                    error_message=None,
                )
            )
        except Exception as error:
            items.append(
                BatchItemResult(
                    input_path=str(pdf),
                    status="failed",
                    paper_id=None,
                    paper_json=None,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

        if progress_callback is not None:
            progress_callback(index, len(pdfs), pdf, "completed")

    catalog_result = rebuild_catalog(normalized_config.project_root)
    finished = utc_now()
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    report_path = (
        normalized_config.project_root / "data" / "knowledge" / "last_batch_report.json"
    )

    report = BatchParseReport(
        run_id=run_id,
        input_source=input_source,
        recursive=recursive,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        total_count=len(items),
        success_count=sum(item.status == "success" for item in items),
        reused_count=sum(item.status == "reused" for item in items),
        failed_count=sum(item.status == "failed" for item in items),
        catalog_record_count=len(catalog_result.records),
        catalog_issues=catalog_result.issues,
        report_path=report_path,
        items=items,
    )
    write_json_atomic(report_path, report.to_dict())
    return report
