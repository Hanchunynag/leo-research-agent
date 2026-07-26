"""LEO Research Agent 的单一命令入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from app.ingestion.batch import batch_parse_directory
from app.knowledge.catalog import (
    library_status,
    load_catalog,
    rebuild_catalog,
)
from app.parsing.pipeline import PaperParseConfig, parse_paper


PROJECT_ROOT = Path(__file__).resolve().parent


def add_mineru_options(command: argparse.ArgumentParser) -> None:
    """为单篇和批量入口添加同一组 MinerU 参数。"""

    command.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
    )
    command.add_argument(
        "--backend",
        choices=[
            "pipeline",
            "vlm-engine",
            "hybrid-engine",
            "vlm-http-client",
            "hybrid-http-client",
        ],
        default="pipeline",
    )
    command.add_argument("--language")
    command.add_argument(
        "--mineru-executable",
        type=Path,
        help="显式指定 MinerU 专用 venv 中的 mineru 命令。",
    )
    command.add_argument(
        "--force-mineru",
        action="store_true",
        help="忽略已有 MinerU 产物并重新解析。",
    )
    command.add_argument(
        "--no-formula",
        action="store_true",
        help="关闭 MinerU 公式解析。",
    )
    command.add_argument(
        "--no-table",
        action="store_true",
        help="关闭 MinerU 表格解析。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地 LEO 论文解析工具。",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    parse_command = subparsers.add_parser(
        "parse",
        help="一次完成 PDF 入库、MinerU 解析和 paper.json 输出。",
    )
    parse_command.add_argument("pdf", type=Path)
    add_mineru_options(parse_command)

    batch_command = subparsers.add_parser(
        "batch",
        help="容错地批量解析目录中的 PDF，并重建 papers.jsonl。",
    )
    batch_command.add_argument("directory", type=Path)
    batch_command.add_argument(
        "--recursive",
        action="store_true",
        help="递归查找子目录中的 PDF。",
    )
    add_mineru_options(batch_command)

    library_command = subparsers.add_parser(
        "library",
        help="重建、查看和检查全局论文目录。",
    )
    library_subparsers = library_command.add_subparsers(
        dest="library_command",
        required=True,
    )
    library_subparsers.add_parser(
        "rebuild",
        help="从全部 canonical paper.json 重建 papers.jsonl。",
    )
    library_subparsers.add_parser(
        "list",
        help="列出 papers.jsonl 中的论文。",
    )
    library_subparsers.add_parser(
        "status",
        help="检查 raw、parsed、canonical 和目录的一致性。",
    )
    library_subparsers.add_parser(
        "works",
        help="列出按 work_id 归并的逻辑论文及 PDF 版本。",
    )

    subparsers.add_parser(
        "ui",
        help="启动本地 Gradio 页面。",
    )

    subparsers.add_parser(
        "academic-mcp",
        help="通过 stdio 启动外部学术搜索与开放全文 MCP。",
    )

    metadata_command = subparsers.add_parser(
        "metadata",
        help="通过外部 MCP 核验本地论文标题和元数据。",
    )
    metadata_subparsers = metadata_command.add_subparsers(
        dest="metadata_command",
        required=True,
    )
    for action, help_text in (
        ("resolve", "只返回外部候选，不修改本地文件。"),
        ("enrich", "严格自动核验并合并元数据，保存候选报告。"),
        ("normalize", "为已有核验元数据补身份并规范化 raw PDF 文件名。"),
    ):
        action_command = metadata_subparsers.add_parser(
            action,
            help=help_text,
        )
        action_command.add_argument("paper_id")
        if action != "normalize":
            action_command.add_argument("--limit", type=int, default=5)
        if action == "enrich":
            action_command.add_argument(
                "--candidate-index",
                type=int,
                help="显式选择从 0 开始的候选序号；省略时使用严格自动规则。",
            )

    return parser


def print_json(payload: Any) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def config_from_args(args: argparse.Namespace) -> PaperParseConfig:
    return PaperParseConfig(
        project_root=PROJECT_ROOT,
        mineru_executable=args.mineru_executable,
        method=args.method,
        backend=args.backend,
        language=args.language,
        formula_enabled=not args.no_formula,
        table_enabled=not args.no_table,
        force_mineru=args.force_mineru,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "ui":
        from app.ui.gradio_app import main as run_ui

        run_ui()
        return

    if args.command == "academic-mcp":
        from app.academic_mcp.server import main as run_academic_mcp

        run_academic_mcp([])
        return

    if args.command == "library":
        if args.library_command == "rebuild":
            build_result = rebuild_catalog(PROJECT_ROOT)
            print_json(build_result.summary())
            if build_result.issues:
                raise SystemExit(1)
            return

        if args.library_command == "list":
            load_result = load_catalog(PROJECT_ROOT)
            print_json(
                {
                    "catalog_path": str(load_result.catalog_path),
                    "record_count": len(load_result.records),
                    "issue_count": len(load_result.issues),
                    "issues": [asdict(issue) for issue in load_result.issues],
                    "records": [record.to_dict() for record in load_result.records],
                }
            )
            if load_result.issues:
                raise SystemExit(1)
            return

        if args.library_command == "works":
            from app.knowledge.works import scan_work_catalog

            records, issues, unresolved = scan_work_catalog(PROJECT_ROOT)
            print_json(
                {
                    "record_count": len(records),
                    "records": [record.to_dict() for record in records],
                    "unresolved_paper_ids": unresolved,
                    "issues": [asdict(issue) for issue in issues],
                }
            )
            if issues:
                raise SystemExit(1)
            return

        status = library_status(PROJECT_ROOT)
        print_json(status.to_dict())
        if not status.catalog_consistent:
            raise SystemExit(1)
        return

    if args.command == "metadata":
        from app.academic_mcp.client import AcademicMCPClient
        from app.knowledge.metadata_enrichment import (
            enrich_paper_metadata,
            normalize_verified_paper,
        )

        if args.metadata_command == "normalize":
            print_json(
                normalize_verified_paper(
                    project_root=PROJECT_ROOT,
                    paper_id=args.paper_id,
                )
            )
            return

        result = asyncio.run(
            enrich_paper_metadata(
                project_root=PROJECT_ROOT,
                paper_id=args.paper_id,
                resolver=AcademicMCPClient(PROJECT_ROOT),
                limit=args.limit,
                selected_index=getattr(args, "candidate_index", None),
                apply=args.metadata_command == "enrich",
            )
        )
        print_json(result.to_dict())
        if args.metadata_command == "enrich" and not result.updated:
            raise SystemExit(2)
        return

    config = config_from_args(args)

    if args.command == "batch":
        report = batch_parse_directory(
            input_directory=args.directory,
            config=config,
            recursive=args.recursive,
        )
        print_json(report.to_dict())
        if report.failed_count or report.catalog_issues:
            raise SystemExit(1)
        return

    parse_result = parse_paper(
        input_path=args.pdf,
        config=config,
    )
    rebuild_catalog(PROJECT_ROOT)

    print_json(asdict(parse_result))


if __name__ == "__main__":
    main()
