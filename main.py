"""LEO Research Agent 的单一命令入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from app.parsing.pipeline import PaperParseConfig, parse_paper


PROJECT_ROOT = Path(__file__).resolve().parent


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
    parse_command.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
    )
    parse_command.add_argument(
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
    parse_command.add_argument("--language")
    parse_command.add_argument(
        "--mineru-executable",
        type=Path,
        help="显式指定 MinerU 专用 venv 中的 mineru 命令。",
    )
    parse_command.add_argument(
        "--force-mineru",
        action="store_true",
        help="忽略已有 MinerU 产物并重新解析。",
    )
    parse_command.add_argument(
        "--no-formula",
        action="store_true",
        help="关闭 MinerU 公式解析。",
    )
    parse_command.add_argument(
        "--no-table",
        action="store_true",
        help="关闭 MinerU 表格解析。",
    )

    subparsers.add_parser(
        "ui",
        help="启动本地 Gradio 页面。",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "ui":
        from app.ui.gradio_app import main as run_ui

        run_ui()
        return

    config = PaperParseConfig(
        project_root=PROJECT_ROOT,
        mineru_executable=args.mineru_executable,
        method=args.method,
        backend=args.backend,
        language=args.language,
        formula_enabled=not args.no_formula,
        table_enabled=not args.no_table,
        force_mineru=args.force_mineru,
    )
    result = parse_paper(
        input_path=args.pdf,
        config=config,
    )

    print(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
