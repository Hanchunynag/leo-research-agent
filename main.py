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


def add_embedding_options(command: argparse.ArgumentParser) -> None:
    """为 Dense 构建、检索和评测提供同一组模型参数。"""

    command.add_argument("--model", default="BAAI/bge-m3")
    command.add_argument(
        "--revision",
        help="模型的精确 revision/commit SHA；正式基线应显式固定。",
    )
    command.add_argument("--device", help="例如 cpu、mps 或 cuda。")
    command.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "huggingface",
        help="Hugging Face 模型缓存目录。",
    )
    command.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
    )
    command.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用本地 Hugging Face 缓存，不访问网络。",
    )
    command.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭模型编码进度条。",
    )


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

    knowledge_command = subparsers.add_parser(
        "knowledge",
        help="从 canonical paper.json 构建论文结构、Chunk 和本地索引。",
    )
    knowledge_subparsers = knowledge_command.add_subparsers(
        dest="knowledge_command",
        required=True,
    )
    knowledge_build_command = knowledge_subparsers.add_parser(
        "build",
        help="增量构建结构化知识层和 BM25 索引。",
    )
    knowledge_build_command.add_argument(
        "--force",
        action="store_true",
        help="忽略输入指纹，重新生成每篇论文的结构和 Chunk。",
    )
    knowledge_build_command.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="单个 Chunk 的最大近似词元数，默认 700。",
    )
    knowledge_build_command.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=80,
        help="可吸收到直属子章节上下文的小 Chunk 阈值，默认 80。",
    )
    knowledge_build_command.add_argument(
        "--overlap-tokens",
        type=int,
        default=80,
        help="同一章节连续 Chunk 的最大重叠上下文词元数，默认 80。",
    )

    search_command = subparsers.add_parser(
        "search",
        help="从本地 BM25 索引检索带页码和 block 来源的论文证据。",
    )
    search_command.add_argument("query")
    search_command.add_argument("--limit", type=int, default=10)
    search_command.add_argument("--work-id")
    search_command.add_argument("--document-id")
    search_command.add_argument(
        "--max-chunks-per-work",
        type=int,
        default=2,
        help="每个逻辑论文最多返回的 Chunk 数，默认 2。",
    )

    dense_command = subparsers.add_parser(
        "dense",
        help="构建和查询 BGE-M3 单向量 Qdrant local 索引。",
    )
    dense_subparsers = dense_command.add_subparsers(
        dest="dense_command",
        required=True,
    )
    dense_build_command = dense_subparsers.add_parser(
        "build",
        help="按 Manifest 构建或复用 Dense 索引。",
    )
    dense_build_command.add_argument("--force", action="store_true")
    add_embedding_options(dense_build_command)
    dense_search_command = dense_subparsers.add_parser(
        "search",
        help="从 Dense 索引检索带来源信息的论文证据。",
    )
    dense_search_command.add_argument("query")
    dense_search_command.add_argument("--limit", type=int, default=10)
    dense_search_command.add_argument("--work-id")
    dense_search_command.add_argument("--document-id")
    dense_search_command.add_argument("--max-chunks-per-work", type=int, default=2)
    add_embedding_options(dense_search_command)

    hybrid_command = subparsers.add_parser(
        "hybrid",
        help="使用 RRF 融合 BM25 与 Dense 候选。",
    )
    hybrid_subparsers = hybrid_command.add_subparsers(
        dest="hybrid_command",
        required=True,
    )
    hybrid_search_command = hybrid_subparsers.add_parser(
        "search",
        help="执行 BM25 Top-20 + Dense Top-20 的 RRF 检索。",
    )
    hybrid_search_command.add_argument("query")
    hybrid_search_command.add_argument("--limit", type=int, default=10)
    hybrid_search_command.add_argument("--work-id")
    hybrid_search_command.add_argument("--document-id")
    hybrid_search_command.add_argument("--max-chunks-per-work", type=int, default=2)
    hybrid_search_command.add_argument("--candidate-limit", type=int, default=20)
    hybrid_search_command.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(hybrid_search_command)

    evaluate_command = subparsers.add_parser(
        "evaluate",
        help="运行本地检索与后续 RAG 质量评测。",
    )
    evaluate_subparsers = evaluate_command.add_subparsers(
        dest="evaluate_command",
        required=True,
    )
    retrieval_evaluate_command = evaluate_subparsers.add_parser(
        "retrieval",
        help="在同一人工标注问题集上评测 BM25 或 Dense。",
    )
    retrieval_evaluate_command.add_argument(
        "--retriever",
        choices=["bm25", "dense", "rrf"],
        default="bm25",
    )
    retrieval_evaluate_command.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.jsonl",
    )
    retrieval_evaluate_command.add_argument(
        "--output",
        type=Path,
        help="报告路径；默认按 retriever 写入 data/evaluation。",
    )
    retrieval_evaluate_command.add_argument(
        "--k-values",
        default="1,5,10",
        help="逗号分隔的 Recall@K 列表，最大值同时用于 nDCG，默认 1,5,10。",
    )
    retrieval_evaluate_command.add_argument("--candidate-limit", type=int, default=20)
    retrieval_evaluate_command.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(retrieval_evaluate_command)

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


def parse_k_values(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--k-values 必须是逗号分隔的整数。") from error
    if not values or any(item < 1 or item > 100 for item in values):
        raise ValueError("--k-values 必须在 1 到 100 之间。")
    return sorted(set(values))


def dense_provider_from_args(args: argparse.Namespace) -> Any:
    """延迟创建模型适配器，避免 BM25 命令加载模型。"""

    from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider

    return BGEM3EmbeddingProvider(
        BGEM3Config(
            model_name=args.model,
            revision=args.revision,
            device=args.device,
            cache_folder=args.model_cache,
            batch_size=args.embedding_batch_size,
            local_files_only=args.local_files_only,
            show_progress_bar=not args.no_progress,
        )
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

    if args.command == "knowledge":
        from app.chunking.builder import build_knowledge_base

        knowledge_report = build_knowledge_base(
            project_root=PROJECT_ROOT,
            force=args.force,
            maximum_tokens=args.max_tokens,
            minimum_chunk_tokens=args.min_chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        print_json(knowledge_report.to_dict())
        if knowledge_report.issues:
            raise SystemExit(1)
        return

    if args.command == "search":
        from app.retrieval.search import search_evidence

        print_json(
            search_evidence(
                project_root=PROJECT_ROOT,
                query=args.query,
                limit=args.limit,
                work_id=args.work_id,
                document_id=args.document_id,
                max_chunks_per_work=args.max_chunks_per_work,
            )
        )
        return

    if args.command == "dense":
        provider = dense_provider_from_args(args)
        if args.dense_command == "build":
            from app.indexing.dense import build_dense_index

            print_json(
                build_dense_index(
                    project_root=PROJECT_ROOT,
                    provider=provider,
                    force=args.force,
                ).to_dict()
            )
            return

        from app.retrieval.dense import search_dense_evidence

        print_json(
            search_dense_evidence(
                project_root=PROJECT_ROOT,
                provider=provider,
                query=args.query,
                limit=args.limit,
                work_id=args.work_id,
                document_id=args.document_id,
                max_chunks_per_work=args.max_chunks_per_work,
            )
        )
        return

    if args.command == "hybrid":
        from app.retrieval.hybrid import search_hybrid_evidence

        print_json(
            search_hybrid_evidence(
                project_root=PROJECT_ROOT,
                provider=dense_provider_from_args(args),
                query=args.query,
                limit=args.limit,
                work_id=args.work_id,
                document_id=args.document_id,
                max_chunks_per_work=args.max_chunks_per_work,
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
            )
        )
        return

    if args.command == "evaluate":
        from app.evaluation.retrieval import (
            evaluate_bm25,
            evaluate_dense,
            evaluate_hybrid_rrf,
        )

        output = args.output or (
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / f"{args.retriever}_baseline.json"
        )
        if args.retriever == "dense":
            report = evaluate_dense(
                project_root=PROJECT_ROOT,
                questions_path=args.questions,
                provider=dense_provider_from_args(args),
                output_path=output,
                k_values=parse_k_values(args.k_values),
            )
        elif args.retriever == "rrf":
            report = evaluate_hybrid_rrf(
                project_root=PROJECT_ROOT,
                questions_path=args.questions,
                provider=dense_provider_from_args(args),
                output_path=output,
                k_values=parse_k_values(args.k_values),
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
            )
        else:
            report = evaluate_bm25(
                project_root=PROJECT_ROOT,
                questions_path=args.questions,
                output_path=output,
                k_values=parse_k_values(args.k_values),
            )
        print_json(report)
        return

    config = config_from_args(args)

    if args.command == "batch":
        batch_report = batch_parse_directory(
            input_directory=args.directory,
            config=config,
            recursive=args.recursive,
        )
        print_json(batch_report.to_dict())
        if batch_report.failed_count or batch_report.catalog_issues:
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
