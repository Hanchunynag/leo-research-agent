"""LEO Research Agent 的单一命令入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
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


def add_reranker_options(command: argparse.ArgumentParser) -> None:
    """为 RRF 精排与评测提供同一组 Cross-Encoder 参数。"""

    command.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    command.add_argument(
        "--reranker-revision",
        help="Reranker 的精确 revision/commit SHA。",
    )
    command.add_argument(
        "--reranker-device",
        help="默认沿用 --device。",
    )
    command.add_argument("--reranker-batch-size", type=int, default=4)
    command.add_argument("--reranker-max-length", type=int, default=1024)


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
    from app.agentic.config import AgenticRAGConfig

    agentic_defaults = AgenticRAGConfig.from_environment(PROJECT_ROOT / ".env")
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

    rerank_command = subparsers.add_parser(
        "rerank",
        help="使用 BGE Cross-Encoder 精排 RRF Top-20。",
    )
    rerank_subparsers = rerank_command.add_subparsers(
        dest="rerank_command",
        required=True,
    )
    rerank_search_command = rerank_subparsers.add_parser(
        "search",
        help="召回 RRF 候选并执行 Cross-Encoder 精排。",
    )
    rerank_search_command.add_argument("query")
    rerank_search_command.add_argument("--limit", type=int, default=10)
    rerank_search_command.add_argument("--work-id")
    rerank_search_command.add_argument("--document-id")
    rerank_search_command.add_argument("--max-chunks-per-work", type=int, default=2)
    rerank_search_command.add_argument("--candidate-limit", type=int, default=20)
    rerank_search_command.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(rerank_search_command)
    add_reranker_options(rerank_search_command)

    context_command = subparsers.add_parser(
        "context",
        help="把检索结果组装为带稳定来源标记的 LLM 证据包。",
    )
    context_subparsers = context_command.add_subparsers(
        dest="context_command",
        required=True,
    )
    context_build_command = context_subparsers.add_parser(
        "build",
        help="按 token budget 构建 EvidenceItem/ContextBundle。",
    )
    context_build_command.add_argument("query")
    context_build_command.add_argument(
        "--mode",
        choices=["fast", "accurate"],
        default="fast",
    )
    context_build_command.add_argument("--retrieval-limit", type=int, default=10)
    context_build_command.add_argument("--token-budget", type=int, default=6000)
    context_build_command.add_argument("--max-evidence", type=int, default=8)
    context_build_command.add_argument("--max-evidence-per-work", type=int, default=2)
    context_build_command.add_argument("--work-id")
    context_build_command.add_argument("--document-id")
    context_build_command.add_argument("--candidate-limit", type=int, default=20)
    context_build_command.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(context_build_command)
    add_reranker_options(context_build_command)

    answer_command = subparsers.add_parser(
        "answer",
        help="使用本地 OpenAI-compatible 模型生成逐条引用并可失败关闭的回答。",
    )
    answer_command.add_argument("query")
    answer_command.add_argument(
        "--mode",
        choices=["fast", "accurate"],
        default="fast",
    )
    answer_command.add_argument(
        "--retrieval-mode",
        choices=["fast", "agentic"],
        default="fast",
        help="fast 保持原单轮流程；agentic 启用 Session、多轮检索和语义验证。",
    )
    answer_command.add_argument("--retrieval-limit", type=int, default=10)
    answer_command.add_argument("--token-budget", type=int, default=6000)
    answer_command.add_argument("--max-evidence", type=int, default=8)
    answer_command.add_argument("--max-evidence-per-work", type=int, default=2)
    answer_command.add_argument("--work-id")
    answer_command.add_argument("--document-id")
    answer_command.add_argument(
        "--candidate-limit", type=int, default=agentic_defaults.candidate_limit
    )
    answer_command.add_argument(
        "--rerank-top-k", type=int, default=agentic_defaults.rerank_top_k
    )
    answer_command.add_argument(
        "--final-top-k", type=int, default=agentic_defaults.final_top_k
    )
    answer_command.add_argument(
        "--max-retrieval-rounds",
        type=int,
        default=agentic_defaults.max_retrieval_rounds,
    )
    answer_command.add_argument(
        "--max-structure-repairs",
        type=int,
        choices=[0, 1],
        default=agentic_defaults.max_structure_repairs,
        help="每个结构化 LLM 阶段允许的 JSON 修复次数，默认 1。",
    )
    answer_command.add_argument(
        "--max-answer-repairs",
        type=int,
        choices=[0, 1],
        default=agentic_defaults.max_answer_repairs,
        help="整次运行允许的 Answer Repair 次数，默认 1。",
    )
    answer_command.add_argument(
        "--max-total-latency-ms",
        type=int,
        default=agentic_defaults.max_total_latency_ms,
        help="可选 Harness 总时限；默认不设置硬时限。",
    )
    answer_command.add_argument(
        "--rrf-k", type=int, default=agentic_defaults.rrf_k
    )
    answer_command.add_argument(
        "--llm-base-url",
        help=(
            "临时覆盖 LEO_LLM_BASE_URL；也可配置在项目根目录 .env。"
        ),
    )
    answer_command.add_argument(
        "--llm-model",
        help="临时覆盖 LEO_LLM_MODEL。",
    )
    answer_command.add_argument(
        "--llm-api-key",
        help="临时覆盖本地密钥；推荐改用 .env，避免进入 shell history。",
    )
    answer_command.add_argument("--llm-timeout", type=float)
    answer_command.add_argument("--llm-max-tokens", type=int)
    answer_command.add_argument(
        "--prompt-layout",
        choices=["query_first", "context_first"],
        help=(
            "Prompt 顺序；动态问答默认 query_first，Context Session 默认 "
            "context_first。"
        ),
    )
    answer_command.add_argument(
        "--context-session",
        help="创建或复用一个固定 ContextBundle 快照，例如 leo_timing。",
    )
    answer_command.add_argument(
        "--session-id",
        help="Agentic Session ID；省略时自动创建并在输出中返回。",
    )
    answer_command.add_argument(
        "--force-new-topic",
        action="store_true",
        help="在现有 Agentic Session 中强制创建独立 Topic。",
    )
    answer_command.add_argument(
        "--disable-reranker",
        action="store_true",
        default=not agentic_defaults.reranker_enabled,
        help="Agentic 模式禁用 Cross-Encoder，并显式回退到 RRF。",
    )
    answer_command.add_argument(
        "--enable-reranker",
        action="store_false",
        dest="disable_reranker",
        default=not agentic_defaults.reranker_enabled,
        help="覆盖环境配置并启用 Agentic Cross-Encoder。",
    )
    answer_command.add_argument(
        "--disable-semantic-validation",
        action="store_true",
        default=not agentic_defaults.semantic_validation_enabled,
        help="只保留结构校验；用于受控消融实验。",
    )
    answer_command.add_argument(
        "--enable-semantic-validation",
        action="store_false",
        dest="disable_semantic_validation",
        default=not agentic_defaults.semantic_validation_enabled,
        help="覆盖环境配置并启用 Claim-Citation 语义验证。",
    )
    answer_command.add_argument(
        "--session-db-path",
        type=Path,
        default=agentic_defaults.session_db_path,
        help="覆盖本地 Agentic SQLite 路径。",
    )
    answer_command.add_argument(
        "--context-compaction-threshold",
        type=float,
        default=agentic_defaults.context_compaction_threshold,
        help="达到模型窗口比例后追加 Compaction 事件，默认 0.70。",
    )
    answer_command.add_argument(
        "--same-topic-threshold",
        type=float,
        default=agentic_defaults.same_topic_threshold,
        help="Topic Router 同主题阈值，默认 0.75。",
    )
    answer_command.add_argument(
        "--new-topic-threshold",
        type=float,
        default=agentic_defaults.new_topic_threshold,
        help="Topic Router 新主题阈值，默认 0.45。",
    )
    answer_command.add_argument(
        "--model-context-window",
        type=int,
        default=agentic_defaults.model_context_window,
        help="用于自动 Compaction 的模型上下文窗口，默认 32768。",
    )
    answer_command.add_argument(
        "--recent-events-after-compaction",
        type=int,
        default=agentic_defaults.recent_events_after_compaction,
        help="Compaction 保留的最近原始事件数，默认 8。",
    )
    answer_command.add_argument(
        "--semantic-weight",
        type=float,
        default=agentic_defaults.semantic_weight,
        help="Topic Router 语义相似度权重，默认 0.40。",
    )
    answer_command.add_argument(
        "--entity-weight",
        type=float,
        default=agentic_defaults.entity_weight,
        help="Topic Router 实体重合权重，默认 0.25。",
    )
    answer_command.add_argument(
        "--context-dependency-weight",
        type=float,
        default=agentic_defaults.context_dependency_weight,
        help="Topic Router 上下文依赖权重，默认 0.20。",
    )
    answer_command.add_argument(
        "--evidence-overlap-weight",
        type=float,
        default=agentic_defaults.evidence_overlap_weight,
        help="Topic Router 证据重合权重，默认 0.15。",
    )
    answer_command.add_argument(
        "--refresh-context-session",
        action="store_true",
        help="用当前问题重新检索并显式覆盖指定 Context Session。",
    )
    answer_command.add_argument(
        "--include-context",
        action="store_true",
        help="调试时在 JSON 中包含完整 ContextBundle；默认省略。",
    )
    add_embedding_options(answer_command)
    add_reranker_options(answer_command)
    answer_command.set_defaults(
        agentic_allow_model_downloads=agentic_defaults.allow_model_downloads,
    )

    session_command = subparsers.add_parser(
        "session",
        help="查看、列出或压缩本地 Agentic Session。",
    )
    session_command.add_argument(
        "--session-db-path",
        type=Path,
        default=agentic_defaults.session_db_path,
    )
    session_subparsers = session_command.add_subparsers(
        dest="session_command",
        required=True,
    )
    session_subparsers.add_parser("list", help="列出全部本地 Session。")
    for action, help_text in (
        ("show", "查看 Session、Topic 和事件/证据数量。"),
        ("evidence", "查看 Topic Evidence Registry。"),
        ("compact", "为活动 Topic 手动追加 Compaction。"),
    ):
        command = session_subparsers.add_parser(action, help=help_text)
        command.add_argument("session_id")

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
        help="在同一人工标注问题集上评测召回、候选池或精排。",
    )
    retrieval_evaluate_command.add_argument(
        "--retriever",
        choices=["bm25", "dense", "rrf", "oracle", "reranker"],
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
    add_reranker_options(retrieval_evaluate_command)

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


def reranker_provider_from_args(args: argparse.Namespace) -> Any:
    """延迟创建 Cross-Encoder，普通召回命令不加载 Reranker。"""

    from app.reranking.bge import BGERerankerConfig, BGERerankerProvider

    return BGERerankerProvider(
        BGERerankerConfig(
            model_name=args.reranker_model,
            revision=args.reranker_revision,
            device=args.reranker_device or args.device,
            cache_folder=args.model_cache,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            local_files_only=args.local_files_only,
            show_progress_bar=not args.no_progress,
        )
    )


def answer_provider_from_args(args: argparse.Namespace) -> Any:
    """按 CLI > 环境变量 > `.env` 的顺序创建回答模型。"""

    from app.generation.openai_compatible import (
        OpenAICompatibleAnswerProvider,
        OpenAICompatibleConfig,
    )
    from app.generation.settings import load_local_llm_settings

    settings = load_local_llm_settings(PROJECT_ROOT)
    base_url = args.llm_base_url or settings.base_url
    model = args.llm_model or settings.model
    missing = [
        name
        for name, value in (
            ("LEO_LLM_BASE_URL", base_url),
            ("LEO_LLM_MODEL", model),
        )
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        joined = "、".join(missing)
        raise ValueError(
            f"缺少 {joined}；请复制 .env.example 为 .env 后填写。"
        )
    assert isinstance(base_url, str)
    assert isinstance(model, str)
    settings_api_key = (
        settings.api_key.get_secret_value() if settings.api_key is not None else None
    )
    api_key = (
        args.llm_api_key
        if args.llm_api_key is not None
        else settings_api_key
    )
    normalized_api_key = api_key.strip() if api_key is not None else None
    prompt_layout = (
        args.prompt_layout
        or settings.prompt_layout
        or ("context_first" if args.context_session else "query_first")
    )

    return OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig(
            base_url=base_url.strip(),
            model=model.strip(),
            api_key=normalized_api_key or None,
            timeout_seconds=(
                args.llm_timeout
                if args.llm_timeout is not None
                else settings.timeout_seconds
            ),
            max_tokens=(
                args.llm_max_tokens
                if args.llm_max_tokens is not None
                else settings.max_tokens
            ),
            prompt_layout=prompt_layout,
        )
    )


def retrieval_runtime_from_args(
    args: argparse.Namespace,
    *,
    include_reranker: bool,
) -> Any:
    from app.runtime.retrieval import RetrievalRuntime

    return RetrievalRuntime(
        project_root=PROJECT_ROOT,
        embedding_provider=dense_provider_from_args(args),
        reranker_provider=(
            reranker_provider_from_args(args) if include_reranker else None
        ),
    )


def agentic_service_from_args(args: argparse.Namespace, answer_provider: Any) -> Any:
    """延迟组装 Agentic 编排层，fast 模式不导入这些模块。"""

    from app.agentic.config import AgenticRAGConfig
    from app.agentic.provider import OpenAIAgenticReasoningProvider
    from app.agentic.reranking import DirectAnswerReranker
    from app.agentic.service import AgenticRAGService
    from app.agentic.store import AgenticSessionStore

    runtime = retrieval_runtime_from_args(
        args,
        include_reranker=not args.disable_reranker,
    )
    config = AgenticRAGConfig(
        candidate_limit=args.candidate_limit,
        rerank_top_k=args.rerank_top_k,
        final_top_k=args.final_top_k,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_structure_repairs=args.max_structure_repairs,
        max_answer_repairs=args.max_answer_repairs,
        max_total_latency_ms=args.max_total_latency_ms,
        fail_closed=True,
        allow_model_downloads=(
            args.agentic_allow_model_downloads and not args.local_files_only
        ),
        rrf_k=args.rrf_k,
        reranker_enabled=not args.disable_reranker,
        semantic_validation_enabled=not args.disable_semantic_validation,
        same_topic_threshold=args.same_topic_threshold,
        new_topic_threshold=args.new_topic_threshold,
        semantic_weight=args.semantic_weight,
        entity_weight=args.entity_weight,
        context_dependency_weight=args.context_dependency_weight,
        evidence_overlap_weight=args.evidence_overlap_weight,
        context_compaction_threshold=args.context_compaction_threshold,
        model_context_window=args.model_context_window,
        recent_events_after_compaction=args.recent_events_after_compaction,
        session_db_path=args.session_db_path,
    )
    store = AgenticSessionStore(
        PROJECT_ROOT,
        database_path=args.session_db_path,
    )
    return AgenticRAGService(
        runtime,
        OpenAIAgenticReasoningProvider(
            answer_provider,
            max_structure_repairs=config.max_structure_repairs,
        ),
        store,
        DirectAnswerReranker(
            runtime.reranker_provider,
            enabled=not args.disable_reranker,
        ),
        config,
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

    if args.command == "session":
        from app.agentic.prompting import compact_topic
        from app.agentic.store import AgenticSessionStore

        agentic_store = AgenticSessionStore(
            PROJECT_ROOT,
            database_path=args.session_db_path,
        )
        try:
            if args.session_command == "list":
                print_json({"sessions": agentic_store.list_sessions()})
                return
            if args.session_command == "show":
                print_json(agentic_store.session_details(args.session_id))
                return
            if args.session_command == "evidence":
                session_record = agentic_store.get_session(args.session_id)
                print_json(
                    {
                        "session_id": args.session_id,
                        "active_topic_id": session_record.get("active_topic_id"),
                        "evidence": agentic_store.list_evidence(args.session_id),
                    }
                )
                return
            session_record = agentic_store.get_session(args.session_id)
            topic_id = session_record.get("active_topic_id")
            if not isinstance(topic_id, str) or not topic_id:
                raise ValueError("Session 没有可压缩的活动 Topic。")
            compaction_report = compact_topic(
                agentic_store,
                args.session_id,
                topic_id,
            )
            print_json(compaction_report.model_dump(mode="json"))
            return
        except (KeyError, OSError, ValueError) as error:
            raise SystemExit(f"Session 错误：{error}") from error

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

    if args.command == "rerank":
        from app.retrieval.reranked import search_reranked_evidence

        print_json(
            search_reranked_evidence(
                project_root=PROJECT_ROOT,
                embedding_provider=dense_provider_from_args(args),
                reranker_provider=reranker_provider_from_args(args),
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

    if args.command == "context":
        runtime = retrieval_runtime_from_args(
            args,
            include_reranker=args.mode == "accurate",
        )
        bundle = runtime.build_context(
            query=args.query,
            mode=args.mode,
            retrieval_limit=args.retrieval_limit,
            token_budget=args.token_budget,
            max_evidence=args.max_evidence,
            max_evidence_per_work=args.max_evidence_per_work,
            work_id=args.work_id,
            document_id=args.document_id,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
        )
        print_json(bundle.to_dict())
        return

    if args.command == "answer":
        from app.generation.service import GroundedAnswerService
        from app.generation.security import redact_sensitive_text

        try:
            answer_provider = answer_provider_from_args(args)
        except ValueError as error:
            safe_error = redact_sensitive_text(
                error,
                known_secrets=(getattr(args, "llm_api_key", None),),
            )
            raise SystemExit(f"LLM 配置错误：{safe_error}") from error
        if args.retrieval_mode == "agentic":
            if args.context_session:
                raise SystemExit(
                    "--context-session 属于固定快照模式，不能与 agentic 同时使用；"
                    "请改用 --session-id。"
                )
            try:
                agentic_result = agentic_service_from_args(args, answer_provider).answer(
                    args.query,
                    session_id=args.session_id,
                    force_new_topic=args.force_new_topic,
                    include_context=args.include_context,
                )
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                provider_key = getattr(
                    getattr(answer_provider, "config", None),
                    "api_key",
                    None,
                )
                safe_error = redact_sensitive_text(
                    error,
                    known_secrets=(args.llm_api_key, provider_key),
                )
                raise SystemExit(f"Agentic RAG 错误：{safe_error}") from error
            print_json(agentic_result)
            if not agentic_result.get("answerable"):
                raise SystemExit(2)
            return
        if args.refresh_context_session and not args.context_session:
            raise SystemExit(
                "--refresh-context-session 必须与 --context-session 一起使用。"
            )
        if args.context_session:
            from app.context.session import ContextSessionStore

            store = ContextSessionStore(PROJECT_ROOT)
            try:
                existed = store.exists(args.context_session)
                if args.refresh_context_session or not existed:
                    runtime = retrieval_runtime_from_args(
                        args,
                        include_reranker=args.mode == "accurate",
                    )
                    context = runtime.build_context(
                        query=args.query,
                        mode=args.mode,
                        retrieval_limit=args.retrieval_limit,
                        token_budget=args.token_budget,
                        max_evidence=args.max_evidence,
                        max_evidence_per_work=args.max_evidence_per_work,
                        work_id=args.work_id,
                        document_id=args.document_id,
                        candidate_limit=args.candidate_limit,
                        rrf_k=args.rrf_k,
                    )
                    session = store.save(args.context_session, context)
                    session_state = "refreshed" if existed else "created"
                else:
                    runtime = None
                    session = store.load(args.context_session)
                    context = session.context
                    session_state = "reused"
            except (OSError, ValueError) as error:
                raise SystemExit(f"Context Session 错误：{error}") from error
            service = GroundedAnswerService(runtime, answer_provider)
            answer = service.answer_from_context(context, query=args.query)
            diagnostics = dict(answer.diagnostics)
            diagnostics["context_session"] = {
                "session_id": session.session_id,
                "state": session_state,
                "context_hash": session.context_hash,
                "source_query": session.context.query,
                "retrieval_skipped": session_state == "reused",
            }
            answer = replace(answer, diagnostics=diagnostics)
        else:
            runtime = retrieval_runtime_from_args(
                args,
                include_reranker=args.mode == "accurate",
            )
            service = GroundedAnswerService(runtime, answer_provider)
            answer = service.answer(
                query=args.query,
                mode=args.mode,
                retrieval_limit=args.retrieval_limit,
                token_budget=args.token_budget,
                max_evidence=args.max_evidence,
                max_evidence_per_work=args.max_evidence_per_work,
                work_id=args.work_id,
                document_id=args.document_id,
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
            )
        print_json(answer.to_dict(include_context=args.include_context))
        if not answer.answerable:
            raise SystemExit(2)
        return

    if args.command == "evaluate":
        from app.evaluation.retrieval import (
            evaluate_bm25,
            evaluate_candidate_pool_oracle,
            evaluate_dense,
            evaluate_hybrid_rrf,
            evaluate_reranked,
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
        elif args.retriever == "oracle":
            report = evaluate_candidate_pool_oracle(
                project_root=PROJECT_ROOT,
                questions_path=args.questions,
                provider=dense_provider_from_args(args),
                output_path=output,
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
            )
        elif args.retriever == "reranker":
            report = evaluate_reranked(
                project_root=PROJECT_ROOT,
                questions_path=args.questions,
                embedding_provider=dense_provider_from_args(args),
                reranker_provider=reranker_provider_from_args(args),
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
