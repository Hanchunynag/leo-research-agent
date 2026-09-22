"""Single CLI entry point for corpus operations and CrewAI Scholar Runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from app.ingestion.batch import batch_parse_directory
from app.knowledge.catalog import library_status, load_catalog, rebuild_catalog
from app.parsing.pipeline import PaperParseConfig, parse_paper

PROJECT_ROOT = Path(__file__).resolve().parent


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def add_embedding_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", default="BAAI/bge-m3")
    command.add_argument("--revision")
    command.add_argument("--device")
    command.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "data" / "models" / "huggingface")
    command.add_argument("--embedding-batch-size", type=int, default=8)
    command.add_argument("--local-files-only", action="store_true")
    command.add_argument("--no-progress", action="store_true")


def add_reranker_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    command.add_argument("--reranker-revision")
    command.add_argument("--reranker-device")
    command.add_argument("--reranker-batch-size", type=int, default=4)
    command.add_argument("--reranker-max-length", type=int, default=1024)


def add_mineru_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto")
    command.add_argument("--backend", choices=["pipeline", "vlm-engine", "hybrid-engine", "vlm-http-client", "hybrid-http-client"], default="pipeline")
    command.add_argument("--language")
    command.add_argument("--mineru-executable", type=Path)
    command.add_argument("--force-mineru", action="store_true")
    command.add_argument("--no-formula", action="store_true")
    command.add_argument("--no-table", action="store_true")
    command.add_argument("--paddleocr-executable", type=Path)
    command.add_argument("--no-table-recovery", action="store_true")
    command.add_argument("--no-formula-recovery", action="store_true")


def _task_type_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--task-type", choices=["RESEARCH", "SUPPORT_CLAIM", "WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT", "REVIEW"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 LEO 论文知识库与 CrewAI Scholar 生产入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_command = subparsers.add_parser("parse", help="解析一篇 PDF 并写入 canonical corpus。")
    parse_command.add_argument("pdf", type=Path)
    add_mineru_options(parse_command)
    batch_command = subparsers.add_parser("batch", help="批量解析目录中的 PDF。")
    batch_command.add_argument("directory", type=Path)
    batch_command.add_argument("--recursive", action="store_true")
    add_mineru_options(batch_command)

    library = subparsers.add_parser("library", help="查看和重建论文目录。")
    library_sub = library.add_subparsers(dest="library_command", required=True)
    library_sub.add_parser("rebuild")
    library_sub.add_parser("list")
    library_sub.add_parser("status")
    library_sub.add_parser("works")

    web = subparsers.add_parser("web", help="启动 FastAPI + React 生产入口。")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    subparsers.add_parser("ui", help="启动本地 Gradio 页面。")

    jobs = subparsers.add_parser("jobs", help="查看、取消或执行持久化 Provider/Scholar Job。")
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True)
    status = jobs_sub.add_parser("status")
    status.add_argument("job_id", nargs="?")
    cancel = jobs_sub.add_parser("cancel")
    cancel.add_argument("job_id")
    work = jobs_sub.add_parser("work")
    work.add_argument("--max-jobs", type=int, default=10)

    subparsers.add_parser("academic-mcp", help="通过 stdio 启动学术搜索 MCP。")
    metadata = subparsers.add_parser("metadata", help="核验本地论文元数据。")
    metadata_sub = metadata.add_subparsers(dest="metadata_command", required=True)
    for action in ("resolve", "enrich", "normalize"):
        item = metadata_sub.add_parser(action)
        item.add_argument("paper_id")
        if action != "normalize":
            item.add_argument("--limit", type=int, default=5)
        if action == "enrich":
            item.add_argument("--candidate-index", type=int)

    knowledge = subparsers.add_parser("knowledge", help="构建和检查结构化知识层。")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    build = knowledge_sub.add_parser("build")
    build.add_argument("--force", action="store_true")
    build.add_argument("--max-tokens", type=int, default=700)
    build.add_argument("--min-chunk-tokens", type=int, default=80)
    build.add_argument("--overlap-tokens", type=int, default=80)
    knowledge_sub.add_parser("status")
    database = knowledge_sub.add_parser("database")
    database.add_argument("database_action", choices=["init", "status"])

    search = subparsers.add_parser("search", help="检索本地论文证据。")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--work-id")
    search.add_argument("--document-id")
    search.add_argument("--max-chunks-per-work", type=int, default=2)

    dense = subparsers.add_parser("dense")
    dense_sub = dense.add_subparsers(dest="dense_command", required=True)
    dense_build = dense_sub.add_parser("build")
    dense_build.add_argument("--force", action="store_true")
    add_embedding_options(dense_build)
    dense_search = dense_sub.add_parser("search")
    dense_search.add_argument("query")
    dense_search.add_argument("--limit", type=int, default=10)
    dense_search.add_argument("--work-id")
    dense_search.add_argument("--document-id")
    dense_search.add_argument("--max-chunks-per-work", type=int, default=2)
    add_embedding_options(dense_search)

    hybrid = subparsers.add_parser("hybrid")
    hybrid_sub = hybrid.add_subparsers(dest="hybrid_command", required=True)
    hybrid_search = hybrid_sub.add_parser("search")
    hybrid_search.add_argument("query")
    hybrid_search.add_argument("--limit", type=int, default=10)
    hybrid_search.add_argument("--work-id")
    hybrid_search.add_argument("--document-id")
    hybrid_search.add_argument("--max-chunks-per-work", type=int, default=2)
    hybrid_search.add_argument("--candidate-limit", type=int, default=20)
    hybrid_search.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(hybrid_search)

    hierarchical = subparsers.add_parser("hierarchical")
    hierarchical_sub = hierarchical.add_subparsers(dest="hierarchical_command", required=True)
    hierarchical_build = hierarchical_sub.add_parser("build")
    hierarchical_build.add_argument("--force", action="store_true")
    hierarchical_build.add_argument("--max-tokens", type=int, default=700)
    hierarchical_build.add_argument("--min-chunk-tokens", type=int, default=80)
    hierarchical_build.add_argument("--overlap-tokens", type=int, default=80)
    add_embedding_options(hierarchical_build)
    hierarchical_search = hierarchical_sub.add_parser("search")
    hierarchical_search.add_argument("query")
    hierarchical_search.add_argument("--limit", type=int, default=10)
    hierarchical_search.add_argument("--paper-limit", type=int, default=10)
    hierarchical_search.add_argument("--paper-candidate-limit", type=int, default=30)
    hierarchical_search.add_argument("--chunk-candidate-limit", type=int, default=40)
    hierarchical_search.add_argument("--max-chunks-per-work", type=int, default=2)
    hierarchical_search.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(hierarchical_search)
    add_reranker_options(hierarchical_search)

    rerank = subparsers.add_parser("rerank")
    rerank_sub = rerank.add_subparsers(dest="rerank_command", required=True)
    rerank_search = rerank_sub.add_parser("search")
    rerank_search.add_argument("query")
    rerank_search.add_argument("--limit", type=int, default=10)
    rerank_search.add_argument("--work-id")
    rerank_search.add_argument("--document-id")
    rerank_search.add_argument("--max-chunks-per-work", type=int, default=2)
    rerank_search.add_argument("--candidate-limit", type=int, default=20)
    rerank_search.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(rerank_search)
    add_reranker_options(rerank_search)

    context = subparsers.add_parser("context")
    context_build = context.add_subparsers(dest="context_command", required=True).add_parser("build")
    context_build.add_argument("query")
    context_build.add_argument("--mode", choices=["fast", "accurate", "hierarchical"], default="fast")
    context_build.add_argument("--retrieval-limit", type=int, default=10)
    context_build.add_argument("--token-budget", type=int, default=6000)
    context_build.add_argument("--max-evidence", type=int, default=8)
    context_build.add_argument("--max-evidence-per-work", type=int, default=2)
    context_build.add_argument("--work-id")
    context_build.add_argument("--document-id")
    context_build.add_argument("--candidate-limit", type=int, default=20)
    context_build.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(context_build)
    add_reranker_options(context_build)

    scholar = subparsers.add_parser("scholar", help="唯一 CrewAI Scholar Run 入口。")
    scholar_sub = scholar.add_subparsers(dest="scholar_command", required=True)
    release = scholar_sub.add_parser("release-check")
    request = scholar_sub.add_parser("request")
    request.add_argument("instruction")
    request.add_argument("--project-id", required=True)
    _task_type_options(request)
    request.add_argument("--session-id")
    request.add_argument("--thread-id")
    request.add_argument("--idempotency-key")
    resume = scholar_sub.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--resume-value", default="null")
    resume.add_argument("--idempotency-key")
    scholar_status = scholar_sub.add_parser("status")
    scholar_status.add_argument("run_id")
    patch = scholar_sub.add_parser("patch")
    patch_sub = patch.add_subparsers(dest="patch_command", required=True)
    show = patch_sub.add_parser("show")
    show.add_argument("patch_id")
    for action in ("accept", "reject"):
        decision = patch_sub.add_parser(action)
        decision.add_argument("patch_id")
        decision.add_argument("--project-id", required=True)
        decision.add_argument("--expected-base-hash", required=True)
        decision.add_argument("--actor", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    retrieval = evaluate_sub.add_parser("retrieval")
    retrieval.add_argument("--retriever", choices=["bm25", "dense", "rrf", "oracle", "reranker"], default="bm25")
    retrieval.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.jsonl")
    retrieval.add_argument("--output", type=Path)
    retrieval.add_argument("--k-values", default="1,5,10")
    retrieval.add_argument("--candidate-limit", type=int, default=20)
    retrieval.add_argument("--rrf-k", type=int, default=60)
    add_embedding_options(retrieval)
    add_reranker_options(retrieval)
    generation = evaluate_sub.add_parser("generation")
    generation.add_argument("--predictions", type=Path, required=True)
    generation.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "generation_report.json")

    return parser


def config_from_args(args: argparse.Namespace) -> PaperParseConfig:
    return PaperParseConfig(
        project_root=PROJECT_ROOT,
        mineru_executable=args.mineru_executable,
        method=args.method,
        backend=args.backend,
        language=args.language,
        formula_enabled=not args.no_formula,
        table_enabled=not args.no_table,
        paddleocr_executable=args.paddleocr_executable,
        table_recovery_enabled=not args.no_table_recovery,
        formula_recovery_enabled=not args.no_formula_recovery,
        force_mineru=args.force_mineru,
    )


def parse_k_values(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("--k-values 必须是逗号分隔的整数。") from error
    if not values or any(item < 1 or item > 100 for item in values):
        raise ValueError("--k-values 必须在 1 到 100 之间。")
    return values


def dense_provider_from_args(args: argparse.Namespace) -> Any:
    from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
    return BGEM3EmbeddingProvider(BGEM3Config(model_name=args.model, revision=args.revision, device=args.device, cache_folder=args.model_cache, batch_size=args.embedding_batch_size, local_files_only=args.local_files_only, show_progress_bar=not args.no_progress))


def reranker_provider_from_args(args: argparse.Namespace) -> Any:
    from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
    return BGERerankerProvider(BGERerankerConfig(model_name=args.reranker_model, revision=args.reranker_revision, device=args.reranker_device or args.device, cache_folder=args.model_cache, batch_size=args.reranker_batch_size, max_length=args.reranker_max_length, local_files_only=args.local_files_only, show_progress_bar=not args.no_progress))


def retrieval_runtime_from_args(args: argparse.Namespace, *, include_reranker: bool) -> Any:
    from app.runtime.retrieval import RetrievalRuntime
    return RetrievalRuntime(PROJECT_ROOT, dense_provider_from_args(args), reranker_provider_from_args(args) if include_reranker else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "web":
        import uvicorn
        from app.web.api import create_app
        if not 1 <= args.port <= 65_535:
            raise SystemExit("--port 必须在 1 到 65535 之间。")
        uvicorn.run(create_app(PROJECT_ROOT), host=args.host, port=args.port, log_level="info")
        return

    if args.command == "ui":
        from app.ui.gradio_app import main as run_ui
        run_ui()
        return

    if args.command == "jobs":
        from app.jobs import PersistentJobRepository
        repository = PersistentJobRepository(PROJECT_ROOT)
        if args.jobs_command == "status":
            print_json(asdict(repository.get(args.job_id)) if args.job_id else {"jobs": [asdict(value) for value in repository.list()]})
            return
        if args.jobs_command == "cancel":
            print_json(asdict(repository.request_cancel(args.job_id)))
            return
        from app.academic_mcp.service import AcademicDiscoveryService
        from app.scholar.research.providers import build_bootstrap_provider_composition
        backend = AcademicDiscoveryService.create(PROJECT_ROOT)
        composition = build_bootstrap_provider_composition(PROJECT_ROOT, backend, worker_id="cli-provider-worker")
        composition.worker.recover_after_restart()
        completed = composition.worker.run_until_idle(max_jobs=max(1, args.max_jobs))
        asyncio.run(backend.close())
        print_json({"jobs": [asdict(value) for value in completed]})
        return

    if args.command == "academic-mcp":
        from app.academic_mcp.server import main as run_academic_mcp
        run_academic_mcp([])
        return

    if args.command == "scholar":
        from app.scholar.approval import PatchApprovalRequest, PatchApprovalService
        if args.scholar_command == "release-check":
            from app.evaluation.multi_agent import release_gate
            print_json(release_gate(PROJECT_ROOT))
            return
        from app.scholar.runs import ScholarRunManager
        from app.orchestration.contracts import OrchestrationRequest
        manager = ScholarRunManager(PROJECT_ROOT)
        if args.scholar_command == "request":
            contract = OrchestrationRequest(request_id=args.idempotency_key or f"cli:{args.instruction[:80]}", project_id=args.project_id, instruction=args.instruction, session_id=args.session_id, task_type=args.task_type, thread_id=args.thread_id)
            print_json(manager.create(contract, idempotency_key=args.idempotency_key))
            return
        if args.scholar_command == "resume":
            try:
                value = json.loads(args.resume_value)
            except json.JSONDecodeError as error:
                raise SystemExit("--resume-value 必须是合法 JSON。") from error
            print_json(manager.enqueue_resume(args.run_id, resume_value=value, idempotency_key=args.idempotency_key))
            return
        if args.scholar_command == "status":
            print_json(manager.snapshot(args.run_id))
            return
        service = PatchApprovalService(PROJECT_ROOT)
        if args.patch_command == "show":
            print_json(asdict(service.get_preview(args.patch_id)))
            return
        result = service.approve(PatchApprovalRequest(patch_id=args.patch_id, project_id=args.project_id, decision="ACCEPT" if args.patch_command == "accept" else "REJECT", expected_base_hash=args.expected_base_hash, actor=args.actor))
        print_json(asdict(result))
        if result.error_code:
            raise SystemExit(2)
        return

    if args.command == "parse" or args.command == "batch":
        config = config_from_args(args)
        if args.command == "batch":
            report = batch_parse_directory(input_directory=args.directory, config=config, recursive=args.recursive)
            print_json(report.to_dict())
            if report.failed_count or report.catalog_issues:
                raise SystemExit(1)
            return
        result = parse_paper(input_path=args.pdf, config=config)
        rebuild_catalog(PROJECT_ROOT)
        print_json(asdict(result))
        return

    if args.command == "library":
        if args.library_command == "rebuild":
            result = rebuild_catalog(PROJECT_ROOT)
            print_json(result.summary())
            if result.issues:
                raise SystemExit(1)
        elif args.library_command == "list":
            result = load_catalog(PROJECT_ROOT)
            print_json({"catalog_path": str(result.catalog_path), "records": [record.to_dict() for record in result.records], "issues": [asdict(issue) for issue in result.issues]})
        elif args.library_command == "works":
            from app.knowledge.works import scan_work_catalog
            records, issues, unresolved = scan_work_catalog(PROJECT_ROOT)
            print_json({"records": [record.to_dict() for record in records], "issues": [asdict(issue) for issue in issues], "unresolved_paper_ids": unresolved})
        else:
            print_json(library_status(PROJECT_ROOT).to_dict())
        return

    if args.command == "metadata":
        from app.academic_mcp.client import AcademicMCPClient
        from app.knowledge.metadata_enrichment import enrich_paper_metadata, normalize_verified_paper
        if args.metadata_command == "normalize":
            print_json(normalize_verified_paper(project_root=PROJECT_ROOT, paper_id=args.paper_id))
            return
        result = asyncio.run(enrich_paper_metadata(project_root=PROJECT_ROOT, paper_id=args.paper_id, resolver=AcademicMCPClient(PROJECT_ROOT), limit=args.limit, selected_index=getattr(args, "candidate_index", None), apply=args.metadata_command == "enrich"))
        print_json(result.to_dict())
        return

    if args.command == "knowledge":
        if args.knowledge_command == "status":
            from app.knowledge.corpus import knowledge_index_readiness
            from app.knowledge import knowledge_runtime_status
            print_json({**knowledge_runtime_status(PROJECT_ROOT), "knowledge_index": knowledge_index_readiness(PROJECT_ROOT), "corpus": library_status(PROJECT_ROOT).to_dict()})
            return
        if args.knowledge_command == "database":
            from app.persistence import build_knowledge_repository
            repository = build_knowledge_repository(PROJECT_ROOT)
            print_json({"enabled": repository is not None, "action": args.database_action})
            if repository is not None:
                repository.close()
            return
        from app.chunking.builder import build_knowledge_base
        report = build_knowledge_base(PROJECT_ROOT, force=args.force, maximum_tokens=args.max_tokens, minimum_chunk_tokens=args.min_chunk_tokens, overlap_tokens=args.overlap_tokens)
        print_json(report.to_dict())
        return

    if args.command == "search":
        from app.retrieval.search import search_evidence
        print_json(search_evidence(PROJECT_ROOT, args.query, limit=args.limit, work_id=args.work_id, document_id=args.document_id, max_chunks_per_work=args.max_chunks_per_work))
        return
    if args.command == "dense":
        provider = dense_provider_from_args(args)
        if args.dense_command == "build":
            from app.indexing.dense import build_dense_index
            print_json(build_dense_index(PROJECT_ROOT, provider, force=args.force).to_dict())
        else:
            from app.retrieval.dense import search_dense_evidence
            print_json(search_dense_evidence(PROJECT_ROOT, provider, args.query, limit=args.limit, work_id=args.work_id, document_id=args.document_id, max_chunks_per_work=args.max_chunks_per_work))
        return
    if args.command == "hybrid":
        from app.retrieval.hybrid import search_hybrid_evidence
        print_json(search_hybrid_evidence(PROJECT_ROOT, dense_provider_from_args(args), args.query, limit=args.limit, work_id=args.work_id, document_id=args.document_id, max_chunks_per_work=args.max_chunks_per_work, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k))
        return
    if args.command == "hierarchical":
        if args.hierarchical_command == "build":
            from app.indexing.hierarchical import build_hierarchical_indexes
            print_json(build_hierarchical_indexes(PROJECT_ROOT, dense_provider_from_args(args), force=args.force, maximum_tokens=args.max_tokens, minimum_chunk_tokens=args.min_chunk_tokens, overlap_tokens=args.overlap_tokens))
        else:
            from app.retrieval.hierarchical import search_hierarchical_evidence
            print_json(search_hierarchical_evidence(PROJECT_ROOT, dense_provider_from_args(args), args.query, reranker_provider=reranker_provider_from_args(args), limit=args.limit, paper_limit=args.paper_limit, paper_candidate_limit=args.paper_candidate_limit, chunk_candidate_limit=args.chunk_candidate_limit, max_chunks_per_work=args.max_chunks_per_work, rrf_k=args.rrf_k))
        return
    if args.command == "rerank":
        from app.retrieval.reranked import search_reranked_evidence
        print_json(search_reranked_evidence(PROJECT_ROOT, embedding_provider=dense_provider_from_args(args), reranker_provider=reranker_provider_from_args(args), query=args.query, limit=args.limit, work_id=args.work_id, document_id=args.document_id, max_chunks_per_work=args.max_chunks_per_work, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k))
        return
    if args.command == "context":
        runtime = retrieval_runtime_from_args(args, include_reranker=args.mode == "accurate")
        print_json(runtime.build_context(query=args.query, mode=args.mode, retrieval_limit=args.retrieval_limit, token_budget=args.token_budget, max_evidence=args.max_evidence, max_evidence_per_work=args.max_evidence_per_work, work_id=args.work_id, document_id=args.document_id, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k).to_dict())
        return

    if args.command == "evaluate":
        from app.storage import write_json_atomic
        if args.evaluate_command == "generation":
            from app.evaluation.generation import evaluate_generation_files
            report = evaluate_generation_files(args.predictions, args.output)
            print_json(report)
            return
        from app.evaluation.retrieval import evaluate_bm25, evaluate_candidate_pool_oracle, evaluate_dense, evaluate_hybrid_rrf, evaluate_reranked
        output = args.output or PROJECT_ROOT / "data" / "evaluation" / f"{args.retriever}_baseline.json"
        values = parse_k_values(args.k_values)
        if args.retriever == "bm25":
            report = evaluate_bm25(PROJECT_ROOT, args.questions, output_path=output, k_values=values)
        elif args.retriever == "dense":
            report = evaluate_dense(PROJECT_ROOT, args.questions, dense_provider_from_args(args), output_path=output, k_values=values)
        elif args.retriever == "rrf":
            report = evaluate_hybrid_rrf(PROJECT_ROOT, args.questions, dense_provider_from_args(args), output_path=output, k_values=values, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k)
        elif args.retriever == "oracle":
            report = evaluate_candidate_pool_oracle(PROJECT_ROOT, args.questions, dense_provider_from_args(args), output_path=output, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k)
        else:
            report = evaluate_reranked(PROJECT_ROOT, args.questions, dense_provider_from_args(args), reranker_provider_from_args(args), output_path=output, k_values=values, candidate_limit=args.candidate_limit, rrf_k=args.rrf_k)
        write_json_atomic(output, report)
        print_json(report)
        return

    raise SystemExit(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
