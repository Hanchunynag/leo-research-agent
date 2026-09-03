"""FastAPI 进程内复用模型、RAG Service 与 Session Store。"""

from __future__ import annotations

import os
import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Protocol

from app.agentic.config import AgenticRAGConfig
from app.agentic.prompting import compact_topic
from app.agentic.store import AgenticSessionStore
from app.generation.settings import load_local_llm_settings
from app.ingestion.ingest import calculate_sha256
from app.knowledge.catalog import library_status, load_catalog, rebuild_catalog
from app.parsing.pipeline import PaperParseConfig, parse_paper
from app.web.models import AnswerRequest, ParseOptions


class EmitProgress(Protocol):
    """后台任务对 Web 层的最小进度回调。"""

    def __call__(
        self,
        stage: str,
        message: str,
        progress: float,
        details: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class WebRuntimeConfig:
    """只包含本地 Web 进程必需的模型与网络无关配置。"""

    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str | None = None
    device: str | None = None
    model_cache: Path | None = None
    embedding_batch_size: int = 8
    reranker_batch_size: int = 4
    reranker_max_length: int = 1024
    local_files_only: bool = True

    @classmethod
    def from_environment(cls, project_root: Path) -> WebRuntimeConfig:
        """从 ``LEO_WEB_*`` 读取配置，未指定时复用现有 Dense Manifest。"""

        root = project_root.expanduser().resolve()
        file_values: dict[str, str] = {}
        env_file = root / ".env"
        if env_file.is_file():
            try:
                lines = env_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
                    continue
                key, _, value = cleaned.partition("=")
                key = key.strip()
                if key.startswith("LEO_WEB_"):
                    file_values[key] = value.strip().strip("'\"")

        def raw(name: str) -> str | None:
            key = f"LEO_WEB_{name}"
            return os.getenv(key, file_values.get(key))

        def boolean(name: str, default: bool) -> bool:
            value = raw(name)
            if value is None:
                return default
            normalized = value.strip().casefold()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"LEO_WEB_{name} 必须是布尔值。")

        def integer(name: str, default: int) -> int:
            value = raw(name)
            return int(value) if value is not None else default

        defaults = cls()
        manifest: dict[str, Any] = {}
        try:
            from app.indexing.dense import load_dense_manifest

            manifest = load_dense_manifest(root)
        except (FileNotFoundError, OSError, ValueError):
            manifest = {}
        manifest_model = manifest.get("model_name")
        if not isinstance(manifest_model, str) or not manifest_model.strip():
            manifest_model = defaults.embedding_model
        manifest_revision = manifest.get("model_revision")
        if not isinstance(manifest_revision, str) or not manifest_revision.strip():
            manifest_revision = None

        configured_model = raw("EMBEDDING_MODEL")
        configured_revision = raw("EMBEDDING_REVISION")
        cache = raw("MODEL_CACHE")
        cache_path = Path(cache).expanduser() if cache else None
        if cache_path is not None and not cache_path.is_absolute():
            cache_path = root / cache_path
        return cls(
            embedding_model=(configured_model or str(manifest_model)).strip(),
            embedding_revision=(configured_revision or manifest_revision),
            reranker_model=(raw("RERANKER_MODEL") or defaults.reranker_model),
            reranker_revision=raw("RERANKER_REVISION") or None,
            device=raw("DEVICE") or None,
            model_cache=cache_path
            or root / "data" / "models" / "huggingface",
            embedding_batch_size=integer(
                "EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size
            ),
            reranker_batch_size=integer(
                "RERANKER_BATCH_SIZE", defaults.reranker_batch_size
            ),
            reranker_max_length=integer(
                "RERANKER_MAX_LENGTH", defaults.reranker_max_length
            ),
            local_files_only=boolean(
                "LOCAL_FILES_ONLY", defaults.local_files_only
            ),
        )

    def __post_init__(self) -> None:
        if self.embedding_batch_size < 1 or self.reranker_batch_size < 1:
            raise ValueError("Web 模型 batch size 必须大于 0。")
        if self.reranker_max_length < 32:
            raise ValueError("Web Reranker max length 不能小于 32。")


class LocalRAGWebRuntime:
    """为 API 提供论文、会话、解析和问答能力的长驻门面。"""

    def __init__(
        self,
        project_root: Path,
        config: WebRuntimeConfig | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.config = config or WebRuntimeConfig.from_environment(self.project_root)
        self.agentic_config = AgenticRAGConfig.from_environment(
            self.project_root / ".env"
        )
        self.store = AgenticSessionStore(
            self.project_root,
            database_path=self.agentic_config.session_db_path,
        )
        self._retrieval: Any | None = None
        self._service: Any | None = None
        self._operation_lock = Lock()
        self._bootstrap_backend: Any | None = None
        self._bootstrap_composition: Any | None = None
        self._bootstrap_stop = Event()
        self._bootstrap_thread: Thread | None = None

    def _ensure_bootstrap_providers(self) -> Any:
        if self._bootstrap_composition is not None:
            return self._bootstrap_composition
        from app.academic_mcp.service import AcademicDiscoveryService
        from app.research import build_bootstrap_provider_composition

        self._bootstrap_backend = AcademicDiscoveryService.create(
            self.project_root,
            contact_email=os.getenv("LEO_ACADEMIC_CONTACT_EMAIL"),
        )

        def index_after_parse(root: Path) -> dict[str, Any]:
            from app.knowledge_engine import KnowledgeIndexService

            return KnowledgeIndexService(project_root=root).synchronize_after_parse(
                self._retrieval_runtime().embedding_provider
            )

        composition = build_bootstrap_provider_composition(
            self.project_root,
            self._bootstrap_backend,
            worker_id="web-bootstrap-worker",
            indexer=index_after_parse,
        )
        self._bootstrap_composition = composition

        def work() -> None:
            composition.worker.recover_after_restart()
            while not self._bootstrap_stop.is_set():
                result = composition.worker.run_once()
                if result is None:
                    self._bootstrap_stop.wait(0.25)

        self._bootstrap_thread = Thread(
            target=work,
            name="leo-bootstrap-worker",
            daemon=True,
        )
        self._bootstrap_thread.start()
        return composition

    def _build_retrieval_runtime(self) -> Any:
        from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
        from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
        from app.runtime.retrieval import RetrievalRuntime

        embedding = BGEM3EmbeddingProvider(
            BGEM3Config(
                model_name=self.config.embedding_model,
                revision=self.config.embedding_revision,
                device=self.config.device,
                cache_folder=self.config.model_cache,
                batch_size=self.config.embedding_batch_size,
                local_files_only=self.config.local_files_only,
                show_progress_bar=False,
            )
        )
        reranker = BGERerankerProvider(
            BGERerankerConfig(
                model_name=self.config.reranker_model,
                revision=self.config.reranker_revision,
                device=self.config.device,
                cache_folder=self.config.model_cache,
                batch_size=self.config.reranker_batch_size,
                max_length=self.config.reranker_max_length,
                local_files_only=self.config.local_files_only,
                show_progress_bar=False,
            )
        )
        return RetrievalRuntime(self.project_root, embedding, reranker)

    def _retrieval_runtime(self) -> Any:
        if self._retrieval is None:
            self._retrieval = self._build_retrieval_runtime()
        return self._retrieval

    def _build_service(self) -> Any:
        from app.agentic.provider import OpenAIAgenticReasoningProvider
        from app.generation.openai_compatible import (
            OpenAICompatibleAnswerProvider,
            OpenAICompatibleConfig,
        )
        from app.langchain_agent import build_langchain_agent_service
        from app.workspaces import WorkspaceService

        llm = load_local_llm_settings(self.project_root)
        if not llm.base_url or not llm.model:
            raise ValueError(
                "缺少 LEO_LLM_BASE_URL 或 LEO_LLM_MODEL，请先配置本地 .env。"
            )
        api_key = llm.api_key.get_secret_value() if llm.api_key else None
        answer_provider = OpenAICompatibleAnswerProvider(
            OpenAICompatibleConfig(
                base_url=llm.base_url,
                model=llm.model,
                api_key=api_key,
                timeout_seconds=llm.timeout_seconds,
                max_tokens=llm.max_tokens,
                prompt_layout=llm.prompt_layout or "context_first",
                json_mode=llm.json_mode,
            )
        )
        retrieval = self._retrieval_runtime()
        from app.knowledge_engine import build_configured_unified_service

        knowledge = build_configured_unified_service(
            self.project_root,
            retrieval,
            answer_provider,
            llm_model_name=llm.model,
        )
        workspaces = WorkspaceService(self.project_root)
        bootstrap = self._ensure_bootstrap_providers()
        return build_langchain_agent_service(
            self.project_root,
            knowledge,
            workspaces,
            OpenAIAgenticReasoningProvider(
                answer_provider,
                max_structure_repairs=self.agentic_config.max_structure_repairs,
            ),
            session_store=self.store,
            extra_tools=bootstrap.gateway_handlers,
        )

    def close(self) -> None:
        self._bootstrap_stop.set()
        if self._bootstrap_thread is not None:
            self._bootstrap_thread.join(timeout=2.0)
        if self._bootstrap_backend is not None:
            close = self._bootstrap_backend.close
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # CLI and synchronous shutdown paths do not own an event loop.
                asyncio.run(close())
            else:
                # FastAPI lifespan shutdown already runs inside an event loop.
                # Scheduling the coroutine avoids both nested ``asyncio.run``
                # and an un-awaited coroutine warning.
                loop.create_task(close())

    def answer(self, request: AnswerRequest, emit: EmitProgress) -> dict[str, Any]:
        """串行执行问答，避免共享 Service 的运行诊断互相污染。"""

        with self._operation_lock:
            if self._service is None:
                emit("loading_models", "正在加载 Embedding 和 Reranker。", 0.08)
                self._service = self._build_service()
            emit(
                "langchain_rag",
                "正在执行 LangChain 翻译、双语检索、生成和验证。",
                0.20,
            )
            result = self._service.answer(
                request.query,
                session_id=request.session_id,
                force_new_topic=request.force_new_topic,
                include_context=request.include_context,
                progress_callback=emit,
            )
            harness = result.get("diagnostics", {}).get("harness", {})
            emit(
                "validated",
                "回答已完成证据和 Claim-Citation 验证。",
                0.92,
                {
                    "answerable": bool(result.get("answerable")),
                    "termination_reason": harness.get("termination_reason"),
                },
            )
            return result

    def resume(self, thread_id: str, user_input: str, emit: EmitProgress) -> dict[str, Any]:
        """Continue a LangGraph clarification with the original thread checkpoint."""

        with self._operation_lock:
            if self._service is None:
                raise ValueError("没有可恢复的 LangGraph 任务。")
            emit("resuming", "正在恢复 LangGraph 研究任务。", 0.20)
            result = self._service.resume(thread_id, user_input, progress_callback=emit)
            harness = result.get("diagnostics", {}).get("harness", {})
            emit(
                "validated",
                "恢复后的回答已完成证据和 Claim-Citation 验证。",
                0.92,
                {"answerable": bool(result.get("answerable")), "termination_reason": harness.get("termination_reason")},
            )
            return result

    def parse_pdf(
        self,
        pdf_path: Path,
        options: ParseOptions,
        emit: EmitProgress,
    ) -> dict[str, Any]:
        """通过现有唯一 Pipeline 解析 PDF，不建立 Web 专用分支逻辑。"""

        stage_progress = {
            "ingesting": 0.08,
            "prechecking": 0.14,
            "waiting_for_mineru": 0.18,
            "running_mineru": 0.25,
            "normalizing": 0.82,
            "writing": 0.94,
        }

        def progress(stage: str) -> None:
            emit(stage, f"PDF 处理阶段：{stage}", stage_progress.get(stage, 0.1))

        with self._operation_lock:
            from app.corpus import CanonicalCorpusService

            corpus = CanonicalCorpusService(self.project_root)
            duplicate = corpus.duplicate_for_hash(calculate_sha256(pdf_path))
            if duplicate is not None:
                from app.knowledge_engine import KnowledgeIndexService

                emit(
                    "writing",
                    "检测到相同 content_hash，复用已有 Canonical Document。",
                    0.9,
                    {"document_id": duplicate.document_id},
                )
                indexes = KnowledgeIndexService(
                    project_root=self.project_root
                ).synchronize_after_parse(
                    self._retrieval_runtime().embedding_provider,
                    emit=emit,
                    catalog_builder=rebuild_catalog,
                )
                # Scope/索引代际可能已推进，下次问答重新组合并固定新版本。
                self._service = None
                canonical = corpus.documents.canonical(duplicate.document_id) or {}
                pipeline = canonical.get("pipeline")
                mineru_value = (
                    pipeline.get("mineru_output_directory")
                    if isinstance(pipeline, dict)
                    else None
                )
                mineru_path = Path(str(mineru_value)) if mineru_value else Path("")
                if mineru_value and not mineru_path.is_absolute():
                    mineru_path = self.project_root / mineru_path
                return {
                    "paper": {
                        "paper_id": duplicate.paper_id,
                        "document_id": duplicate.document_id,
                        "sha256": duplicate.source_sha256,
                        "paper_json": str(self.project_root / duplicate.canonical_path),
                        "mineru_directory": str(mineru_path) if mineru_value else "",
                    },
                    "catalog": indexes["catalog"],
                    "knowledge": indexes["knowledge"],
                    "dense": indexes["dense"],
                }
            result = parse_paper(
                input_path=pdf_path,
                config=PaperParseConfig(
                    project_root=self.project_root,
                    method=options.method,
                    backend=options.backend,
                    formula_enabled=options.formula_enabled,
                    table_enabled=options.table_enabled,
                    force_mineru=options.force_mineru,
                ),
                progress_callback=progress,
            )
            from app.knowledge_engine import KnowledgeIndexService

            indexes = KnowledgeIndexService(
                project_root=self.project_root
            ).synchronize_after_parse(
                self._retrieval_runtime().embedding_provider,
                emit=emit,
                catalog_builder=rebuild_catalog,
            )
            # Scope/索引代际可能已推进，下次问答重新组合并固定新版本。
            self._service = None
            return {
                "paper": {
                    **asdict(result),
                    "raw_pdf": str(result.raw_pdf),
                    "paper_json": str(result.paper_json),
                    "mineru_output_directory": str(
                        result.mineru_output_directory
                    ),
                },
                "catalog": indexes["catalog"],
                "knowledge": indexes["knowledge"],
                "dense": indexes["dense"],
            }

    def list_papers(self) -> dict[str, Any]:
        catalog = load_catalog(self.project_root)
        return {
            "records": [record.to_dict() for record in catalog.records],
            "issues": [asdict(issue) for issue in catalog.issues],
            "status": library_status(self.project_root).to_dict(),
        }

    def list_sessions(self) -> dict[str, Any]:
        return {"sessions": self.store.list_sessions()}

    def session_details(self, session_id: str) -> dict[str, Any]:
        return self.store.session_details(session_id)

    def session_evidence(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        return {
            "session_id": session_id,
            "active_topic_id": session.get("active_topic_id"),
            "evidence": self.store.list_evidence(session_id),
        }

    def session_transcript(self, session_id: str) -> dict[str, Any]:
        """从活动 Topic 的 append-only 事件恢复可展示的对话。"""

        session = self.store.get_session(session_id)
        topic_id = session.get("active_topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            return {"session_id": session_id, "topic_id": None, "messages": []}
        messages: list[dict[str, Any]] = []
        for event in self.store.list_events(session_id, topic_id):
            content = event.get("content")
            if not isinstance(content, dict):
                continue
            if event.get("event_type") == "user_query":
                query = content.get("query")
                if isinstance(query, str) and query:
                    messages.append({"role": "user", "text": query})
            elif event.get("event_type") == "answer":
                answer = content.get("answer")
                answerable = bool(content.get("answerable"))
                refusal = content.get("refusal_reason")
                text = answer if answerable else refusal
                if isinstance(text, str) and text:
                    outcome = content.get("outcome")
                    messages.append(
                        {
                            "role": "assistant",
                            "text": text,
                            "answerable": answerable,
                            "outcome": (
                                outcome
                                if isinstance(outcome, dict)
                                else {
                                    "code": "legacy_refusal",
                                    "stage": "unknown",
                                    "message": text,
                                    "retryable": False,
                                }
                            ),
                        }
                    )
        return {
            "session_id": session_id,
            "topic_id": topic_id,
            "messages": messages,
        }

    def compact_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        topic_id = session.get("active_topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("Session 没有可压缩的活动 Topic。")
        report = compact_topic(self.store, session_id, topic_id)
        return report.model_dump(mode="json")

    def public_status(self) -> dict[str, Any]:
        """只返回可展示配置，绝不包含 API Key。"""

        llm = load_local_llm_settings(self.project_root)
        from app.knowledge_engine import knowledge_serving_status

        serving = knowledge_serving_status(self.project_root)
        return {
            "service": "leo-research-agent-web",
            "rag_mode": "langchain_bilingual_rag",
            "llm_configured": bool(llm.base_url and llm.model),
            "llm_model": llm.model,
            "embedding_model": self.config.embedding_model,
            "embedding_revision": self.config.embedding_revision,
            "reranker_model": self.config.reranker_model,
            "reranker_revision": self.config.reranker_revision,
            "local_files_only": self.config.local_files_only,
            "models_initialized": self._retrieval is not None,
            **serving,
            "candidate_limit": self.agentic_config.candidate_limit,
            "rerank_top_k": self.agentic_config.rerank_top_k,
            "final_top_k": self.agentic_config.final_top_k,
            "max_retrieval_rounds": self.agentic_config.max_retrieval_rounds,
            "evidence_mmr_lambda": self.agentic_config.evidence_mmr_lambda,
        }
