"""Web-side corpus operations for the CrewAI Scholar application.

The Web process does not answer questions and does not execute agents. It owns
only upload/parse/index progress and read-only corpus status; Scholar requests
go through ``ScholarRunManager`` and the external Worker.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.ingestion.ingest import calculate_sha256
from app.knowledge.catalog import library_status, load_catalog, rebuild_catalog
from app.parsing.pipeline import PaperParseConfig, parse_paper
from app.web.models import ParseOptions


class EmitProgress(Protocol):
    def __call__(self, stage: str, message: str, progress: float, details: dict[str, Any] | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class WebRuntimeConfig:
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
    def from_environment(cls, project_root: Path) -> "WebRuntimeConfig":
        root = project_root.expanduser().resolve()
        defaults = cls()
        values: dict[str, str] = {}
        env_file = root / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    cleaned = line.strip()
                    if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
                        continue
                    key, _, value = cleaned.partition("=")
                    if key.strip().startswith("LEO_WEB_"):
                        values[key.strip()] = value.strip().strip("'\"")
            except OSError:
                pass

        def raw(name: str) -> str | None:
            return os.getenv(f"LEO_WEB_{name}", values.get(f"LEO_WEB_{name}"))

        def integer(name: str, default: int) -> int:
            return int(raw(name) or default)

        def boolean(name: str, default: bool) -> bool:
            value = raw(name)
            if value is None:
                return default
            normalized = value.casefold()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"LEO_WEB_{name} 必须是布尔值。")

        manifest: dict[str, Any] = {}
        try:
            from app.indexing.dense import load_dense_manifest

            manifest = load_dense_manifest(root)
        except (FileNotFoundError, OSError, ValueError):
            pass
        model = raw("EMBEDDING_MODEL") or manifest.get("model_name") or defaults.embedding_model
        revision = raw("EMBEDDING_REVISION") or manifest.get("model_revision") or None
        cache = raw("MODEL_CACHE")
        cache_path = Path(cache).expanduser() if cache else root / "data" / "models" / "huggingface"
        if not cache_path.is_absolute():
            cache_path = root / cache_path
        return cls(
            embedding_model=str(model),
            embedding_revision=str(revision) if revision else None,
            reranker_model=raw("RERANKER_MODEL") or defaults.reranker_model,
            reranker_revision=raw("RERANKER_REVISION") or None,
            device=raw("DEVICE") or None,
            model_cache=cache_path,
            embedding_batch_size=integer("EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size),
            reranker_batch_size=integer("RERANKER_BATCH_SIZE", defaults.reranker_batch_size),
            reranker_max_length=integer("RERANKER_MAX_LENGTH", defaults.reranker_max_length),
            local_files_only=boolean("LOCAL_FILES_ONLY", defaults.local_files_only),
        )

    def __post_init__(self) -> None:
        if self.embedding_batch_size < 1 or self.reranker_batch_size < 1:
            raise ValueError("Web 模型 batch size 必须大于 0。")
        if self.reranker_max_length < 32:
            raise ValueError("Web Reranker max length 不能小于 32。")


class LocalRAGWebRuntime:
    """Corpus facade retained for upload/index jobs only."""

    def __init__(self, project_root: Path, config: WebRuntimeConfig | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.config = config or WebRuntimeConfig.from_environment(self.project_root)
        self._retrieval: Any | None = None

    def _retrieval_runtime(self) -> Any:
        if self._retrieval is None:
            from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
            from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
            from app.runtime.retrieval import RetrievalRuntime

            self._retrieval = RetrievalRuntime(
                self.project_root,
                BGEM3EmbeddingProvider(BGEM3Config(
                    model_name=self.config.embedding_model,
                    revision=self.config.embedding_revision,
                    device=self.config.device,
                    cache_folder=self.config.model_cache,
                    batch_size=self.config.embedding_batch_size,
                    local_files_only=self.config.local_files_only,
                    show_progress_bar=False,
                )),
                BGERerankerProvider(BGERerankerConfig(
                    model_name=self.config.reranker_model,
                    revision=self.config.reranker_revision,
                    device=self.config.device,
                    cache_folder=self.config.model_cache,
                    batch_size=self.config.reranker_batch_size,
                    max_length=self.config.reranker_max_length,
                    local_files_only=self.config.local_files_only,
                    show_progress_bar=False,
                )),
            )
        return self._retrieval

    def parse_pdf(self, pdf_path: Path, options: ParseOptions, emit: EmitProgress) -> dict[str, Any]:
        stage_progress = {"ingesting": 0.08, "prechecking": 0.14, "waiting_for_mineru": 0.18, "running_mineru": 0.25, "normalizing": 0.82, "writing": 0.94}

        def progress(stage: str) -> None:
            emit(stage, f"PDF 处理阶段：{stage}", stage_progress.get(stage, 0.1))

        from app.corpus import CanonicalCorpusService
        from app.knowledge import KnowledgeIndexService

        corpus = CanonicalCorpusService(self.project_root)
        duplicate = corpus.duplicate_for_hash(calculate_sha256(pdf_path))
        if duplicate is not None:
            emit("writing", "检测到相同 content_hash，复用已有 Canonical Document。", 0.9, {"document_id": duplicate.document_id})
            canonical = corpus.documents.canonical(duplicate.document_id) or {}
            pipeline = canonical.get("pipeline")
            mineru_value = pipeline.get("mineru_output_directory") if isinstance(pipeline, dict) else None
            mineru_path = Path(str(mineru_value)) if mineru_value else Path("")
            if mineru_value and not mineru_path.is_absolute():
                mineru_path = self.project_root / mineru_path
            result = {"paper": {"paper_id": duplicate.paper_id, "document_id": duplicate.document_id, "sha256": duplicate.source_sha256, "paper_json": str(self.project_root / duplicate.canonical_path), "mineru_directory": str(mineru_path) if mineru_value else ""}}
        else:
            parsed = parse_paper(
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
            result = {"paper": {**asdict(parsed), "raw_pdf": str(parsed.raw_pdf), "paper_json": str(parsed.paper_json), "mineru_output_directory": str(parsed.mineru_output_directory)}}
        indexes = KnowledgeIndexService(project_root=self.project_root).synchronize_after_parse(self._retrieval_runtime().embedding_provider, emit=emit, catalog_builder=rebuild_catalog)
        result.update({"catalog": indexes["catalog"], "knowledge": indexes["knowledge"], "dense": indexes["dense"]})
        return result

    def list_papers(self) -> dict[str, Any]:
        catalog = load_catalog(self.project_root)
        return {"records": [record.to_dict() for record in catalog.records], "issues": [asdict(issue) for issue in catalog.issues], "status": library_status(self.project_root).to_dict()}

    def public_status(self) -> dict[str, Any]:
        from app.generation.settings import load_local_llm_settings
        from app.knowledge.corpus import corpus_summary
        from app.knowledge import knowledge_runtime_status

        llm = load_local_llm_settings(self.project_root)
        return {"service": "leo-research-agent-web", "rag_mode": "crewai_scholar", "orchestration_backend": "crewai", "llm_configured": bool(llm.base_url and llm.model), "llm_model": llm.model, "embedding_model": self.config.embedding_model, "embedding_revision": self.config.embedding_revision, "reranker_model": self.config.reranker_model, "reranker_revision": self.config.reranker_revision, "local_files_only": self.config.local_files_only, "models_initialized": self._retrieval is not None, "corpus_summary": corpus_summary(self.project_root).to_dict(), **knowledge_runtime_status(self.project_root)}

    def close(self) -> None:
        return None
