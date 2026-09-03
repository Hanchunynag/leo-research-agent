"""复用模型实例，并提供 fast/accurate 两档证据上下文入口。"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from app.context.assembly import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_MAX_EVIDENCE,
    assemble_context_bundle,
)
from app.context.models import ContextBundle
from app.embeddings.base import EmbeddingProvider
from app.reranking.base import RerankerProvider
from app.retrieval.hybrid import search_hybrid_evidence
from app.retrieval.hierarchical import search_hierarchical_evidence
from app.retrieval.reranked import search_reranked_evidence


RetrievalMode = Literal["fast", "accurate", "hierarchical"]


class RetrievalRuntime:
    """在 UI、API 或本地 Agent 进程生命周期内复用 Dense/Reranker。"""

    def __init__(
        self,
        project_root: Path,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider | None = None,
        hierarchical_enabled: bool = True,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.hierarchical_enabled = hierarchical_enabled
        self._embedding_warmed = False
        self._reranker_warmed = False

    def warmup(self, include_reranker: bool = True) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        if not self._embedding_warmed:
            started = perf_counter()
            vector = self.embedding_provider.embed_query("warmup retrieval query")
            if not vector:
                raise RuntimeError("Embedding warmup 返回了空向量。")
            self._embedding_warmed = True
            diagnostics["embedding"] = {
                "status": "warmed",
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            }
        else:
            diagnostics["embedding"] = {"status": "already_warm", "elapsed_ms": 0.0}

        if include_reranker:
            if self.reranker_provider is None:
                raise RuntimeError("accurate 模式需要 RerankerProvider。")
            if not self._reranker_warmed:
                started = perf_counter()
                scores = self.reranker_provider.score(
                    "warmup relevance query",
                    ["warmup candidate document"],
                )
                if len(scores) != 1:
                    raise RuntimeError("Reranker warmup 返回的分数数量不正确。")
                self._reranker_warmed = True
                diagnostics["reranker"] = {
                    "status": "warmed",
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                }
            else:
                diagnostics["reranker"] = {
                    "status": "already_warm",
                    "elapsed_ms": 0.0,
                }
        return diagnostics

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "fast",
        limit: int = 10,
        work_id: str | None = None,
        document_id: str | None = None,
        max_chunks_per_work: int = 2,
        candidate_limit: int = 20,
        rrf_k: int = 60,
        paper_limit: int = 10,
        paper_candidate_limit: int = 30,
        chunk_candidate_limit: int = 40,
        paper_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hierarchical_ready = (
            (self.project_root / "data" / "index" / "paper_bm25.json").is_file()
            and (self.project_root / "data" / "index" / "paper_dense_manifest.json").is_file()
            and (self.project_root / "data" / "index" / "qdrant_papers_dense").is_dir()
        )
        if mode == "hierarchical" or (self.hierarchical_enabled and hierarchical_ready):
            # An explicit hierarchical call uses the configured reranker when
            # available.  Automatic use from legacy ``fast`` remains cheap;
            # callers can request ``accurate`` for the Cross Encoder stage.
            reranker = (
                self.reranker_provider
                if mode in {"hierarchical", "accurate"}
                else None
            )
            result = search_hierarchical_evidence(
                project_root=self.project_root,
                embedding_provider=self.embedding_provider,
                query=query,
                reranker_provider=reranker,
                limit=limit,
                paper_limit=paper_limit,
                paper_candidate_limit=paper_candidate_limit,
                chunk_candidate_limit=chunk_candidate_limit,
                max_chunks_per_work=max_chunks_per_work,
                rrf_k=rrf_k,
                paper_filters=paper_filters,
            )
            self._embedding_warmed = True
            if reranker is not None:
                self._reranker_warmed = True
            return result
        if mode == "fast":
            result = search_hybrid_evidence(
                project_root=self.project_root,
                provider=self.embedding_provider,
                query=query,
                limit=limit,
                work_id=work_id,
                document_id=document_id,
                max_chunks_per_work=max_chunks_per_work,
                candidate_limit=candidate_limit,
                rrf_k=rrf_k,
            )
            self._embedding_warmed = True
            return result
        if mode != "accurate":
            raise ValueError("mode 必须是 fast、accurate 或 hierarchical。")
        if self.reranker_provider is None:
            raise RuntimeError("accurate 模式需要 RerankerProvider。")
        result = search_reranked_evidence(
            project_root=self.project_root,
            embedding_provider=self.embedding_provider,
            reranker_provider=self.reranker_provider,
            query=query,
            limit=limit,
            work_id=work_id,
            document_id=document_id,
            max_chunks_per_work=max_chunks_per_work,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
        )
        self._embedding_warmed = True
        self._reranker_warmed = True
        return result

    def build_context(
        self,
        query: str,
        *,
        mode: RetrievalMode = "fast",
        retrieval_limit: int = 10,
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        max_evidence_per_work: int = 2,
        work_id: str | None = None,
        document_id: str | None = None,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ) -> ContextBundle:
        started = perf_counter()
        retrieval = self.retrieve(
            query,
            mode=mode,
            limit=retrieval_limit,
            work_id=work_id,
            document_id=document_id,
            max_chunks_per_work=max_evidence_per_work,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
        )
        retrieval_ms = (perf_counter() - started) * 1000
        raw_results = retrieval.get("results")
        results = (
            [value for value in raw_results if isinstance(value, dict)]
            if isinstance(raw_results, list)
            else []
        )
        retrieval_diagnostics = {
            "retriever": retrieval.get("retriever"),
            "result_count": retrieval.get("result_count"),
            "elapsed_ms": round(retrieval_ms, 3),
            "timing": retrieval.get("timing"),
            "candidate_limit": candidate_limit,
            "rrf_k": rrf_k,
            "embedding_model": getattr(self.embedding_provider, "model_name", None),
            "embedding_revision": getattr(self.embedding_provider, "revision", None),
            "reranker_model": (
                getattr(self.reranker_provider, "model_name", None)
                if mode == "accurate"
                else None
            ),
            "reranker_revision": (
                getattr(self.reranker_provider, "revision", None)
                if mode == "accurate"
                else None
            ),
        }
        return assemble_context_bundle(
            query=query,
            retrieval_mode=mode,
            results=results,
            token_budget=token_budget,
            max_evidence=max_evidence,
            max_evidence_per_work=max_evidence_per_work,
            retrieval_diagnostics=retrieval_diagnostics,
        )
