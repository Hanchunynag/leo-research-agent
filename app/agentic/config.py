"""Agentic RAG 的集中配置与边界校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgenticRAGConfig:
    candidate_limit: int = 20
    rerank_top_k: int = 8
    final_top_k: int = 5
    max_retrieval_rounds: int = 2
    max_structure_repairs: int = 1
    max_answer_repairs: int = 1
    max_total_latency_ms: int | None = None
    fail_closed: bool = True
    allow_model_downloads: bool = True
    rrf_k: int = 60
    reranker_enabled: bool = True
    semantic_validation_enabled: bool = True
    same_topic_threshold: float = 0.75
    new_topic_threshold: float = 0.45
    semantic_weight: float = 0.40
    entity_weight: float = 0.25
    context_dependency_weight: float = 0.20
    evidence_overlap_weight: float = 0.15
    context_compaction_threshold: float = 0.70
    model_context_window: int = 32_768
    recent_events_after_compaction: int = 8
    session_db_path: Path | None = None

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> AgenticRAGConfig:
        """从环境变量和可选 ``.env`` 加载配置，进程环境优先。"""

        defaults = cls()
        file_values: dict[str, str] = {}
        if env_file is not None and env_file.is_file():
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
                if key.startswith("LEO_AGENTIC_"):
                    file_values[key] = value.strip().strip("'\"")

        def raw(name: str) -> str | None:
            key = f"LEO_AGENTIC_{name}"
            return os.getenv(key, file_values.get(key))

        def integer(name: str, default: int) -> int:
            value = raw(name)
            return int(value) if value is not None else default

        def optional_integer(name: str, default: int | None) -> int | None:
            value = raw(name)
            return int(value) if value not in {None, ""} else default

        def number(name: str, default: float) -> float:
            value = raw(name)
            return float(value) if value is not None else default

        def boolean(name: str, default: bool) -> bool:
            value = raw(name)
            if value is None:
                return default
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"LEO_AGENTIC_{name} 必须是布尔值。")

        database = raw("SESSION_DB_PATH")
        return cls(
            candidate_limit=integer("CANDIDATE_LIMIT", defaults.candidate_limit),
            rerank_top_k=integer("RERANK_TOP_K", defaults.rerank_top_k),
            final_top_k=integer("FINAL_TOP_K", defaults.final_top_k),
            max_retrieval_rounds=integer(
                "MAX_RETRIEVAL_ROUNDS", defaults.max_retrieval_rounds
            ),
            max_structure_repairs=integer(
                "MAX_STRUCTURE_REPAIRS", defaults.max_structure_repairs
            ),
            max_answer_repairs=integer(
                "MAX_ANSWER_REPAIRS", defaults.max_answer_repairs
            ),
            max_total_latency_ms=optional_integer(
                "MAX_TOTAL_LATENCY_MS", defaults.max_total_latency_ms
            ),
            fail_closed=boolean("FAIL_CLOSED", defaults.fail_closed),
            allow_model_downloads=boolean(
                "ALLOW_MODEL_DOWNLOADS", defaults.allow_model_downloads
            ),
            rrf_k=integer("RRF_K", defaults.rrf_k),
            reranker_enabled=boolean(
                "RERANKER_ENABLED", defaults.reranker_enabled
            ),
            semantic_validation_enabled=boolean(
                "SEMANTIC_VALIDATION_ENABLED",
                defaults.semantic_validation_enabled,
            ),
            same_topic_threshold=number(
                "SAME_TOPIC_THRESHOLD", defaults.same_topic_threshold
            ),
            new_topic_threshold=number(
                "NEW_TOPIC_THRESHOLD", defaults.new_topic_threshold
            ),
            semantic_weight=number("SEMANTIC_WEIGHT", defaults.semantic_weight),
            entity_weight=number("ENTITY_WEIGHT", defaults.entity_weight),
            context_dependency_weight=number(
                "CONTEXT_DEPENDENCY_WEIGHT",
                defaults.context_dependency_weight,
            ),
            evidence_overlap_weight=number(
                "EVIDENCE_OVERLAP_WEIGHT", defaults.evidence_overlap_weight
            ),
            context_compaction_threshold=number(
                "CONTEXT_COMPACTION_THRESHOLD",
                defaults.context_compaction_threshold,
            ),
            model_context_window=integer(
                "MODEL_CONTEXT_WINDOW", defaults.model_context_window
            ),
            recent_events_after_compaction=integer(
                "RECENT_EVENTS_AFTER_COMPACTION",
                defaults.recent_events_after_compaction,
            ),
            session_db_path=Path(database) if database else None,
        )

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("candidate_limit", self.candidate_limit, 100),
            ("rerank_top_k", self.rerank_top_k, 50),
            ("final_top_k", self.final_top_k, 20),
            ("max_retrieval_rounds", self.max_retrieval_rounds, 5),
        ):
            if isinstance(value, bool) or value < 1 or value > maximum:
                raise ValueError(f"{name} 必须在 1 到 {maximum} 之间。")
        if self.rerank_top_k > self.candidate_limit:
            raise ValueError("rerank_top_k 不能大于 candidate_limit。")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k 不能大于 rerank_top_k。")
        if self.rrf_k < 1 or self.rrf_k > 10_000:
            raise ValueError("rrf_k 必须在 1 到 10000 之间。")
        if self.max_structure_repairs not in {0, 1}:
            raise ValueError("max_structure_repairs 只能是 0 或 1。")
        if self.max_answer_repairs not in {0, 1}:
            raise ValueError("max_answer_repairs 只能是 0 或 1。")
        if self.max_total_latency_ms is not None and self.max_total_latency_ms < 1:
            raise ValueError("max_total_latency_ms 必须为空或大于 0。")
        if not self.fail_closed:
            raise ValueError("Scientific RAG 必须保持 fail_closed=true。")
        if not 0 < self.new_topic_threshold < self.same_topic_threshold < 1:
            raise ValueError("Topic 阈值必须满足 0 < new < same < 1。")
        weight_sum = sum(
            (
                self.semantic_weight,
                self.entity_weight,
                self.context_dependency_weight,
                self.evidence_overlap_weight,
            )
        )
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("Topic Router 权重之和必须为 1。")
        if not 0.1 <= self.context_compaction_threshold <= 0.95:
            raise ValueError("context_compaction_threshold 必须在 0.1 到 0.95 之间。")
        if self.model_context_window < 1024:
            raise ValueError("model_context_window 不能小于 1024。")
        if self.recent_events_after_compaction < 1:
            raise ValueError("recent_events_after_compaction 不能小于 1。")
