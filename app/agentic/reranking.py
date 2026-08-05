"""面向直接回答能力的候选精排与安全回退。"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any, Protocol

from app.agentic.models import QueryPlan
from app.indexing.dense import dense_chunk_text
from app.indexing.tokenization import normalize_search_text
from app.reranking.base import RerankerProvider


class Reranker(Protocol):
    """Agentic 编排层所需的候选精排契约。"""

    last_diagnostics: dict[str, Any]

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int,
        plan: QueryPlan,
    ) -> list[dict[str, Any]]:
        """结合 Cross-Encoder 分数与直接性等级排序，失败时回退 RRF。"""

        ...


def directness_grade(candidate: dict[str, Any], plan: QueryPlan) -> int:
    """以 0-3 评估候选是否直接回答类别化问题。"""

    text = normalize_search_text(
        f"{' '.join(candidate.get('section_path') or [])} "
        f"{candidate.get('content') or ''}"
    )
    background = any(
        marker in text
        for marker in ("introduction", "background", "challenge", "related work")
    )
    if plan.target_category in {"measurement", "observable"}:
        measurement = any(
            marker in text
            for marker in (
                "measurement",
                "observable",
                "载波相位",
                "多普勒",
                "伪距",
                "carrier phase",
                "doppler",
                "pseudorange",
            )
        )
        direct = measurement and any(
            marker in text
            for marker in (
                "used as input",
                "were used",
                "estimate",
                "estimated",
                "用于",
                "估计",
            )
        )
        predicted_only = (
            any(marker in text for marker in ("predicted ephemer", "sgp4", "tle"))
            and not measurement
        )
        if predicted_only:
            return 0
        if direct and not background:
            return 3
        if measurement:
            return 2 if not background else 1
        return 0
    if plan.intent == "synthesis" and plan.target_category == "method":
        section = normalize_search_text(
            " ".join(str(value) for value in candidate.get("section_path") or [])
        )
        method_section = any(
            marker in section
            for marker in (
                "abstract",
                "method",
                "framework",
                "approach",
                "algorithm",
                "conclusion",
                "result",
                "experiment",
            )
        )
        contribution = any(
            marker in text
            for marker in (
                "we propose",
                "is proposed",
                "proposed method",
                "we develop",
                "is developed",
                "this paper presented",
                "this article studied",
                "framework",
                "scheme",
                "approach",
                "method",
                "estimate",
                "tracking",
                "compensation",
                "correction",
                "refinement",
                "prediction",
                "validation",
                "experiment",
            )
        )
        if contribution and method_section and not background:
            return 3
        if contribution and not background:
            return 2
        if contribution or method_section:
            return 1
        return 0
    query_terms = set(
        normalize_search_text(" ".join(plan.retrieval_queries)).split()
    )
    overlap = sum(term in text for term in query_terms if len(term) > 2)
    if overlap >= 2 and not background:
        return 3
    if overlap:
        return 2 if not background else 1
    return 0


class DirectAnswerReranker:
    """Cross-Encoder 相关性与类别直接性联合排序，失败时回退 RRF。"""

    def __init__(self, provider: RerankerProvider | None, *, enabled: bool = True) -> None:
        self.provider = provider
        self.enabled = enabled
        self.last_diagnostics: dict[str, Any] = {}

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int,
        plan: QueryPlan,
    ) -> list[dict[str, Any]]:
        """结合模型相关性和 0–3 直接性评分返回有界候选。"""

        started = perf_counter()
        values = [dict(candidate) for candidate in candidates]
        fallback = not self.enabled or self.provider is None
        error_message: str | None = None
        scores: list[float]
        if fallback:
            scores = [float(-index) for index in range(len(values))]
        else:
            provider = self.provider
            assert provider is not None
            try:
                scores = provider.score(
                    query,
                    [dense_chunk_text(candidate) for candidate in values],
                )
                if len(scores) != len(values):
                    raise RuntimeError("Reranker 分数数量不匹配。")
            except Exception as error:
                fallback = True
                error_message = f"{type(error).__name__}: {error}"
                scores = [float(-index) for index in range(len(values))]

        for candidate, score in zip(values, scores, strict=True):
            candidate["rrf_rank"] = candidate.get("rank")
            candidate["reranker_score"] = float(score)
            candidate["directness_grade"] = directness_grade(candidate, plan)
        values.sort(
            key=lambda item: (
                -int(item["directness_grade"]),
                -float(item["reranker_score"]),
                int(item.get("rrf_rank") or 10**9),
                str(item.get("chunk_id") or ""),
            )
        )
        selected = values[:top_k]
        for rank, candidate in enumerate(selected, 1):
            candidate["rank"] = rank
            candidate["retrieval_source"] = (
                "agentic_direct_reranked" if not fallback else "agentic_rrf_fallback"
            )
        self.last_diagnostics = {
            "enabled": self.enabled,
            "model": getattr(self.provider, "model_name", None),
            "revision": getattr(self.provider, "revision", None),
            "candidate_count": len(values),
            "output_count": len(selected),
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "fallback_used": fallback,
            "fallback_error": error_message,
        }
        return selected
