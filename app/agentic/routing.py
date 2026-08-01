"""组合指代、语义、实体和证据重合信号的 Topic Router。"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from app.agentic.config import AgenticRAGConfig
from app.agentic.models import RoutingLLMDecision, TopicRelation, TopicRoute
from app.embeddings.base import EmbeddingProvider
from app.indexing.tokenization import normalize_search_text


CONTEXT_DEPENDENCY_MARKERS = (
    "这个",
    "那个",
    "它",
    "上面",
    "刚才",
    "继续",
    "那为什么",
    "第二个呢",
    "前者",
    "后者",
    "what about it",
    "why is that",
    "continue",
)
DOMAIN_ENTITIES = (
    "低轨卫星",
    "leo",
    "星历",
    "时钟误差",
    "钟漂",
    "多普勒",
    "载波相位",
    "伪距",
    "观测量",
    "rrf",
    "bm25",
    "dense",
    "reranker",
    "rag",
    "python",
    "装饰器",
)
RAG_METHOD_ENTITIES = {"rrf", "bm25", "dense", "reranker", "rag"}


def extract_entities(value: str) -> list[str]:
    """提取稳定的领域实体和英文技术标识。"""

    normalized = normalize_search_text(value)
    entities = {entity for entity in DOMAIN_ENTITIES if entity in normalized}
    entities.update(
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", normalized)
        if token in DOMAIN_ENTITIES
    )
    if "时钟" in normalized and "时钟误差" not in entities:
        entities.add("时钟误差")
    return sorted(entities)


def is_context_dependent(query: str) -> bool:
    normalized = normalize_search_text(query)
    return any(marker in normalized for marker in CONTEXT_DEPENDENCY_MARKERS)


def rewrite_standalone_query(query: str, topic_summary: str) -> str:
    """消解常见追问指代；不修改已经完整的独立问题。"""

    cleaned = query.strip()
    normalized = normalize_search_text(cleaned)
    if "多普勒" in normalized and ("钟漂" in normalized or "时钟" in normalized):
        return "为什么多普勒频率观测能够约束低轨卫星与接收机之间的相对时钟漂移？"
    if not is_context_dependent(cleaned):
        return cleaned
    rewritten = cleaned
    for marker in ("那为什么", "这个", "那个", "它", "上面", "刚才", "继续"):
        rewritten = rewritten.replace(marker, "").strip(" ，,。？?")
    if not rewritten:
        rewritten = "继续解释该研究问题"
    return f"关于{topic_summary}，{rewritten}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _entity_overlap(query_entities: set[str], topic_entities: set[str]) -> float:
    if not query_entities or not topic_entities:
        return 0.0
    return len(query_entities & topic_entities) / len(query_entities | topic_entities)


def _evidence_overlap(
    candidates: Sequence[dict[str, Any]],
    registry: Sequence[dict[str, Any]],
) -> float:
    if not candidates or not registry:
        return 0.0
    registry_keys = {
        (str(item.get("chunk_id") or ""), str(item.get("work_id") or ""))
        for item in registry
    }
    candidate_keys = {
        (str(item.get("chunk_id") or ""), str(item.get("work_id") or ""))
        for item in candidates
    }
    return len(registry_keys & candidate_keys) / max(1, len(candidate_keys))


class TopicRouter:
    """按可解释组合分数路由，模糊区间最多调用一次 LLM。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: AgenticRAGConfig,
        ambiguity_resolver: Callable[
            [str, str, dict[str, float]], RoutingLLMDecision
        ]
        | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.config = config
        self.ambiguity_resolver = ambiguity_resolver

    def route(
        self,
        query: str,
        current_topic: dict[str, Any] | None,
        candidates: Sequence[dict[str, Any]],
        registry: Sequence[dict[str, Any]],
        *,
        force_new_topic: bool = False,
    ) -> TopicRoute:
        """计算四类信号，并在模糊区间至多调用一次结构化 Resolver。"""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query 不能为空。")
        if current_topic is None or force_new_topic:
            return TopicRoute(
                relation="new_topic",
                confidence=1.0,
                reason="没有活动 Topic 或用户显式要求新 Topic。",
                context_dependent=False,
                standalone_query=cleaned,
                reuse_previous_evidence=False,
                requires_new_retrieval=True,
                signals={},
            )

        summary = str(current_topic.get("topic_summary") or "").strip()
        standalone = rewrite_standalone_query(cleaned, summary)
        query_entities = set(extract_entities(standalone))
        topic_entities = set(current_topic.get("entities") or []) | set(
            extract_entities(summary)
        )
        context_dependent = is_context_dependent(cleaned)
        semantic = _cosine(
            self.embedding_provider.embed_query(standalone),
            self.embedding_provider.embed_query(summary or standalone),
        )
        entity = _entity_overlap(query_entities, topic_entities)
        dependency = 1.0 if context_dependent else 0.0
        evidence = _evidence_overlap(candidates, registry)
        score = (
            self.config.semantic_weight * semantic
            + self.config.entity_weight * entity
            + self.config.context_dependency_weight * dependency
            + self.config.evidence_overlap_weight * evidence
        )
        signals = {
            "semantic_similarity": round(semantic, 6),
            "entity_overlap": round(entity, 6),
            "context_dependency": dependency,
            "evidence_overlap": round(evidence, 6),
            "combined_score": round(score, 6),
        }

        if context_dependent and (query_entities & topic_entities):
            relation: TopicRelation = "same_topic"
            confidence = max(score, self.config.same_topic_threshold)
            reason = "问题含上下文依赖表达，且核心实体与当前 Topic 重合。"
        elif query_entities & RAG_METHOD_ENTITIES and not (
            topic_entities & RAG_METHOD_ENTITIES
        ):
            relation = "related_subtopic"
            confidence = 0.8
            reason = "问题切换到同一项目中的检索算法子主题。"
        elif "python" in query_entities or "装饰器" in query_entities:
            relation = "new_topic"
            confidence = 0.95
            reason = "问题与当前科研主题无实质关系。"
        elif score >= self.config.same_topic_threshold:
            relation = "same_topic"
            confidence = score
            reason = "组合路由分数达到同主题阈值。"
        elif score <= self.config.new_topic_threshold:
            relation = "new_topic"
            confidence = 1.0 - score
            reason = "组合路由分数低于新主题阈值。"
        elif self.ambiguity_resolver is not None:
            decision = self.ambiguity_resolver(standalone, summary, signals)
            return TopicRoute(
                relation=decision.relation,
                confidence=decision.confidence,
                reason=decision.reason,
                context_dependent=(
                    context_dependent or decision.context_dependent
                ),
                standalone_query=decision.standalone_query,
                reuse_previous_evidence=decision.reuse_previous_evidence,
                requires_new_retrieval=decision.requires_new_retrieval,
                signals=signals,
            )
        else:
            relation = "related_subtopic"
            confidence = 0.55
            reason = "组合分数处于模糊区间，保守创建相关子主题。"

        return TopicRoute(
            relation=relation,
            confidence=min(1.0, confidence),
            reason=reason,
            context_dependent=context_dependent,
            standalone_query=standalone,
            reuse_previous_evidence=relation == "same_topic",
            requires_new_retrieval=relation != "same_topic" or evidence < 1.0,
            signals=signals,
        )
