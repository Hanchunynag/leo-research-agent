"""Deterministic query-drift guard with one optional structured adjudication."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Callable

from app.agentic.models import (
    QueryExpansionResult, QueryValidationDecision, QueryValidationResult, RetrievalQuery,
)
from app.embeddings.base import EmbeddingProvider
from app.indexing.tokenization import normalize_search_text


CATEGORY_TERMS = {
    "measurement": {"measurement", "observable", "观测", "测量"},
    "method": {"method", "algorithm", "model", "方法", "算法", "模型"},
    "prior": {"prior", "prediction", "先验", "预测"},
    "state": {"state", "状态"},
}
CONSTRAINT_PATTERN = re.compile(
    r"\b(?:GPS|GNSS|LEO|EKF|ESEKF|SGP4|HPOP|Orbcomm|Starlink|OneWeb|Iridium)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:hz|khz|mhz|m|km|s|ms|db)\b|"
    r"(?:高噪声|低噪声|仿真|实测|数据集|星座|卫星)", re.IGNORECASE,
)


def extract_constraints(text: str) -> set[str]:
    return {normalize_search_text(value) for value in CONSTRAINT_PATTERN.findall(text)}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norms if norms else 0.0


class QueryDriftValidator:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None,
                 adjudicator: Callable[[str, str], bool] | None = None,
                 min_similarity: float = 0.38, duplicate_similarity: float = 0.96) -> None:
        self.embedding_provider = embedding_provider
        self.adjudicator = adjudicator
        self.min_similarity = min_similarity
        self.duplicate_similarity = duplicate_similarity

    def validate(self, expansion: QueryExpansionResult) -> QueryValidationResult:
        original = expansion.queries[0]
        if original.query_id != "RQ0" or original.purpose != "original" or original.text != expansion.original_query:
            raise ValueError("RQ0 must preserve the original query exactly")
        original_constraints = extract_constraints(original.text) | {
            normalize_search_text(value) for value in original.required_constraints if value.strip()
        }
        accepted = [original]
        rejected: list[RetrievalQuery] = []
        decisions = [QueryValidationDecision(query_id="RQ0", accepted=True)]
        original_vector = (self.embedding_provider.embed_query(original.text)
                           if self.embedding_provider else None)
        vectors: list[list[float] | None] = [original_vector]
        adjudication_used = False
        for query in expansion.queries[1:5]:
            reasons: list[str] = []
            constraints = extract_constraints(query.text) | {
                normalize_search_text(value) for value in query.required_constraints if value.strip()
            }
            missing = original_constraints - constraints
            if missing:
                reasons.append("missing_constraints:" + ",".join(sorted(missing)))
            if query.target_category != original.target_category:
                reasons.append("target_category_changed")
            original_norm = normalize_search_text(original.text)
            query_norm = normalize_search_text(query.text)
            for category, terms in CATEGORY_TERMS.items():
                original_has = any(term in original_norm for term in terms)
                query_has = any(term in query_norm for term in terms)
                if query_has and not original_has and category != original.target_category:
                    reasons.append("new_category:" + category)
            similarity: float | None = None
            vector = None
            if self.embedding_provider:
                vector = self.embedding_provider.embed_query(query.text)
                similarity = _cosine(original_vector or [], vector)
                if similarity < self.min_similarity:
                    reasons.append("semantic_drift")
                if any(_cosine(previous or [], vector) >= self.duplicate_similarity for previous in vectors):
                    reasons.append("near_duplicate")
            else:
                if query_norm == original_norm or query_norm in {normalize_search_text(v.text) for v in accepted}:
                    reasons.append("duplicate")
            if reasons and self.adjudicator and not adjudication_used and set(reasons) <= {"semantic_drift"}:
                adjudication_used = True
                if self.adjudicator(original.text, query.text):
                    reasons.clear()
            decision = QueryValidationDecision(query_id=query.query_id,
                accepted=not reasons, reasons=reasons, semantic_similarity=similarity)
            decisions.append(decision)
            if reasons:
                rejected.append(query)
            else:
                accepted.append(query)
                vectors.append(vector)
        return QueryValidationResult(accepted_queries=accepted,
                                     rejected_queries=rejected, decisions=decisions)
