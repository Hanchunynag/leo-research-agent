"""Bounded adaptive scientific query expansion and decomposition."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from app.agentic.models import QueryExpansionResult, QueryPlan, RetrievalQuery
from app.generation.openai_compatible import parse_json_object


QUERY_EXPANSION_PROMPT = """Generate bounded retrieval queries for scientific GraphRAG.
RQ0 must be the original query unchanged. Preserve every entity, category, satellite,
constellation, method, dataset, time, and scenario constraint. Add no unsupported entity.
Use at most five total queries. Relationship questions require a relationship_probe; global
questions use community_probe. Return strict JSON matching the schema and no markdown."""

PURPOSE_WEIGHTS = {
    "original": 1.0, "focused_followup": 1.0, "subquestion": 0.9,
    "relationship_probe": 0.9, "terminology_expansion": 0.8,
    "community_probe": 0.8, "paraphrase": 0.7,
}


class ExpansionProvider(Protocol):
    def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]: ...


def _mode(query: str, plan: QueryPlan) -> tuple[str, str]:
    normalized = query.casefold()
    if any(value in normalized for value in ("关系", "影响", "区别", "relation", "affect", "difference")):
        return "multi_hop" if len(plan.subquestions) > 1 else "compound", "relationship"
    if any(value in normalized for value in ("整个", "总体", "主要路线", "趋势", "across the corpus", "overall trend")):
        return "global", "global"
    if any(value in normalized for value in ("型号", "缩写", "公式", "exact", "error code")) or re.search(r"\b[A-Z][A-Z0-9-]{1,}\b", query):
        return "simple", "exact"
    if len(plan.subquestions) > 1:
        return "compound", "local"
    return "simple", "local"


def _original(query: str, plan: QueryPlan) -> RetrievalQuery:
    return RetrievalQuery(query_id="RQ0", text=query, purpose="original",
                          target_category=plan.target_category, required_entities=[],
                          required_constraints=plan.answer_constraints,
                          excluded_categories=plan.excluded_categories, weight=1.0)


class AdaptiveQueryExpander:
    def __init__(self, provider: ExpansionProvider | None = None,
                 max_variants: int = 5) -> None:
        if not 1 <= max_variants <= 5:
            raise ValueError("max_variants must be 1..5")
        self.provider = provider
        self.max_variants = max_variants

    def expand(self, query: str, plan: QueryPlan) -> QueryExpansionResult:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        complexity, retrieval_mode = _mode(cleaned, plan)
        fallback = self._deterministic(cleaned, plan, complexity, retrieval_mode)
        if self.provider is None or self.max_variants == 1:
            return fallback
        try:
            payload = self.provider.chat_completion([
                {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": "Schema:\n" + str(QueryExpansionResult.model_json_schema()) +
                 "\nPlan:\n" + plan.model_dump_json() + "\nOriginal query:\n" + cleaned},
            ], max_tokens=2400)
            content = payload["choices"][0]["message"]["content"]
            result = QueryExpansionResult.model_validate(parse_json_object(str(content)))
            return self._normalize(result, cleaned, plan, complexity, retrieval_mode)
        except (KeyError, IndexError, TypeError, ValueError):
            return fallback

    def _normalize(self, result: QueryExpansionResult, query: str, plan: QueryPlan,
                   complexity: str, retrieval_mode: str) -> QueryExpansionResult:
        original = _original(query, plan)
        others = [value for value in result.queries
                  if value.query_id != "RQ0" and value.text.strip() != query]
        unique: list[RetrievalQuery] = []
        seen = {query.casefold()}
        for value in others:
            normalized = " ".join(value.text.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value.model_copy(update={
                "query_id": f"RQ{len(unique) + 1}",
                "weight": PURPOSE_WEIGHTS[value.purpose],
            }))
        return QueryExpansionResult(original_query=query, complexity=complexity,
            retrieval_mode=retrieval_mode, queries=[original, *unique[:self.max_variants - 1]])

    def _deterministic(self, query: str, plan: QueryPlan, complexity: str,
                       mode: str) -> QueryExpansionResult:
        queries = [_original(query, plan)]
        purpose = "community_probe" if mode == "global" else (
            "relationship_probe" if mode == "relationship" else "subquestion")
        budget = {"simple": 1, "compound": 3, "multi_hop": 4, "global": 4}[complexity]
        for subquestion in plan.subquestions:
            if len(queries) >= min(self.max_variants, budget + 1):
                break
            if subquestion.question.strip() == query:
                continue
            queries.append(RetrievalQuery(
                query_id=f"RQ{len(queries)}", text=subquestion.question,
                purpose=purpose, target_category=plan.target_category,
                required_entities=[], required_constraints=plan.answer_constraints,
                excluded_categories=plan.excluded_categories,
                weight=PURPOSE_WEIGHTS[purpose],
            ))
        return QueryExpansionResult(original_query=query, complexity=complexity,
                                    retrieval_mode=mode, queries=queries)

    def focused(self, queries: Sequence[str], plan: QueryPlan) -> list[RetrievalQuery]:
        output: list[RetrievalQuery] = []
        for text in queries[:2]:
            if text.strip():
                output.append(RetrievalQuery(query_id=f"FQ{len(output) + 1}", text=text.strip(),
                    purpose="focused_followup", target_category=plan.target_category,
                    required_entities=[], required_constraints=plan.answer_constraints,
                    excluded_categories=plan.excluded_categories, weight=1.0))
        return output
