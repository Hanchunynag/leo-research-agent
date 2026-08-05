"""Configured route and cross-query Weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from app.agentic.models import RetrievalQuery
from app.graph.models import EvidenceCandidate


DEFAULT_ROUTE_WEIGHTS = {
    "lexical": 0.9, "dense": 1.0, "graph_direct": 1.3,
    "graph_path": 1.15, "community": 0.8,
}
DEFAULT_QUERY_WEIGHTS = {
    "original": 1.0, "focused_followup": 1.0, "subquestion": 0.9,
    "relationship_probe": 0.9, "terminology_expansion": 0.8,
    "community_probe": 0.8, "paraphrase": 0.7,
}


@dataclass(frozen=True)
class FusionConfig:
    rrf_k: int = 60
    route_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROUTE_WEIGHTS))
    query_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUERY_WEIGHTS))

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if any(value <= 0 for value in (*self.route_weights.values(), *self.query_weights.values())):
            raise ValueError("fusion weights must be positive")


def weighted_rrf(
    rankings: Mapping[str, Mapping[str, Sequence[EvidenceCandidate]]],
    queries: Sequence[RetrievalQuery], *, config: FusionConfig | None = None,
    limit: int = 40,
) -> list[EvidenceCandidate]:
    settings = config or FusionConfig()
    query_map = {value.query_id: value for value in queries}
    scores: defaultdict[str, float] = defaultdict(float)
    candidates: dict[str, EvidenceCandidate] = {}
    query_ids: defaultdict[str, set[str]] = defaultdict(set)
    routes: defaultdict[str, set[str]] = defaultdict(set)
    query_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    route_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for query_id, route_rankings in rankings.items():
        query = query_map.get(query_id)
        if query is None:
            continue
        query_weight = settings.query_weights.get(query.purpose, query.weight)
        for route, values in route_rankings.items():
            route_weight = settings.route_weights.get(route, 1.0)
            seen: set[str] = set()
            for fallback_rank, candidate in enumerate(values, 1):
                candidate_id = candidate.evidence_id
                if not candidate_id or candidate_id in seen:
                    continue
                seen.add(candidate_id)
                contribution = query_weight * route_weight / (settings.rrf_k + fallback_rank)
                scores[candidate_id] += contribution
                candidates.setdefault(candidate_id, candidate)
                query_ids[candidate_id].add(query_id)
                routes[candidate_id].add(route)
                query_ranks[candidate_id][query_id] = min(
                    query_ranks[candidate_id].get(query_id, fallback_rank), fallback_rank)
                route_scores[candidate_id][route] = (
                    route_scores[candidate_id].get(route, 0.0) + contribution)
    ordered = sorted(candidates, key=lambda key: (-scores[key], key))[:limit]
    output: list[EvidenceCandidate] = []
    for rank, candidate_id in enumerate(ordered, 1):
        candidate = candidates[candidate_id]
        output.append(candidate.model_copy(update={
            "query_ids": sorted(query_ids[candidate_id]),
            "routes": sorted(routes[candidate_id]),
            "per_query_ranks": query_ranks[candidate_id],
            "per_route_scores": {key: round(value, 12)
                                 for key, value in route_scores[candidate_id].items()},
            "fusion_score": round(scores[candidate_id], 12),
            "original_fusion_rank": rank,
        }))
    return output
