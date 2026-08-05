"""Route dispatcher for bounded multi-query GraphRAG retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

from app.agentic.models import RetrievalQuery
from app.graph.models import EvidenceCandidate
from app.retrieval.query_fusion import FusionConfig, weighted_rrf


Route = Callable[[RetrievalQuery], Sequence[EvidenceCandidate]]


def chunk_candidate(value: dict[str, Any], route: str) -> EvidenceCandidate:
    chunk_key = str(value.get("chunk_key") or value.get("chunk_id") or "")
    evidence_id = "CH_" + hashlib.sha256(chunk_key.encode()).hexdigest()[:20]
    return EvidenceCandidate(evidence_id=evidence_id, candidate_type="chunk",
        text=str(value.get("content") or ""), source_chunk_keys=[chunk_key],
        source_claim_ids=[], entity_ids=[], relation_ids=[], community_ids=[],
        routes=[route], directness_grade=int(value.get("directness_grade") or 1),
        metadata={key: value.get(key) for key in (
            "chunk_id", "work_id", "document_id", "paper_id", "title", "section_path",
            "page_start", "page_end", "block_ids", "content_zone")})


class MultiQueryRetriever:
    def __init__(self, routes: dict[str, Route], *, fusion_config: FusionConfig | None = None,
                 max_candidates: int = 40) -> None:
        self.routes = routes
        self.fusion_config = fusion_config or FusionConfig()
        self.max_candidates = max_candidates

    def retrieve(self, queries: Sequence[RetrievalQuery], enabled_routes: Sequence[str]) -> tuple[list[EvidenceCandidate], dict[str, Any]]:
        rankings: dict[str, dict[str, Sequence[EvidenceCandidate]]] = {}
        route_counts: dict[str, int] = {}
        for query in queries[:5]:
            rankings[query.query_id] = {}
            for route_name in enabled_routes:
                route = self.routes.get(route_name)
                if route is None:
                    continue
                values = list(route(query))
                rankings[query.query_id][route_name] = values
                route_counts[route_name] = route_counts.get(route_name, 0) + len(values)
        fused = weighted_rrf(rankings, queries, config=self.fusion_config,
                             limit=self.max_candidates)
        return fused, {"generated_query_count": len(queries),
            "per_route_candidate_count": route_counts,
            "unique_candidate_count": len({candidate.evidence_id for routes in rankings.values()
                                             for values in routes.values() for candidate in values}),
            "cross_query_fusion_count": len(fused)}
