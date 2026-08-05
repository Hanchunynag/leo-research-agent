"""Default online GraphRAG runtime over FTS5, Qdrant, Neo4j and communities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from app.agentic.models import RetrievalQuery
from app.embeddings.base import EmbeddingProvider
from app.graph.client import Neo4jClient, Neo4jSettings
from app.graph.models import EvidenceCandidate
from app.graph.retrieval import GraphRetriever
from app.index_registry.store import IndexRegistryStore
from app.retrieval.incremental_dense import search_community_dense, search_incremental_dense
from app.retrieval.lexical_fts import search_lexical_evidence
from app.retrieval.multi_query import chunk_candidate
from app.retrieval.query_fusion import FusionConfig, weighted_rrf


RELATION_MARKERS = re.compile(r"(?:与|和|跟|versus|vs\.?|between|relation|关系|影响|区别)", re.I)


class GraphRAGRetrievalRuntime:
    is_graphrag = True

    def __init__(self, project_root: Path, embedding_provider: EmbeddingProvider,
                 reranker_provider: Any | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.registry = IndexRegistryStore(self.project_root)
        self.active_epoch = self.registry.active_epoch()
        if self.active_epoch is None:
            raise RuntimeError("GraphRAG requires an active index epoch; run knowledge sync")
        settings = Neo4jSettings.from_environment(self.project_root / ".env")
        self.neo4j = Neo4jClient(settings)
        self.graph = GraphRetriever(self.neo4j.driver, settings.database)
        self.last_diagnostics: dict[str, Any] = {}

    def close(self) -> None:
        self.neo4j.close()

    def _queries(self, values: Sequence[str]) -> list[RetrievalQuery]:
        output: list[RetrievalQuery] = []
        for index, text in enumerate(dict.fromkeys(value.strip() for value in values if value.strip())):
            output.append(RetrievalQuery(query_id=f"RQ{index}", text=text,
                purpose="original" if index == 0 else "subquestion",
                target_category="other", required_entities=[], required_constraints=[],
                excluded_categories=[], weight=1.0 if index == 0 else 0.9))
            if len(output) == 5:
                break
        return output

    def _linked_entities(self, text: str) -> list[dict[str, Any]]:
        pieces = [value.strip(" ，,。？?的") for value in RELATION_MARKERS.split(text)
                  if len(value.strip()) >= 2]
        candidates: list[dict[str, Any]] = []
        for piece in [*pieces, text]:
            try:
                candidates.extend(self.graph.link_entity(piece, 2))
            except Exception:
                continue
        unique: dict[str, dict[str, Any]] = {}
        for value in candidates:
            unique.setdefault(str(value["entity_id"]), value)
        return list(unique.values())[:4]

    def retrieve_multi(self, values: Sequence[str], *, limit: int = 40,
                       rrf_k: int = 60) -> dict[str, Any]:
        queries = self._queries(values)
        rankings: dict[str, dict[str, list[EvidenceCandidate]]] = {}
        route_counts: dict[str, int] = {}
        graph_paths = direct_relations = 0
        relationship_mode = any(RELATION_MARKERS.search(query.text) for query in queries)
        relationship_status = "none" if relationship_mode else "not_applicable"
        global_mode = any(marker in query.text.casefold() for query in queries
                          for marker in ("总体", "整个论文库", "主要路线", "趋势", "global", "overall"))
        for query in queries:
            routes: dict[str, list[EvidenceCandidate]] = {}
            lexical = search_lexical_evidence(self.project_root, query.text, limit=20,
                                               max_chunks_per_work=20)["results"]
            routes["lexical"] = [chunk_candidate(value, "lexical") for value in lexical]
            dense = search_incremental_dense(self.project_root, self.embedding_provider,
                                              query.text, limit=20)["results"]
            routes["dense"] = [chunk_candidate(value, "dense") for value in dense]
            linked = self._linked_entities(query.text)
            graph_direct: list[EvidenceCandidate] = []
            graph_path: list[EvidenceCandidate] = []
            if relationship_mode and len(linked) >= 2:
                relation = self.graph.relationship_search(linked[0]["entity_id"],
                    linked[1]["entity_id"], self.active_epoch, max_paths=20)
                candidates = list(relation["candidates"])
                if relation["mode"] == "direct":
                    relationship_status = "direct"
                    graph_direct.extend(candidates)
                elif relation["mode"] == "inferred_path":
                    if relationship_status != "direct":
                        relationship_status = "inferred_path"
                    graph_path.extend(candidates)
            elif not relationship_mode:
                for entity in linked[:2]:
                    graph_direct.extend(self.graph.local(entity["entity_id"], self.active_epoch, 10))
            routes["graph_direct"] = graph_direct
            routes["graph_path"] = graph_path
            direct_relations += len(graph_direct)
            graph_paths += len(graph_path)
            if global_mode:
                try:
                    reports = search_community_dense(self.project_root,
                        self.embedding_provider, query.text, limit=10)["results"]
                except (RuntimeError, ValueError):
                    reports = []
                routes["community"] = [self._community_candidate(value) for value in reports]
            rankings[query.query_id] = routes
            for route, candidates in routes.items():
                route_counts[route] = route_counts.get(route, 0) + len(candidates)
        fused = weighted_rrf(rankings, queries, config=FusionConfig(rrf_k=rrf_k), limit=limit)
        results = [self._materialize(candidate, rank) for rank, candidate in enumerate(fused, 1)]
        results = [value for value in results if value is not None]
        self.last_diagnostics = {"generated_query_count": len(queries),
            "accepted_query_count": len(queries), "rejected_query_count": 0,
            "query_drift_reasons": [], "retrieval_mode": "graphrag",
            "per_route_candidate_count": route_counts,
            "unique_candidate_count": len(fused), "cross_query_fusion_count": len(fused),
            "graph_path_count": graph_paths, "direct_relation_count": direct_relations}
        self.last_diagnostics["relationship_status"] = relationship_status
        return {"retriever": "agentic_graphrag", "result_count": len(results),
                "results": results, "diagnostics": self.last_diagnostics}

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.retrieve_multi([query], limit=int(kwargs.get("limit", 20)),
                                   rrf_k=int(kwargs.get("rrf_k", 60)))

    def _materialize(self, candidate: EvidenceCandidate, rank: int) -> dict[str, Any] | None:
        chunks = self.registry.visible_chunks(self.active_epoch)
        source = next((chunks[key] for key in candidate.source_chunk_keys if key in chunks), None)
        if source is None:
            return None
        value = dict(source)
        graph_prefix = candidate.text.strip()
        if candidate.candidate_type != "chunk":
            if candidate.candidate_type == "graph_path":
                graph_prefix = ("现有文献没有直接给出完整的 A-B 表述；以下关系由两组"
                                "有来源证据共同支持，属于间接推断。\n" + graph_prefix)
            value["content"] = graph_prefix + "\n\nOriginal source chunk:\n" + str(source.get("content") or "")
        value.update({"rank": rank, "score": candidate.fusion_score,
            "fusion_score": candidate.fusion_score, "retrieval_source": "graphrag_fused",
            "candidate_type": candidate.candidate_type,
            "matched_query_ids": candidate.query_ids, "matched_routes": candidate.routes,
            "per_query_ranks": candidate.per_query_ranks,
            "per_route_scores": candidate.per_route_scores,
            "source_claim_ids": candidate.source_claim_ids,
            "directness_grade": candidate.directness_grade or 1})
        return value

    @staticmethod
    def _community_candidate(value: dict[str, Any]) -> EvidenceCandidate:
        report = value.get("full_report")
        parsed = json.loads(report) if isinstance(report, str) else {}
        community_id = str(value.get("community_id") or "")
        return EvidenceCandidate(evidence_id="CM_" + community_id,
            candidate_type="community_report",
            text=str(parsed.get("summary") or value.get("summary") or ""),
            source_chunk_keys=list(value.get("source_chunk_keys") or []),
            source_claim_ids=list(value.get("source_claim_ids") or []),
            community_ids=[community_id], routes=["community"], directness_grade=1,
            metadata=value)
