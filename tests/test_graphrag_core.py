from __future__ import annotations

from pathlib import Path

from app.agentic.models import QueryExpansionResult, RetrievalQuery
from app.agentic.query_validation import QueryDriftValidator
from app.graph.communities import community_fingerprint
from app.graph.extraction import validate_extraction
from app.graph.models import (
    ChunkGraphExtraction, EvidenceCandidate, ExtractedEntity, ExtractedRelation,
)
from app.graph.ontology import EntityType, RelationPredicate, validate_relation_types
from app.graph.resolution import EntityResolver
from app.index_registry.diff import diff_chunks, stable_chunk_key, versioned_chunk
from app.index_registry.outbox import make_operation, operation_id_for
from app.index_registry.store import IndexRegistryStore
from app.indexing.incremental_dense import sync_incremental_dense
from app.retrieval.query_fusion import FusionConfig, weighted_rrf


def chunk(content: str = "Pseudorange rate constrains velocity state.") -> dict:
    return {
        "chunk_id": "D1_c0001", "document_id": "D1", "work_id": "W1",
        "paper_id": "P1", "section_id": "D1_s1", "section_path": ["Method"],
        "chunk_policy_version": "2.1", "block_ids": ["b1", "b2"],
        "content": content, "title": "Paper", "page_start": 1, "page_end": 1,
    }


def test_stable_chunk_key_ignores_sequence_chunk_id() -> None:
    first = chunk()
    second = {**first, "chunk_id": "D1_c9999"}
    assert stable_chunk_key(first) == stable_chunk_key(second)


def test_incremental_diff_all_categories() -> None:
    old_a = versioned_chunk(chunk("A"))
    old_b = versioned_chunk({**chunk("B"), "block_ids": ["b3"]})
    unchanged = dict(chunk("A"))
    added = {**chunk("C"), "block_ids": ["b4"]}
    changed = {**chunk("B changed"), "block_ids": ["b3"]}
    kinds = {item.chunk_key: item.kind for item in diff_chunks(
        [unchanged, added, changed], {old_a["chunk_key"]: old_a, old_b["chunk_key"]: old_b}
    )}
    assert kinds[old_a["chunk_key"]] == "unchanged"
    assert "added" in kinds.values()
    assert kinds[old_b["chunk_key"]] == "changed"
    deleted = diff_chunks([], {old_a["chunk_key"]: old_a})
    assert deleted[0].kind == "deleted"


def test_evidence_quote_and_ontology_validation() -> None:
    extraction = ChunkGraphExtraction(entities=[
        ExtractedEntity(local_id="m", name="pseudorange rate", entity_type="measurement"),
        ExtractedEntity(local_id="s", name="velocity", entity_type="state"),
    ], relations=[ExtractedRelation(subject_local_id="m", predicate="CONSTRAINS",
        object_local_id="s", confidence=0.9, evidence_quote="constrains velocity",
        polarity="support", qualifiers={})])
    valid, issues = validate_extraction(extraction, chunk()["content"])
    assert len(valid.relations) == 1 and not issues
    invalid, issues = validate_extraction(extraction, "unrelated")
    assert not invalid.relations and issues
    assert validate_relation_types(EntityType.MEASUREMENT, RelationPredicate.CONSTRAINS,
                                   EntityType.STATE)[0]
    assert not validate_relation_types(EntityType.METHOD, RelationPredicate.CONSTRAINS,
                                       EntityType.STATE)[0]


def test_query_drift_guard_preserves_rq0_and_rejects_category_change() -> None:
    original = RetrievalQuery(query_id="RQ0", text="Orbcomm 多普勒观测如何约束速度状态？",
        purpose="original", target_category="measurement", required_entities=["Orbcomm"],
        required_constraints=["Orbcomm"], excluded_categories=["method"], weight=1.0)
    drift = RetrievalQuery(query_id="RQ1", text="Starlink 使用哪些方法？", purpose="paraphrase",
        target_category="method", required_entities=["Starlink"], required_constraints=[],
        excluded_categories=[], weight=0.7)
    result = QueryDriftValidator().validate(QueryExpansionResult(
        original_query=original.text, complexity="compound", retrieval_mode="local",
        queries=[original, drift]))
    assert result.accepted_queries == [original]
    assert "target_category_changed" in result.decisions[1].reasons


def _candidate(evidence_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(evidence_id=evidence_id, candidate_type="chunk", text=evidence_id)


def test_cross_query_weighted_rrf_accumulates_routes_and_queries() -> None:
    queries = [
        RetrievalQuery(query_id="RQ0", text="q", purpose="original", target_category="other",
                       required_entities=[], required_constraints=[], excluded_categories=[], weight=1),
        RetrievalQuery(query_id="RQ1", text="q2", purpose="paraphrase", target_category="other",
                       required_entities=[], required_constraints=[], excluded_categories=[], weight=.7),
    ]
    fused = weighted_rrf({
        "RQ0": {"dense": [_candidate("A"), _candidate("B")],
                "graph_direct": [_candidate("A")]},
        "RQ1": {"lexical": [_candidate("A")]},
    }, queries, config=FusionConfig(rrf_k=60), limit=10)
    assert fused[0].evidence_id == "A"
    assert fused[0].query_ids == ["RQ0", "RQ1"]
    assert set(fused[0].routes) == {"dense", "graph_direct", "lexical"}


def test_community_fingerprint_is_order_independent() -> None:
    assert community_fingerprint(["E2", "E1"], ["USES", "AFFECTS"]) == \
           community_fingerprint(["E1", "E2"], ["AFFECTS", "USES"])


def test_epoch_visibility_and_outbox_idempotency(tmp_path: Path) -> None:
    store = IndexRegistryStore(tmp_path)
    epoch1 = store.create_epoch()
    value = versioned_chunk(chunk())
    store.put_chunk_version(epoch1, value)
    operation = make_operation(epoch=epoch1, store="lexical", operation_type="upsert",
        object_id=value["chunk_key"], target_hash=value["content_hash"], payload={})
    store.enqueue(operation)
    store.enqueue(operation)
    assert len(store.list_operations(epoch1)) == 1
    store.mark_operation(operation.operation_id, "completed")
    store.activate_epoch(epoch1)
    assert value["chunk_key"] in store.visible_chunks()
    epoch2 = store.create_epoch()
    store.invalidate_chunk(epoch2, value["chunk_key"])
    assert value["chunk_key"] in store.visible_chunks()
    store.activate_epoch(epoch2)
    assert value["chunk_key"] not in store.visible_chunks()
    assert operation.operation_id == operation_id_for(
        epoch1, "lexical", "upsert", value["chunk_key"], value["content_hash"])


def test_unchanged_dense_sync_never_calls_embedding(tmp_path: Path) -> None:
    class Provider:
        model_name = "BAAI/bge-m3"
        revision = "test"
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("unchanged sync must not embed")

    value = versioned_chunk(chunk())
    diff = diff_chunks([chunk()], {value["chunk_key"]: value})
    report = sync_incremental_dense(tmp_path, Provider(), 2, diff)  # type: ignore[arg-type]
    assert report.embedded_count == 0
    assert report.upserted_count == 0


def test_entity_resolution_is_type_aware_and_blocks_known_false_merges(tmp_path: Path) -> None:
    resolver = EntityResolver(IndexRegistryStore(tmp_path), fuzzy_threshold=80)
    ekf = resolver.resolve("EKF", "algorithm", 1, ["extended Kalman filter"])
    same = resolver.resolve("扩展卡尔曼滤波", "algorithm", 1)
    esekf = resolver.resolve("ESEKF", "algorithm", 1)
    measurement = resolver.resolve("prediction", "measurement", 1)
    prior = resolver.resolve("prediction", "prior", 1)
    assert ekf.entity_id == same.entity_id
    assert ekf.entity_id != esekf.entity_id
    assert measurement.entity_id != prior.entity_id
