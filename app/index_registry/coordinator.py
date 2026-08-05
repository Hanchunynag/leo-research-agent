"""Plan a complete incremental epoch without coupling backend algorithms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from app.index_registry.diff import diff_chunks
from app.index_registry.outbox import make_operation
from app.index_registry.store import IndexRegistryStore


class IndexCoordinator:
    def __init__(self, registry: IndexRegistryStore) -> None:
        self.registry = registry

    def plan(self, epoch: int, chunks: Iterable[dict[str, Any]], *,
             document_ids: set[str] | None = None) -> tuple[list[Any], dict[str, int]]:
        previous = self.registry.visible_chunks()
        if document_ids is not None:
            previous = {key: value for key, value in previous.items()
                        if str(value.get("document_id")) in document_ids}
        diffs = diff_chunks(chunks, previous)
        counts = Counter(item.kind for item in diffs)
        for item in diffs:
            if item.kind == "unchanged":
                continue
            target = item.current or item.previous or {}
            target_hash = str(target.get("content_hash") or target.get("dense_text_hash") or item.chunk_key)
            stores = ("lexical", "dense", "graph")
            for store in stores:
                if store == "dense" and item.kind == "graph_changed" and not item.dense_changed:
                    continue
                operation_type = "invalidate" if item.kind == "deleted" else "upsert"
                operation = make_operation(
                    epoch=epoch, store=store, operation_type=operation_type,
                    object_id=item.chunk_key, target_hash=target_hash,
                    payload={"diff_kind": item.kind, "chunk": item.current,
                             "previous": item.previous},
                )
                self.registry.enqueue(operation)
        return diffs, {name: counts.get(name, 0) for name in
                       ("added", "dense_changed", "graph_changed", "changed", "deleted", "unchanged")}


def _complete_store_operations(registry: IndexRegistryStore, epoch: int, store: str) -> None:
    for operation in registry.list_operations(epoch):
        if operation["store"] == store and operation["status"] != "completed":
            registry.mark_operation(operation["operation_id"], "completed")


class KnowledgeSyncService:
    """Coordinates FTS5, Qdrant and Neo4j; only Registry activation is the commit point."""

    def __init__(self, project_root: Any, registry: IndexRegistryStore,
                 embedding_provider: Any, graph_provider: Any, neo4j_driver: Any,
                 *, neo4j_database: str = "neo4j", extractor_prompt_version: str = "1.0",
                 ontology_version: str = "1.0", community_prompt_version: str = "1.0",
                 extraction_concurrency: int = 2) -> None:
        from pathlib import Path
        self.project_root = Path(project_root).expanduser().resolve()
        self.registry = registry
        self.embedding_provider = embedding_provider
        self.graph_provider = graph_provider
        self.neo4j_driver = neo4j_driver
        self.neo4j_database = neo4j_database
        self.extractor_prompt_version = extractor_prompt_version
        self.ontology_version = ontology_version
        self.community_prompt_version = community_prompt_version
        self.extraction_concurrency = extraction_concurrency

    def sync(self, chunks: Iterable[dict[str, Any]], *, document_id: str | None = None) -> dict[str, Any]:
        from app.graph.aggregation import rebuild_aggregate_edges
        from app.graph.extraction import (
            extract_chunk_graph, extraction_cache_key,
        )
        from app.graph.models import ChunkGraphExtraction
        from app.graph.persistence import invalidate_chunk_graph, persist_chunk_graph
        from app.graph.resolution import EntityResolver
        from app.graph.schema import ensure_schema
        from app.graph.validation import validate_graph_sources
        from app.indexing.incremental_dense import (
            point_id_for_version, sync_incremental_dense,
        )
        from app.indexing.lexical_fts import sync_lexical
        from app.indexing.incremental_entities import entity_vector_matcher, sync_entity_embeddings

        values = [dict(value) for value in chunks
                  if document_id is None or value.get("document_id") == document_id]
        model_name = str(getattr(self.embedding_provider, "model_name", "BAAI/bge-m3"))
        revision = getattr(self.embedding_provider, "revision", None)
        extractor_model = str(getattr(self.graph_provider, "model_name", ""))
        active_before = self.registry.active_epoch()
        epoch = self.registry.create_epoch(
            embedding_model=model_name, embedding_revision=revision,
            chunk_policy_version=str(values[0].get("chunk_policy_version") if values else "2.1"),
            ontology_version=self.ontology_version, extractor_model=extractor_model,
            extractor_prompt_version=self.extractor_prompt_version,
            community_prompt_version=self.community_prompt_version,
        )
        coordinator = IndexCoordinator(self.registry)
        diffs, counts = coordinator.plan(
            epoch, values, document_ids={document_id} if document_id else None
        )
        dense_diffs = diffs
        if active_before is not None:
            old = self.registry.epoch_record(active_before)
            if (old["embedding_model"], old["embedding_revision"]) != (model_name, revision):
                from app.index_registry.models import ChunkDiff
                dense_diffs = [ChunkDiff("dense_changed", item.chunk_key, item.current,
                                         item.previous, True, item.graph_changed)
                               if item.kind == "unchanged" else item for item in diffs]
        metrics: dict[str, Any] = {"epoch": epoch,
            "added_chunks": counts["added"],
            "changed_chunks": counts["dense_changed"] + counts["graph_changed"] + counts["changed"],
            "deleted_chunks": counts["deleted"], "unchanged_chunks": counts["unchanged"]}
        try:
            lexical = sync_lexical(self.project_root, epoch, diffs)
            metrics["lexical"] = lexical.to_dict()
            _complete_store_operations(self.registry, epoch, "lexical")

            dense = sync_incremental_dense(self.project_root, self.embedding_provider, epoch, dense_diffs)
            metrics.update({"embedded_chunks": dense.embedded_count,
                            "dense": dense.to_dict()})
            for item in dense_diffs:
                if item.kind != "unchanged":
                    self.registry.invalidate_embedding(epoch=epoch, chunk_key=item.chunk_key)
                if item.current is not None and (item.kind == "added" or item.dense_changed):
                    chunk = item.current
                    self.registry.put_embedding_entry(epoch=epoch, chunk_key=item.chunk_key,
                        point_id=point_id_for_version(item.chunk_key, chunk["dense_text_hash"],
                                                      model_name, revision),
                        collection_name=dense.collection_name, model_name=model_name,
                        model_revision=revision, dense_text_hash=chunk["dense_text_hash"])
            _complete_store_operations(self.registry, epoch, "dense")

            ensure_schema(self.neo4j_driver, self.neo4j_database)
            resolver = EntityResolver(self.registry, vector_matcher=entity_vector_matcher(
                self.project_root, self.embedding_provider, active_before))
            graph_extracted = entities_created = entities_reused = claims_created = 0
            for item in diffs:
                if item.kind == "unchanged":
                    continue
                invalidate_chunk_graph(self.neo4j_driver, self.neo4j_database,
                                       item.chunk_key, epoch)
                if item.current is None:
                    continue
                chunk = item.current
                cache_key = extraction_cache_key(chunk["graph_text_hash"], extractor_model,
                    self.extractor_prompt_version, self.ontology_version)
                cached = self.registry.get_graph_extraction(cache_key)
                if cached is None:
                    extraction, _ = extract_chunk_graph(self.graph_provider, chunk,
                        prompt_version=self.extractor_prompt_version,
                        ontology_version=self.ontology_version)
                    self.registry.put_graph_extraction(cache_key=cache_key,
                        chunk_key=item.chunk_key, graph_text_hash=chunk["graph_text_hash"],
                        extractor_model=extractor_model,
                        extractor_prompt_version=self.extractor_prompt_version,
                        ontology_version=self.ontology_version,
                        extraction=extraction.model_dump(mode="json"))
                    graph_extracted += 1
                else:
                    extraction = ChunkGraphExtraction.model_validate(cached)
                graph_metrics = persist_chunk_graph(self.neo4j_driver, self.neo4j_database,
                    chunk, extraction, resolver, epoch, extractor_model=extractor_model,
                    extractor_prompt_version=self.extractor_prompt_version,
                    ontology_version=self.ontology_version)
                entities_created += graph_metrics["entities_created"]
                entities_reused += graph_metrics["entities_reused"]
                claims_created += graph_metrics["relation_claims_created"]
            _complete_store_operations(self.registry, epoch, "graph")
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                entity_rows = session.run(
                    """MATCH (e:Entity) WHERE e.updated_epoch=$epoch
                    RETURN e{.entity_id,.canonical_name,.normalized_name,.entity_type,
                             .aliases,.description} AS entity""", epoch=epoch,
                ).data()
            entity_dense = sync_entity_embeddings(self.project_root,
                self.embedding_provider, [dict(value["entity"]) for value in entity_rows], epoch)
            aggregate_count = rebuild_aggregate_edges(self.neo4j_driver,
                                                       self.neo4j_database, epoch)
            community_metrics = self._sync_communities(epoch)
            validation = validate_graph_sources(self.neo4j_driver, self.neo4j_database, epoch)
            if not validation["valid"]:
                raise RuntimeError("graph source validation failed")
            metrics.update({"graph_extracted_chunks": graph_extracted,
                "entities_created": entities_created, "entities_reused": entities_reused,
                "entity_embeddings": entity_dense,
                "relation_claims_created": claims_created,
                "aggregate_relations": aggregate_count, "graph_validation": validation,
                **community_metrics})

            for item in diffs:
                if item.kind == "unchanged":
                    continue
                self.registry.invalidate_chunk(epoch, item.chunk_key)
                if item.current is not None:
                    self.registry.put_chunk_version(epoch, item.current)
            self.registry.activate_epoch(epoch)
            metrics["status"] = "active"
            return metrics
        except Exception as error:
            self.registry.fail_epoch(epoch, type(error).__name__)
            metrics.update({"status": "failed", "error_type": type(error).__name__})
            raise

    def _sync_communities(self, epoch: int) -> dict[str, int]:
        from qdrant_client import QdrantClient
        from app.graph.communities import detect_communities, persist_communities
        from app.graph.models import CommunityReport
        from app.graph.reports import embed_community_reports, generate_community_report
        from app.indexing.incremental_dense import qdrant_index_path

        with self.neo4j_driver.session(database=self.neo4j_database) as session:
            entities = session.run(
                """MATCH (e:Entity) WHERE EXISTS {
                    MATCH (e)<-[:SUBJECT|OBJECT]-(rc:RelationClaim)
                    WHERE rc.valid_from_epoch <= $epoch AND
                      (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)}
                RETURN DISTINCT e.entity_id AS entity_id""", epoch=epoch
            ).data()
            edge_rows = session.run(
                """MATCH (s:Entity)-[r:RELATED]->(o:Entity)
                RETURN s.entity_id AS source,o.entity_id AS target,r.predicate AS predicate,
                       coalesce(r.aggregate_confidence,0.5) AS weight"""
            ).data()
        communities = detect_communities(
            [str(value["entity_id"]) for value in entities],
            [(str(value["source"]), str(value["target"]), str(value["predicate"]),
              float(value["weight"])) for value in edge_rows], seed=42,
        )
        ids = [value.community_id for value in communities]
        with self.neo4j_driver.session(database=self.neo4j_database) as session:
            session.run(
                """MATCH (c:Community) WHERE c.valid_to_epoch IS NULL
                AND NOT c.community_id IN $ids SET c.valid_to_epoch=$epoch""",
                ids=ids, epoch=epoch,
            ).consume()
        persist_communities(self.neo4j_driver, self.neo4j_database, epoch, communities)
        changed: list[tuple[str, str, CommunityReport]] = []
        for community in communities:
            cached = self.registry.get_community_report(community.fingerprint)
            if cached is not None:
                report = CommunityReport.model_validate(cached)
                self.registry.put_community_version(epoch=epoch,
                    community_id=community.community_id, fingerprint=community.fingerprint,
                    report=report.model_dump(mode="json"))
                continue
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                entity_rows = session.run(
                    """MATCH (e:Entity) WHERE e.entity_id IN $ids
                    RETURN e{.entity_id,.canonical_name,.entity_type,.description} AS entity""",
                    ids=list(community.entity_ids),
                ).data()
                claim_rows = session.run(
                    """MATCH (s:Entity)<-[:SUBJECT]-(rc:RelationClaim)-[:OBJECT]->(o:Entity)
                    WHERE s.entity_id IN $ids AND o.entity_id IN $ids
                      AND rc.valid_from_epoch <= $epoch
                      AND (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)
                    RETURN rc{.*} AS claim""", ids=list(community.entity_ids), epoch=epoch,
                ).data()
            claims = [dict(value["claim"]) for value in claim_rows]
            report = generate_community_report(self.graph_provider,
                community_id=community.community_id,
                entities=[dict(value["entity"]) for value in entity_rows], claims=claims)
            self.registry.put_community_version(epoch=epoch,
                community_id=community.community_id, fingerprint=community.fingerprint,
                report=report.model_dump(mode="json"))
            source_chunks = sorted({str(value.get("chunk_key")) for value in claims
                                    if value.get("chunk_key")})
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                session.run(
                    """MATCH (c:Community {community_id:$id})
                    SET c.title=$title,c.summary=$summary,c.full_report=$full_report,
                        c.relation_count=$relation_count,c.source_claim_ids=$claim_ids,
                        c.source_chunk_keys=$chunk_keys""",
                    id=community.community_id, title=report.title, summary=report.summary,
                    full_report=report.model_dump_json(), relation_count=len(claims),
                    claim_ids=report.source_claim_ids, chunk_keys=source_chunks,
                ).consume()
            changed.append((community.community_id, community.fingerprint, report))
        if changed:
            client = QdrantClient(path=str(qdrant_index_path(self.project_root)))
            try:
                embed_community_reports(client, self.embedding_provider, changed, epoch)
            finally:
                client.close()
        return {"communities_changed": len(changed),
                "community_reports_generated": len(changed)}
