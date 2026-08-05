"""Idempotent source graph and RelationClaim persistence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.graph.models import ChunkGraphExtraction
from app.graph.resolution import EntityResolver, ResolvedEntity


def claim_id_for(
    chunk_key: str, subject_id: str, predicate: str, object_id: str,
    evidence_quote: str, qualifiers: dict[str, Any], polarity: str,
) -> str:
    payload = "\x1f".join((chunk_key, subject_id, predicate, object_id,
                            evidence_quote, json.dumps(qualifiers, sort_keys=True,
                                                       ensure_ascii=False), polarity))
    return "RC_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def persist_chunk_graph(
    driver: object, database: str, chunk: dict[str, Any], extraction: ChunkGraphExtraction,
    resolver: EntityResolver, epoch: int, *, extractor_model: str,
    extractor_prompt_version: str, ontology_version: str,
) -> dict[str, int]:
    resolved: dict[str, ResolvedEntity] = {}
    created = reused = 0
    for entity in extraction.entities:
        value = resolver.resolve(entity.name, entity.entity_type, epoch, entity.aliases)
        resolved[entity.local_id] = value
        created += int(value.created)
        reused += int(not value.created)
    section_id = str(chunk.get("section_id") or "")
    source_params = {
        "work_id": chunk["work_id"], "document_id": chunk["document_id"],
        "paper_id": chunk["paper_id"], "section_id": section_id,
        "section_path": chunk.get("section_path") or [], "chunk_key": chunk["chunk_key"],
        "chunk_id": chunk["chunk_id"], "title": chunk.get("title"),
        "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"),
        "block_ids": chunk.get("block_ids") or [], "content": chunk.get("content"),
        "valid_from_epoch": epoch,
    }
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        session.run(
            """MERGE (w:Work {work_id:$work_id})
            MERGE (d:Document {document_id:$document_id})
            SET d.paper_id=$paper_id
            MERGE (s:Section {section_id:$section_id})
            SET s.section_path=$section_path
            MERGE (c:Chunk {chunk_key:$chunk_key})
            SET c += {chunk_id:$chunk_id,document_id:$document_id,work_id:$work_id,
              paper_id:$paper_id,title:$title,page_start:$page_start,page_end:$page_end,
              block_ids:$block_ids,content:$content,valid_from_epoch:$valid_from_epoch,
              valid_to_epoch:null}
            MERGE (w)-[:HAS_DOCUMENT]->(d)
            MERGE (d)-[:HAS_SECTION]->(s)
            MERGE (s)-[:HAS_CHUNK]->(c)""", **source_params
        ).consume()
        for entity in extraction.entities:
            value = resolved[entity.local_id]
            session.run(
                """MATCH (c:Chunk {chunk_key:$chunk_key})
                MERGE (e:Entity {entity_id:$entity_id})
                ON CREATE SET e.created_epoch=$epoch
                SET e.canonical_name=$canonical_name,e.normalized_name=$normalized_name,
                    e.entity_type=$entity_type,e.aliases=$aliases,e.aliases_text=$aliases_text,
                    e.description=$description,e.updated_epoch=$epoch
                MERGE (c)-[:MENTIONS]->(e)""",
                chunk_key=chunk["chunk_key"], entity_id=value.entity_id,
                canonical_name=value.canonical_name, normalized_name=value.normalized_name,
                entity_type=value.entity_type.value, aliases=entity.aliases,
                aliases_text=" ".join(entity.aliases), description=entity.description, epoch=epoch,
            ).consume()
        claims = 0
        for relation in extraction.relations:
            subject = resolved[relation.subject_local_id]
            object_value = resolved[relation.object_local_id]
            claim_id = claim_id_for(chunk["chunk_key"], subject.entity_id,
                                    relation.predicate.value, object_value.entity_id,
                                    relation.evidence_quote, relation.qualifiers,
                                    relation.polarity)
            params = {
                "claim_id": claim_id, "predicate": relation.predicate.value,
                "description": relation.description, "polarity": relation.polarity,
                "qualifiers_json": json.dumps(relation.qualifiers, ensure_ascii=False,
                                               sort_keys=True),
                "confidence": relation.confidence, "relation_mode": "direct",
                "evidence_text": relation.evidence_quote, "chunk_key": chunk["chunk_key"],
                "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                "work_id": chunk["work_id"], "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"), "block_ids": chunk.get("block_ids") or [],
                "extractor_model": extractor_model,
                "extractor_prompt_version": extractor_prompt_version,
                "ontology_version": ontology_version, "epoch": epoch,
                "subject_id": subject.entity_id, "object_id": object_value.entity_id,
            }
            session.run(
                """MATCH (c:Chunk {chunk_key:$chunk_key}),
                      (s:Entity {entity_id:$subject_id}),(o:Entity {entity_id:$object_id})
                MERGE (rc:RelationClaim {claim_id:$claim_id})
                SET rc += {predicate:$predicate,description:$description,polarity:$polarity,
                  qualifiers_json:$qualifiers_json,confidence:$confidence,
                  relation_mode:$relation_mode,evidence_text:$evidence_text,
                  chunk_key:$chunk_key,chunk_id:$chunk_id,document_id:$document_id,
                  work_id:$work_id,page_start:$page_start,page_end:$page_end,
                  block_ids:$block_ids,extractor_model:$extractor_model,
                  extractor_prompt_version:$extractor_prompt_version,
                  ontology_version:$ontology_version,valid_from_epoch:$epoch,
                  valid_to_epoch:null}
                MERGE (c)-[:SUPPORTS]->(rc)
                MERGE (rc)-[:SUBJECT]->(s)
                MERGE (rc)-[:OBJECT]->(o)""", **params
            ).consume()
            claims += 1
    return {"entities_created": created, "entities_reused": reused,
            "relation_claims_created": claims}


def invalidate_chunk_graph(driver: object, database: str, chunk_key: str, epoch: int) -> int:
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        record = session.run(
            """MATCH (c:Chunk {chunk_key:$chunk_key})
            OPTIONAL MATCH (c)-[:SUPPORTS]->(rc:RelationClaim)
            WHERE rc.valid_to_epoch IS NULL
            SET c.valid_to_epoch=$epoch, rc.valid_to_epoch=$epoch
            RETURN count(rc) AS invalidated""", chunk_key=chunk_key, epoch=epoch
        ).single()
    return int(record["invalidated"] if record else 0)
