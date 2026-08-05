"""Neo4j 5.26-compatible constraints and indexes (no APOC/GDS)."""

from __future__ import annotations


CONSTRAINTS = (
    "CREATE CONSTRAINT work_id_unique IF NOT EXISTS FOR (n:Work) REQUIRE n.work_id IS UNIQUE",
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.document_id IS UNIQUE",
    "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (n:Section) REQUIRE n.section_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_key_unique IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_key IS UNIQUE",
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT relation_claim_id_unique IF NOT EXISTS FOR (n:RelationClaim) REQUIRE n.claim_id IS UNIQUE",
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (n:Community) REQUIRE n.community_id IS UNIQUE",
)

INDEXES = (
    "CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (n:Entity) ON (n.normalized_name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
    "CREATE INDEX claim_predicate IF NOT EXISTS FOR (n:RelationClaim) ON (n.predicate)",
    "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (n:Chunk) ON (n.document_id)",
    "CREATE INDEX chunk_work_id IF NOT EXISTS FOR (n:Chunk) ON (n.work_id)",
)

FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
    "FOR (n:Entity) ON EACH [n.canonical_name, n.normalized_name, n.aliases_text]"
)


def ensure_schema(driver: object, database: str = "neo4j") -> dict[str, int]:
    statements = (*CONSTRAINTS, *INDEXES, FULLTEXT_INDEX)
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        for statement in statements:
            session.run(statement).consume()
    return {"constraints": len(CONSTRAINTS), "indexes": len(INDEXES) + 1}
