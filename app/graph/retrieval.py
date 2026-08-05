"""Parameterized local/direct/two-hop Neo4j retrieval with source-complete paths."""

from __future__ import annotations

import hashlib
from typing import Any

from app.graph.models import EvidenceCandidate
from app.graph.resolution import normalize_entity_name


def _evidence_id(prefix: str, values: list[str]) -> str:
    return prefix + hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:20]


def _claim_text(value: dict[str, Any]) -> str:
    qualifiers = value.get("qualifiers_json") or "{}"
    return (f"{value.get('subject_name')} --{value.get('predicate')}--> "
            f"{value.get('object_name')}\nPolarity: {value.get('polarity')}\n"
            f"Qualifiers: {qualifiers}\nEvidence: {value.get('evidence_text')}")


class GraphRetriever:
    def __init__(self, driver: object, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    def link_entity(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized = normalize_entity_name(name)
        with self.driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            exact = session.run(
                """MATCH (e:Entity) WHERE e.normalized_name=$name OR $name IN e.aliases
                RETURN e.entity_id AS entity_id,e.canonical_name AS canonical_name,
                       e.entity_type AS entity_type,1.0 AS score LIMIT $limit""",
                name=normalized, limit=limit,
            ).data()
            if exact:
                return exact
            return session.run(
                """CALL db.index.fulltext.queryNodes('entity_names',$query,{limit:$limit})
                YIELD node,score RETURN node.entity_id AS entity_id,
                node.canonical_name AS canonical_name,node.entity_type AS entity_type,score""",
                query=name, limit=limit,
            ).data()

    def local(self, entity_id: str, epoch: int, limit: int = 20) -> list[EvidenceCandidate]:
        with self.driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            rows = session.run(
                """MATCH (s:Entity)<-[:SUBJECT]-(rc:RelationClaim)-[:OBJECT]->(o:Entity)
                WHERE (s.entity_id=$entity_id OR o.entity_id=$entity_id)
                  AND rc.valid_from_epoch <= $epoch
                  AND (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)
                RETURN rc{.*} AS claim,s.entity_id AS subject_id,s.canonical_name AS subject_name,
                       o.entity_id AS object_id,o.canonical_name AS object_name
                ORDER BY rc.confidence DESC,rc.claim_id LIMIT $limit""",
                entity_id=entity_id, epoch=epoch, limit=limit,
            ).data()
        return [self._claim_candidate(row) for row in rows]

    def direct_relation(
        self, entity_a: str, entity_b: str, epoch: int, limit: int = 20,
    ) -> list[EvidenceCandidate]:
        with self.driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            rows = session.run(
                """MATCH (s:Entity)<-[:SUBJECT]-(rc:RelationClaim)-[:OBJECT]->(o:Entity)
                WHERE ((s.entity_id=$a AND o.entity_id=$b) OR
                       (s.entity_id=$b AND o.entity_id=$a))
                  AND rc.valid_from_epoch <= $epoch
                  AND (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)
                RETURN rc{.*} AS claim,s.entity_id AS subject_id,s.canonical_name AS subject_name,
                       o.entity_id AS object_id,o.canonical_name AS object_name
                ORDER BY rc.confidence DESC,rc.claim_id LIMIT $limit""",
                a=entity_a, b=entity_b, epoch=epoch, limit=limit,
            ).data()
        return [self._claim_candidate(row) for row in rows]

    def two_hop_paths(
        self, entity_a: str, entity_b: str, epoch: int, limit: int = 20,
    ) -> list[EvidenceCandidate]:
        with self.driver.session(database=self.database) as session:  # type: ignore[attr-defined]
            rows = session.run(
                """MATCH (a:Entity {entity_id:$a})
                MATCH (b:Entity {entity_id:$b})
                MATCH (a)<-[:SUBJECT]-(r1:RelationClaim)-[:OBJECT]->(m:Entity)
                MATCH (m)<-[:SUBJECT]-(r2:RelationClaim)-[:OBJECT]->(b)
                WHERE r1.valid_from_epoch <= $epoch AND r2.valid_from_epoch <= $epoch
                  AND (r1.valid_to_epoch IS NULL OR $epoch < r1.valid_to_epoch)
                  AND (r2.valid_to_epoch IS NULL OR $epoch < r2.valid_to_epoch)
                RETURN a{.entity_id,.canonical_name} AS a,m{.entity_id,.canonical_name} AS m,
                       b{.entity_id,.canonical_name} AS b,r1{.*} AS r1,r2{.*} AS r2
                UNION
                MATCH (a:Entity {entity_id:$a})
                MATCH (b:Entity {entity_id:$b})
                MATCH (b)<-[:SUBJECT]-(r2:RelationClaim)-[:OBJECT]->(m:Entity)
                MATCH (m)<-[:SUBJECT]-(r1:RelationClaim)-[:OBJECT]->(a)
                WHERE r1.valid_from_epoch <= $epoch AND r2.valid_from_epoch <= $epoch
                  AND (r1.valid_to_epoch IS NULL OR $epoch < r1.valid_to_epoch)
                  AND (r2.valid_to_epoch IS NULL OR $epoch < r2.valid_to_epoch)
                RETURN a{.entity_id,.canonical_name} AS a,m{.entity_id,.canonical_name} AS m,
                       b{.entity_id,.canonical_name} AS b,r1{.*} AS r1,r2{.*} AS r2
                LIMIT $limit""", a=entity_a, b=entity_b, epoch=epoch, limit=limit,
            ).data()
        candidates: list[EvidenceCandidate] = []
        for row in rows:
            claim1, claim2 = row["r1"], row["r2"]
            claim_ids = [str(claim1["claim_id"]), str(claim2["claim_id"])]
            chunk_keys = list(dict.fromkeys((str(claim1["chunk_key"]), str(claim2["chunk_key"]))))
            names = [row["a"]["canonical_name"], row["m"]["canonical_name"],
                     row["b"]["canonical_name"]]
            text = (f"{names[0]} --{claim1['predicate']}--> {names[1]} "
                    f"--{claim2['predicate']}--> {names[2]}\n\n"
                    f"Edge 1 evidence:\n{claim1['evidence_text']}\n\n"
                    f"Edge 2 evidence:\n{claim2['evidence_text']}")
            candidates.append(EvidenceCandidate(
                evidence_id=_evidence_id("GP_", claim_ids), candidate_type="graph_path",
                text=text, source_chunk_keys=chunk_keys, source_claim_ids=claim_ids,
                entity_ids=[row[key]["entity_id"] for key in ("a", "m", "b")],
                relation_ids=claim_ids, routes=["graph_path"], directness_grade=2,
                metadata={"mode": "inferred_path", "hop_count": 2,
                          "edge_claims": [claim1, claim2]},
            ))
        return candidates

    def relationship_search(
        self, entity_a: str, entity_b: str, epoch: int, max_paths: int = 20,
    ) -> dict[str, Any]:
        direct = self.direct_relation(entity_a, entity_b, epoch, max_paths)
        if direct:
            return {"mode": "direct", "candidates": direct, "refusal_reason": None}
        paths = self.two_hop_paths(entity_a, entity_b, epoch, max_paths)
        if paths:
            return {"mode": "inferred_path", "candidates": paths,
                    "disclaimer": "现有文献没有直接给出完整的 A-B 表述；以下关系由两组有来源证据共同支持。",
                    "refusal_reason": None}
        return {"mode": "none", "candidates": [],
                "refusal_reason": "当前知识库分别包含 A 和 B 的定义，但没有检索到能够证明二者关系的直接证据或可靠路径。"}

    @staticmethod
    def _claim_candidate(row: dict[str, Any]) -> EvidenceCandidate:
        claim = dict(row["claim"])
        claim.update({key: row[key] for key in
                      ("subject_id", "subject_name", "object_id", "object_name")})
        claim_id = str(claim["claim_id"])
        return EvidenceCandidate(
            evidence_id=claim_id, candidate_type="relation_claim", text=_claim_text(claim),
            source_chunk_keys=[str(claim["chunk_key"])], source_claim_ids=[claim_id],
            entity_ids=[str(claim["subject_id"]), str(claim["object_id"])],
            relation_ids=[claim_id], routes=["graph_direct"], directness_grade=3,
            metadata={"mode": "direct", "claim": claim},
        )
