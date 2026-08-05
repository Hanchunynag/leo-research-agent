"""Graph source integrity and path completeness validation."""

from __future__ import annotations

from typing import Any

from app.graph.models import EvidenceCandidate


def validate_path_evidence(candidate: EvidenceCandidate) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if candidate.candidate_type != "graph_path":
        return True, issues
    edge_claims = candidate.metadata.get("edge_claims")
    if not isinstance(edge_claims, list) or len(edge_claims) != 2:
        issues.append("two-hop path must contain exactly two edge claims")
        return False, issues
    for index, claim in enumerate(edge_claims, 1):
        if not isinstance(claim, dict):
            issues.append(f"edge {index} claim is invalid")
            continue
        for field in ("claim_id", "chunk_key", "evidence_text", "predicate"):
            if not claim.get(field):
                issues.append(f"edge {index} missing {field}")
    known = {str(value.get("claim_id")) for value in edge_claims if isinstance(value, dict)}
    if set(candidate.source_claim_ids) != known:
        issues.append("path source_claim_ids do not match edge claims")
    return not issues, issues


def validate_graph_sources(driver: object, database: str, epoch: int) -> dict[str, Any]:
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        row = session.run(
            """MATCH (rc:RelationClaim)
            WHERE rc.valid_from_epoch <= $epoch
              AND (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)
            OPTIONAL MATCH (c:Chunk)-[:SUPPORTS]->(rc)
            OPTIONAL MATCH (rc)-[:SUBJECT]->(s:Entity)
            OPTIONAL MATCH (rc)-[:OBJECT]->(o:Entity)
            RETURN count(rc) AS claims,
              sum(CASE WHEN c IS NULL THEN 1 ELSE 0 END) AS missing_chunks,
              sum(CASE WHEN s IS NULL OR o IS NULL THEN 1 ELSE 0 END) AS missing_endpoints,
              sum(CASE WHEN rc.page_start IS NULL OR size(rc.block_ids)=0 THEN 1 ELSE 0 END)
                AS missing_provenance""", epoch=epoch,
        ).single()
    result = dict(row) if row else {"claims": 0, "missing_chunks": 0,
                                    "missing_endpoints": 0, "missing_provenance": 0}
    result["valid"] = not any(result.get(key) for key in
                              ("missing_chunks", "missing_endpoints", "missing_provenance"))
    return result
