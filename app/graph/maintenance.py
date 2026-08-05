"""Graph invalidation, cleanup, and operational status helpers."""

from __future__ import annotations

def graph_counts(driver: object, database: str, epoch: int) -> dict[str, int]:
    labels = ("Work", "Document", "Section", "Chunk", "Entity", "RelationClaim", "Community")
    counts: dict[str, int] = {}
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        for label in labels:
            # Labels are fixed constants, never user-controlled input.
            record = session.run(
                f"MATCH (n:{label}) WHERE n.valid_from_epoch IS NULL OR "
                "(n.valid_from_epoch <= $epoch AND (n.valid_to_epoch IS NULL OR $epoch < n.valid_to_epoch)) "
                "RETURN count(n) AS count", epoch=epoch,
            ).single()
            counts[label] = int(record["count"] if record else 0)
    return counts


def cleanup_failed_epoch(driver: object, database: str, epoch: int) -> dict[str, int]:
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        claims = session.run(
            "MATCH (n:RelationClaim {valid_from_epoch:$epoch}) DETACH DELETE n RETURN count(n) AS count",
            epoch=epoch,
        ).single()
        chunks = session.run(
            "MATCH (n:Chunk {valid_from_epoch:$epoch}) DETACH DELETE n RETURN count(n) AS count",
            epoch=epoch,
        ).single()
        communities = session.run(
            "MATCH (n:Community {valid_from_epoch:$epoch}) DETACH DELETE n RETURN count(n) AS count",
            epoch=epoch,
        ).single()
    return {"claims": int(claims["count"] if claims else 0),
            "chunks": int(chunks["count"] if chunks else 0),
            "communities": int(communities["count"] if communities else 0)}
