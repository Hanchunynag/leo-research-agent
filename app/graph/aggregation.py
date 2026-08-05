"""Rebuild current aggregate RELATED edges without losing source claims."""

from __future__ import annotations


def rebuild_aggregate_edges(driver: object, database: str, epoch: int) -> int:
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        session.run("MATCH (:Entity)-[r:RELATED]->(:Entity) DELETE r").consume()
        record = session.run(
            """MATCH (s:Entity)<-[:SUBJECT]-(rc:RelationClaim)-[:OBJECT]->(o:Entity)
            WHERE rc.valid_from_epoch <= $epoch
              AND (rc.valid_to_epoch IS NULL OR $epoch < rc.valid_to_epoch)
            WITH s,o,rc.predicate AS predicate,
                 sum(CASE rc.polarity WHEN 'support' THEN 1 ELSE 0 END) AS support_count,
                 sum(CASE rc.polarity WHEN 'oppose' THEN 1 ELSE 0 END) AS oppose_count,
                 sum(CASE rc.polarity WHEN 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
                 count(DISTINCT rc.work_id) AS source_work_count,
                 avg(rc.confidence) AS aggregate_confidence
            MERGE (s)-[r:RELATED {predicate:predicate}]->(o)
            SET r.support_count=support_count,r.oppose_count=oppose_count,
                r.neutral_count=neutral_count,r.source_work_count=source_work_count,
                r.aggregate_confidence=aggregate_confidence,r.updated_epoch=$epoch
            RETURN count(r) AS count""", epoch=epoch
        ).single()
    return int(record["count"] if record else 0)
