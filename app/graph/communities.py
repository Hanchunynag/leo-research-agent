"""Deterministic offline Leiden communities with stable fingerprints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
@dataclass(frozen=True)
class DetectedCommunity:
    community_id: str
    level: int
    parent_community_id: str | None
    fingerprint: str
    entity_ids: tuple[str, ...]
    predicates: tuple[str, ...]


def community_fingerprint(entity_ids: list[str] | tuple[str, ...],
                          predicates: list[str] | tuple[str, ...]) -> str:
    payload = "entities=" + "\x1e".join(sorted(set(entity_ids))) + \
              "\x1fpredicates=" + "\x1e".join(sorted(set(predicates)))
    return hashlib.sha256(payload.encode()).hexdigest()


def detect_communities(
    entity_ids: list[str], edges: list[tuple[str, str, str, float]], *, seed: int = 42,
) -> list[DetectedCommunity]:
    if not entity_ids:
        return []
    import igraph as ig
    import leidenalg

    ordered = sorted(set(entity_ids))
    index = {entity_id: position for position, entity_id in enumerate(ordered)}
    graph = ig.Graph(n=len(ordered), directed=False)
    graph.vs["entity_id"] = ordered
    valid_edges = [(index[left], index[right]) for left, right, _, _ in edges
                   if left in index and right in index and left != right]
    graph.add_edges(valid_edges)
    weights = [max(0.0001, float(weight)) for left, right, _, weight in edges
               if left in index and right in index and left != right]
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights=weights or None, seed=seed,
    )
    output: list[DetectedCommunity] = []
    for members in partition:
        ids = tuple(sorted(ordered[position] for position in members))
        member_set = set(ids)
        predicates = tuple(sorted({predicate for left, right, predicate, _ in edges
                                   if left in member_set and right in member_set}))
        fingerprint = community_fingerprint(ids, predicates)
        output.append(DetectedCommunity("C_" + fingerprint[:24], 0, None,
                                        fingerprint, ids, predicates))
    output.sort(key=lambda value: value.community_id)
    return output


def persist_communities(driver: object, database: str, epoch: int,
                        communities: list[DetectedCommunity]) -> int:
    with driver.session(database=database) as session:  # type: ignore[attr-defined]
        for community in communities:
            session.run(
                """MERGE (c:Community {community_id:$community_id})
                SET c.level=$level,c.parent_community_id=$parent,c.fingerprint=$fingerprint,
                    c.entity_count=size($entity_ids),c.valid_from_epoch=$epoch,
                    c.valid_to_epoch=null
                WITH c UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id:entity_id}) MERGE (e)-[:IN_COMMUNITY]->(c)""",
                community_id=community.community_id, level=community.level,
                parent=community.parent_community_id, fingerprint=community.fingerprint,
                entity_ids=list(community.entity_ids), epoch=epoch,
            ).consume()
    return len(communities)
