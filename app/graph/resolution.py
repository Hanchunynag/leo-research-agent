"""Persistent type-aware entity resolution with guarded fuzzy matching."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from rapidfuzz import fuzz, process

from app.graph.ontology import EntityType
from app.index_registry.store import IndexRegistryStore


ENTITY_NAMESPACE = uuid.UUID("679769dc-578f-4ef8-bc7b-f18485b8282a")
DOMAIN_ALIASES = {
    "extended kalman filter": "ekf", "扩展卡尔曼滤波": "ekf",
    "error-state extended kalman filter": "esekf", "误差状态扩展卡尔曼滤波": "esekf",
    "pseudo-range": "pseudorange", "伪距": "pseudorange",
    "pseudorange-rate": "pseudorange rate", "伪距率": "pseudorange rate",
    "two line element": "tle", "两行根数": "tle",
}
FORBIDDEN_MERGES = {
    frozenset(("ekf", "esekf")), frozenset(("pseudorange", "pseudorange rate")),
    frozenset(("orbit error", "clock error")), frozenset(("sgp4", "hpop")),
    frozenset(("measurement", "prior")),
}


def normalize_entity_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[_/]+", " ", normalized)
    normalized = re.sub(r"[^\w\u3400-\u9fff.+-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .-")
    return DOMAIN_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    canonical_name: str
    normalized_name: str
    entity_type: EntityType
    strategy: str
    score: float
    created: bool


class EntityResolver:
    def __init__(
        self, registry: IndexRegistryStore, *, fuzzy_threshold: float = 94.0,
        vector_matcher: Callable[[str, EntityType], tuple[str, float] | None] | None = None,
        adjudicator: Callable[[str, str, EntityType], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.fuzzy_threshold = fuzzy_threshold
        self.vector_matcher = vector_matcher
        self.adjudicator = adjudicator

    def _candidates(self, entity_type: EntityType) -> list[sqlite3.Row]:
        with self.registry.connect() as connection:
            return connection.execute(
                "SELECT * FROM entity_aliases WHERE entity_type=? AND active=1",
                (entity_type.value,),
            ).fetchall()

    def resolve(self, name: str, entity_type: EntityType | str, epoch: int,
                aliases: list[str] | None = None) -> ResolvedEntity:
        kind = EntityType(entity_type)
        normalized = normalize_entity_name(name)
        if not normalized:
            raise ValueError("entity name normalizes to empty")
        candidates = self._candidates(kind)
        exact = next((row for row in candidates if row["alias_normalized"] == normalized), None)
        strategy, score, selected = "new", 0.0, exact
        if exact is not None:
            strategy, score = "exact_alias", 1.0
        elif candidates:
            choices = {str(row["entity_id"]): str(row["alias_normalized"]) for row in candidates
                       if frozenset((normalized, str(row["alias_normalized"]))) not in FORBIDDEN_MERGES}
            match = process.extractOne(normalized, choices, scorer=fuzz.WRatio)
            if match and float(match[1]) >= self.fuzzy_threshold:
                selected = next(row for row in candidates if row["entity_id"] == match[2])
                strategy, score = "rapidfuzz", float(match[1]) / 100.0
        if selected is None and self.vector_matcher is not None:
            vector_match = self.vector_matcher(name, kind)
            if vector_match and vector_match[1] >= 0.92:
                selected = next((row for row in candidates if row["entity_id"] == vector_match[0]), None)
                if selected is not None:
                    strategy, score = "description_vector", vector_match[1]
        if selected is not None and strategy in {"rapidfuzz", "description_vector"} and self.adjudicator:
            if not self.adjudicator(name, str(selected["canonical_name"]), kind):
                selected = None
        created = selected is None
        entity_id = (str(uuid.uuid4()) if created else str(selected["entity_id"]))
        canonical = name.strip() if created else str(selected["canonical_name"])
        all_aliases = {normalized, *(normalize_entity_name(value) for value in aliases or [])}
        with self.registry.transaction() as connection:
            for alias in sorted(value for value in all_aliases if value):
                connection.execute(
                    """INSERT OR IGNORE INTO entity_aliases(alias_normalized,entity_type,
                    entity_id,canonical_name,active,created_epoch) VALUES(?,?,?,?,1,?)""",
                    (alias, kind.value, entity_id, canonical, epoch),
                )
            event_id = str(uuid.uuid5(ENTITY_NAMESPACE,
                f"{epoch}\x1f{name}\x1f{kind.value}\x1f{entity_id}\x1f{strategy}"))
            connection.execute(
                """INSERT OR IGNORE INTO entity_resolution_events(event_id,epoch,local_name,
                entity_type,entity_id,strategy,score,details_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (event_id, epoch, name, kind.value, entity_id, strategy, score,
                 json.dumps({"aliases": aliases or []}, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat()),
            )
        return ResolvedEntity(entity_id, canonical, normalize_entity_name(canonical), kind,
                              strategy, score, created)

    def merge(self, source_id: str, target_id: str, epoch: int, reason: str) -> str:
        merge_id = str(uuid.uuid4())
        with self.registry.transaction() as connection:
            connection.execute("UPDATE entity_aliases SET entity_id=? WHERE entity_id=?",
                               (target_id, source_id))
            connection.execute(
                """INSERT INTO entity_merge_events(merge_id,epoch,source_entity_id,
                target_entity_id,reason,reversible,created_at) VALUES(?,?,?,?,?,1,?)""",
                (merge_id, epoch, source_id, target_id, reason,
                 datetime.now(timezone.utc).isoformat()),
            )
        return merge_id

    def rollback_merge(self, merge_id: str) -> None:
        with self.registry.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM entity_merge_events WHERE merge_id=? AND rolled_back_at IS NULL",
                (merge_id,),
            ).fetchone()
            if row is None:
                raise KeyError(merge_id)
            connection.execute("UPDATE entity_aliases SET entity_id=? WHERE entity_id=?",
                               (row["source_entity_id"], row["target_entity_id"]))
            connection.execute("UPDATE entity_merge_events SET rolled_back_at=? WHERE merge_id=?",
                               (datetime.now(timezone.utc).isoformat(), merge_id))
