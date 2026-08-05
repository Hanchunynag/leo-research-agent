"""Strict models for extraction, graph evidence, and community reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.graph.ontology import EntityType, RelationPredicate


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedEntity(StrictModel):
    local_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class ExtractedRelation(StrictModel):
    subject_local_id: str = Field(min_length=1)
    predicate: RelationPredicate
    object_local_id: str = Field(min_length=1)
    description: str = ""
    polarity: Literal["support", "oppose", "neutral"] = "neutral"
    qualifiers: dict[str, str | float | int | bool] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str = Field(min_length=1)


class ChunkGraphExtraction(StrictModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def unique_local_ids(cls, values: list[ExtractedEntity]) -> list[ExtractedEntity]:
        ids = [value.local_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("entity local_id must be unique")
        return values


class CommunityFinding(StrictModel):
    statement: str
    source_claim_ids: list[str]


class CommunityReport(StrictModel):
    title: str
    summary: str
    key_entities: list[str]
    key_relationships: list[str]
    findings: list[CommunityFinding]
    contradictions: list[str]
    source_claim_ids: list[str]


CandidateType = Literal["chunk", "relation_claim", "graph_path", "community_report"]


class EvidenceCandidate(StrictModel):
    evidence_id: str
    candidate_type: CandidateType
    text: str
    source_chunk_keys: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    per_query_ranks: dict[str, int] = Field(default_factory=dict)
    per_route_scores: dict[str, float] = Field(default_factory=dict)
    fusion_score: float = 0.0
    directness_grade: int | None = Field(default=None, ge=0, le=3)
    reranker_score: float | None = None
    original_fusion_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
