"""Agentic Scientific RAG 各阶段的结构化 Pydantic 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TopicRelation = Literal["same_topic", "related_subtopic", "new_topic"]
EvidenceStatus = Literal["sufficient", "partial", "missing"]
EntailmentLabel = Literal[
    "entailed",
    "partially_entailed",
    "not_entailed",
    "contradicted",
]
RepairAction = Literal["keep", "rewrite", "drop", "retrieve_more"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopicRoute(StrictModel):
    relation: TopicRelation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    context_dependent: bool
    standalone_query: str
    reuse_previous_evidence: bool
    requires_new_retrieval: bool
    signals: dict[str, float] = Field(default_factory=dict)


class PlannedSubquestion(StrictModel):
    id: str
    question: str


class EvidenceRequirement(StrictModel):
    subquestion_id: str
    requirement: str


class QueryPlan(StrictModel):
    intent: Literal[
        "fact_list",
        "definition",
        "mechanism",
        "comparison",
        "method",
        "numeric_result",
        "citation_lookup",
        "synthesis",
    ]
    target_category: str
    excluded_categories: list[str]
    subquestions: list[PlannedSubquestion]
    retrieval_queries: list[str]
    required_evidence: list[EvidenceRequirement]
    answer_constraints: list[str]


class CoverageItem(StrictModel):
    subquestion_id: str
    status: EvidenceStatus
    supporting_evidence_ids: list[str]
    missing_information: str = ""


class CoverageReport(StrictModel):
    overall_sufficient: bool
    coverage: list[CoverageItem]
    followup_queries: list[str]


class SemanticClaimResult(StrictModel):
    claim_id: str
    entailment: EntailmentLabel
    query_aligned: bool
    category_correct: bool
    citation_direct: bool
    reason: str
    repair_action: RepairAction
    revised_claim: str | None = None


class SemanticValidationReport(StrictModel):
    valid: bool
    issues: list[str]
    structural_valid: bool
    semantic_valid: bool
    claim_results: list[SemanticClaimResult]
    requires_retrieval: bool
    followup_queries: list[str]


class AgenticClaim(StrictModel):
    claim_id: str
    text: str
    category: str
    source_ids: list[str]
    evidence_ids: list[str]


class AgenticAnswerDraft(StrictModel):
    answerable: bool
    claims: list[AgenticClaim]
    refusal_reason: str | None = None


class RoutingLLMDecision(StrictModel):
    relation: TopicRelation
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    context_dependent: bool
    standalone_query: str
    reuse_previous_evidence: bool
    requires_new_retrieval: bool


class CompactionReport(StrictModel):
    before_tokens: int
    after_tokens: int
    retained_evidence_ids: list[str]
    discarded_event_types: list[str]
    compaction_ordinal: int


class RetrievalRound(StrictModel):
    round: int
    queries: list[str]
    candidate_count: int
    reranked_count: int
    new_evidence_count: int
    coverage_status: EvidenceStatus


class AgenticMetricsRecord(StrictModel):
    retrieval_rounds: int
    new_evidence_count: int
    reused_evidence_count: int
    coverage_sufficient: bool
    citation_count: int
    entailed_claim_count: int
    total_claim_count: int
    cache_hit_rate: float | None = None
    latency_ms: float | None = None
    retrieval_recall_at_k: float | None = None
    reranker_recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    claim_entailment_accuracy: float | None = None
    answerable_correct: bool | None = None
    first_turn: bool | None = None
    compaction_count: int = 0


def stable_model_dump(value: BaseModel) -> dict[str, Any]:
    """返回适合 sort_keys 稳定序列化的 JSON 数据。"""

    return value.model_dump(mode="json")
