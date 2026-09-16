"""Framework-neutral contracts exchanged by Scholar agents and the Flow.

These are intentionally Pydantic models so CrewAI ``Task.output_pydantic``
and non-CrewAI backends share exactly the same validation boundary.  Domain
objects are carried as opaque values where the existing domain model is the
source of truth; they are never replaced by a second evidence or patch model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Route = Literal[
    "RESEARCH",
    "SUPPORT_CLAIM",
    "WRITE_INTRODUCTION",
    "WRITE_CONCLUSION",
    "WRITE_ABSTRACT",
    "REVIEW",
]
_VALID_ROUTES = frozenset(
    {
        "RESEARCH",
        "SUPPORT_CLAIM",
        "WRITE_INTRODUCTION",
        "WRITE_CONCLUSION",
        "WRITE_ABSTRACT",
        "REVIEW",
    }
)
AgentStatus = Literal[
    "COMPLETED",
    "READY",
    "PARTIAL",
    "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT_MANUSCRIPT_STATE",
    "RESEARCH_REQUIRED",
    "FAILED",
]
OrchestrationStatus = Literal[
    "ROUTING",
    "RESEARCHING",
    "WRITING",
    "REVIEWING",
    "REVISION_REQUIRED",
    "WAITING_HUMAN_APPROVAL",
    "APPLYING",
    "COMPLETED",
    "FAILED",
    "INTERRUPTED",
]


class ContractModel(BaseModel):
    """Strict structured-output base used at every agent boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class ResearchAgentOutput(ContractModel):
    status: AgentStatus
    research_summary: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    freshness_status: str = "UNKNOWN"
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    # EvidencePack is the canonical domain result.  It is intentionally not
    # converted into another schema just for CrewAI.
    evidence_packs: list[Any] = Field(default_factory=list)
    domain_result: Any | None = None


class WriterAgentOutput(ContractModel):
    status: AgentStatus
    draft_patch: Any | None = None
    research_required: bool = False
    missing_context: list[str] = Field(default_factory=list)
    used_evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    domain_result: Any | None = None


class ReviewAgentOutput(ContractModel):
    decision: Literal["PASS", "REVISE", "REJECT"]
    issues: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    fact_conflicts: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    review_report: Any | None = None


class SupervisorResult(ContractModel):
    status: AgentStatus | Literal["WAITING_HUMAN_APPROVAL"]
    selected_route: Route | None = None
    final_answer: str | None = None
    pending_action: dict[str, Any] | None = None
    specialist_results: dict[str, Any] = Field(default_factory=dict)
    approval_required: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class OrchestrationRequest(ContractModel):
    request_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    session_id: str | None = None
    task_type: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "project_id", "instruction", mode="before")
    @classmethod
    def _strip_required(cls, value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("request_id/project_id/instruction 不能为空。")
        return value.strip()

    @field_validator("task_type", mode="before")
    @classmethod
    def _normalize_task_type(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("task_type 必须是字符串。")
        normalized = value.strip().upper()
        if normalized not in _VALID_ROUTES:
            raise ValueError(
                "task_type 只允许 RESEARCH、SUPPORT_CLAIM、WRITE_INTRODUCTION、"
                "WRITE_CONCLUSION、WRITE_ABSTRACT 或 REVIEW。"
            )
        return normalized


class OrchestrationResult(ContractModel):
    status: OrchestrationStatus
    request_id: str
    project_id: str
    session_id: str | None = None
    run_id: str | None = None
    thread_id: str | None = None
    trace_id: str | None = None
    selected_route: Route | None = None
    final_answer: str | None = None
    pending_action: dict[str, Any] | None = None
    specialist_results: dict[str, Any] = Field(default_factory=dict)
    approval_required: bool = False
    error_codes: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    value: Any | None = None
    backend: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        """Serialize without leaking arbitrary domain objects to the API."""

        def serialize(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return {
                    str(key): serialize(child)
                    for key, child in value.model_dump(mode="python").items()
                }
            if hasattr(value, "to_dict") and callable(value.to_dict):
                return serialize(value.to_dict())
            if hasattr(value, "__dataclass_fields__"):
                from dataclasses import asdict

                return serialize(asdict(value))
            if isinstance(value, dict):
                return {str(key): serialize(child) for key, child in value.items()}
            if isinstance(value, (list, tuple, set, frozenset)):
                return [serialize(child) for child in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        return serialize(self.model_dump(mode="python"))
