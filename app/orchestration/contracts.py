"""Framework-neutral contracts exchanged by Scholar agents and the Flow.

These are intentionally Pydantic models so CrewAI ``Task.output_pydantic``
and non-CrewAI backends share exactly the same validation boundary.  Domain
objects are carried as opaque values where the existing domain model is the
source of truth; they are never replaced by a second evidence or patch model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tenancy import LOCAL_PRINCIPAL, TenantPrincipal


Route = Literal[
    "RESEARCH",
    "SUPPORT_CLAIM",
    "WRITE_INTRODUCTION",
    "WRITE_CONCLUSION",
    "WRITE_ABSTRACT",
    "REVIEW",
]

# Manager decisions are deliberately narrower than the public task routes.
# A Manager can propose the next cognitive capability, but it can never
# propose a domain mutation such as APPLY_PATCH.
ManagerAction = Literal[
    "CALL_RESEARCH",
    "CALL_WRITER",
    "CALL_REVIEWER",
    "REQUEST_MORE_EVIDENCE",
    "REQUEST_REVISION",
    "COMPLETE",
]
ManagerTarget = Literal["manager", "research", "writer", "reviewer"]
ManagerCompletionStatus = Literal[
    "IN_PROGRESS",
    "READY_TO_COMPLETE",
    "BLOCKED",
    "COMPLETED",
    "WAITING_HUMAN_APPROVAL",
    "FAILED",
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


class DomainResultReference(ContractModel):
    """Bounded handoff reference kept in Flow/Run state.

    The authoritative domain object remains in the existing service/store. The
    reference is sufficient for Manager planning and replay diagnostics while
    preventing arbitrary domain state from becoming a second CrewAI memory.
    """

    result_id: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    producer: Literal["research", "writer", "reviewer"]
    status: str = Field(min_length=1)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunGoal(ContractModel):
    """Immutable intent anchor for one persisted Run.

    This contract is deliberately separate from conversation messages and
    specialist output.  It is written once when a Run is created and is never
    replaced by a Reviewer instruction, a local edit preference, or a later
    Agent result.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    original_instruction: str = Field(min_length=1, max_length=20_000)
    task_type: str = Field(min_length=1, max_length=128)
    success_criteria: tuple[str, ...] = Field(min_length=1)
    hard_constraints: tuple[str, ...] = Field(default_factory=tuple)


class RunBudget(ContractModel):
    """Finite execution budget enforced outside Manager cognition."""

    max_manager_steps: int = Field(default=12, ge=1)
    max_research_rounds: int = Field(default=2, ge=0)
    max_review_rounds: int = Field(default=3, ge=0)
    max_tool_calls: int = Field(default=16, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)


class RunState(ContractModel):
    """Authoritative bounded Workflow State for a Run.

    Only references and bounded summaries are persisted here.  Evidence,
    DraftPatch and ReviewReport remain owned by their existing domain stores.
    """

    stage: str = "ROUTING"
    status: str = "ROUTING"
    current_agent: str | None = None
    selected_route: Route | None = None
    research_status: str | None = None
    writer_ready: bool = False
    last_agent: Literal["research", "writer", "reviewer"] | None = None
    last_specialist_status: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    pending_gaps: list[str] = Field(default_factory=list)
    success_criteria_status: dict[str, bool] = Field(default_factory=dict)
    evidence_references: list[dict[str, Any]] = Field(default_factory=list)
    research_result_reference: dict[str, Any] | None = None
    domain_result_refs: dict[str, DomainResultReference] = Field(default_factory=dict)
    draft_patch_reference: dict[str, Any] | None = None
    draft_patch_id: str | None = None
    research_round: int = Field(default=0, ge=0)
    review_round: int = Field(default=0, ge=0)
    manager_steps: int = Field(default=0, ge=0)
    review_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    last_specialist_result: dict[str, Any] | None = None
    latest_specialist_result: dict[str, Any] | None = None
    review_decision: Literal["PASS", "REVISE", "REJECT"] | None = None
    manager_decisions: list[dict[str, Any]] = Field(default_factory=list)
    recovery_action: ManagerAction | None = None
    pending_approval: dict[str, Any] | None = None
    last_progress_fingerprint: str | None = None
    no_progress_rounds: int = Field(default=0, ge=0)
    user_preferences: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)


def default_run_goal(
    instruction: str,
    task_type: str | None = None,
    *,
    success_criteria: tuple[str, ...] | list[str] | None = None,
    hard_constraints: tuple[str, ...] | list[str] | None = None,
) -> RunGoal:
    """Create a deterministic goal without asking an LLM to define success."""

    route = (task_type or "RESEARCH").strip().upper() or "RESEARCH"
    defaults: dict[str, tuple[str, ...]] = {
        "RESEARCH": (
            "produce a verified research answer",
            "retain traceable evidence references",
            "resolve or explicitly report blocking claims",
        ),
        "SUPPORT_CLAIM": (
            "produce evidence that supports the requested claim",
            "retain traceable evidence references",
            "do not present unresolved claims as verified",
        ),
        "WRITE_INTRODUCTION": (
            "produce a complete Introduction DraftPatch",
            "ground the draft in verified evidence",
            "pass Reviewer validation before human approval",
        ),
        "WRITE_CONCLUSION": (
            "produce a complete Conclusion DraftPatch",
            "preserve confirmed manuscript facts and contributions",
            "pass Reviewer validation before human approval",
        ),
        "WRITE_ABSTRACT": (
            "produce a complete Abstract DraftPatch",
            "preserve confirmed manuscript facts and contributions",
            "pass Reviewer validation before human approval",
        ),
        "REVIEW": (
            "produce a deterministic review of the requested manuscript state",
            "report all blocking issues and unsupported claims",
        ),
    }
    constraints = tuple(hard_constraints or ()) or (
        "Manager may plan only and may not apply a DraftPatch",
        "a local Reviewer gap or user preference cannot replace the OriginalGoal",
        "fail closed on budget exhaustion, unsafe transitions, or premature completion",
    )
    return RunGoal(
        original_instruction=instruction.strip(),
        task_type=route,
        success_criteria=tuple(success_criteria or defaults.get(route, defaults["RESEARCH"])),
        hard_constraints=constraints,
    )


class ManagerDecision(ContractModel):
    """Strict cognition contract returned on every Manager planning turn."""

    next_action: ManagerAction
    target_agent: ManagerTarget | None
    goal_alignment: str = Field(min_length=1, max_length=2000)
    remaining_gap: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)
    required_context_refs: list[str] = Field(default_factory=list)
    completion_status: ManagerCompletionStatus = "IN_PROGRESS"


class OrchestrationRequest(ContractModel):
    request_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    session_id: str | None = None
    task_type: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Identity is resolved by the API/authentication boundary and then copied
    # into the durable Run.  The local default is only for CLI/tests.
    tenant_id: str = LOCAL_PRINCIPAL.tenant_id
    principal_id: str = LOCAL_PRINCIPAL.principal_id

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

    @field_validator("tenant_id", "principal_id", mode="before")
    @classmethod
    def _normalize_identity(cls, value: Any) -> Any:
        if value is None:
            return "local"
        try:
            # Reuse the same identifier policy as the persistence layer.
            TenantPrincipal(tenant_id=str(value), principal_id="local")
        except (TypeError, ValueError) as error:
            raise ValueError("tenant_id/principal_id 必须是安全标识符。") from error
        return str(value).strip()


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
    backend: str = "crewai"

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
