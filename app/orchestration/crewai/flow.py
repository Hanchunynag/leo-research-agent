"""Deterministic CrewAI Flow for the Scholar request lifecycle."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from crewai.flow import Flow, start
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.generation.security import redact_sensitive_text
from app.jobs.worker import JobCancelled
from app.orchestration.contracts import (
    OrchestrationRequest,
    OrchestrationResult,
    ResearchAgentOutput,
    ReviewAgentOutput,
    Route,
    SupervisorResult,
    WriterAgentOutput,
)
from app.orchestration.crewai.tracing import CrewAITraceAdapter


FlowLifecycle = Literal[
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


class FlowState(BaseModel):
    """Only bounded run references and current contracts live in Flow state."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_id: str
    session_id: str | None = None
    run_id: str
    thread_id: str
    trace_id: str
    request: OrchestrationRequest
    status: FlowLifecycle = "ROUTING"
    current_agent: str | None = None
    selected_route: Route | None = None
    # Flow state is intentionally a run-local reference projection. The
    # complete domain objects stay in the existing Research/Writing/Review
    # runtimes and are handed between methods through bounded contracts, not
    # persisted as a second long-term state store by CrewAI.
    research_result_reference: dict[str, Any] | None = None
    draft_patch_id: str | None = None
    review_decision: Literal["PASS", "REVISE", "REJECT"] | None = None
    review_round: int = 0
    pending_approval: dict[str, Any] | None = None
    error_codes: list[str] = Field(default_factory=list)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            str(key): _jsonable(child)
            for key, child in value.model_dump(mode="python").items()
        }
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class CrewAIOrchestrationFlow(Flow[FlowState]):
    """CrewAI Flow whose single start method owns all business transitions.

    The method intentionally contains a small explicit state machine instead
    of allowing CrewAI task ordering or Agent delegation to mutate lifecycle
    state.  CrewAI Memory is disabled; the caller's SessionRuntime remains the
    only durable conversation/run store.
    """

    _skip_auto_memory = True
    _backend: Any = PrivateAttr()
    _trace: CrewAITraceAdapter = PrivateAttr()
    _max_review_rounds: int = PrivateAttr(default=2)
    _research_output: ResearchAgentOutput | None = PrivateAttr(default=None)
    _writer_output: WriterAgentOutput | None = PrivateAttr(default=None)
    _review_output: ReviewAgentOutput | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        backend: Any,
        state: FlowState,
        trace: CrewAITraceAdapter,
        max_review_rounds: int = 2,
    ) -> None:
        # ``Flow(memory=False)`` is rejected by CrewAI 1.15's Pydantic
        # validator; ``None`` plus ``_skip_auto_memory`` is its explicit
        # no-memory configuration.
        super().__init__(
            initial_state=state, memory=None, tracing=False, max_method_calls=100
        )
        self._backend = backend
        self._trace = trace
        self._max_review_rounds = max(0, min(2, max_review_rounds))

    def _transition(self, status: FlowLifecycle, *, agent: str | None = None) -> None:
        self.state.status = status
        self.state.current_agent = agent
        self._trace.transition(status, route=self.state.selected_route)

    def _result(
        self,
        status: Any,
        *,
        final_answer: str | None = None,
        pending_action: dict[str, Any] | None = None,
        value: Any | None = None,
        error_codes: tuple[str, ...] = (),
    ) -> OrchestrationResult:
        result = OrchestrationResult(
            status=status,
            request_id=self.state.request.request_id,
            project_id=self.state.project_id,
            session_id=self.state.session_id,
            run_id=self.state.run_id,
            thread_id=self.state.thread_id,
            trace_id=self.state.trace_id,
            selected_route=self.state.selected_route,
            final_answer=final_answer,
            pending_action=pending_action,
            specialist_results={
                key: _jsonable(value)
                for key, value in {
                    "research": self._research_output,
                    "writer": self._writer_output,
                    "reviewer": self._review_output,
                }.items()
                if value is not None
            },
            approval_required=pending_action is not None,
            error_codes=list(dict.fromkeys((*self.state.error_codes, *error_codes))),
            diagnostics={
                "flow_state": self.state.status,
                "current_agent": self.state.current_agent,
                "review_round": self.state.review_round,
                "research_result_reference": self.state.research_result_reference,
                "draft_patch_id": self.state.draft_patch_id,
                "review_decision": self.state.review_decision,
                "pending_approval": self.state.pending_approval,
                "trace": self._trace.diagnostics(),
            },
            value=value,
            backend="crewai",
        )
        self._backend.persist_result(result)
        return result

    def _fail(self, code: str, *, value: Any | None = None) -> OrchestrationResult:
        self.state.error_codes.append(code)
        self._transition("FAILED")
        return self._result("FAILED", value=value, error_codes=(code,))

    def _research(self) -> ResearchAgentOutput | None:
        self._transition("RESEARCHING", agent="Research Agent")
        try:
            output = self._backend.research(
                self.state.request, self.state.selected_route, self._trace
            )
        except JobCancelled:
            raise
        except Exception as error:
            self._trace.record(
                "research_failed",
                "FAILED",
                {
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", "RESEARCH_FAILED"),
                    "error": redact_sensitive_text(str(error))[:1000],
                },
                kind="agent",
            )
            self.state.error_codes.append(
                str(getattr(error, "code", "RESEARCH_FAILED"))
            )
            return None
        if not isinstance(output, ResearchAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._research_output = output
        self.state.research_result_reference = {
            "status": output.status,
            "evidence_count": len(output.evidence),
            "citation_count": len(output.citations),
            "unresolved_count": len(output.unresolved_claims),
            "freshness_status": output.freshness_status,
        }
        if output.status == "FAILED":
            self.state.error_codes.append(
                str(output.diagnostics.get("error_code") or "RESEARCH_FAILED")
            )
            return None
        return output

    def _write(self) -> WriterAgentOutput | None:
        self._transition("WRITING", agent="Writer Agent")
        try:
            output = self._backend.write(
                self.state.request,
                self.state.selected_route,
                self._research_output,
                self._trace,
                review_round=self.state.review_round,
            )
        except Exception as error:
            self._trace.record(
                "writer_failed",
                "FAILED",
                {
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", "WRITING_FAILED"),
                    "error": redact_sensitive_text(str(error))[:1000],
                },
                kind="agent",
            )
            self.state.error_codes.append(
                str(getattr(error, "code", "WRITING_FAILED"))
            )
            return None
        if not isinstance(output, WriterAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._writer_output = output
        self.state.draft_patch_id = (
            str(getattr(output.draft_patch, "patch_id", ""))
            if output.draft_patch is not None
            else None
        )
        if (
            output.status
            in {"FAILED", "INSUFFICIENT_MANUSCRIPT_STATE", "RESEARCH_REQUIRED"}
            or output.research_required
        ):
            code = (
                output.missing_context[0]
                if output.missing_context
                else "INSUFFICIENT_MANUSCRIPT_STATE"
            )
            self.state.error_codes.append(code)
            return None
        if output.draft_patch is None:
            self.state.error_codes.append("DRAFT_PATCH_REQUIRED")
            return None
        return output

    def _review(self) -> ReviewAgentOutput | None:
        self._transition("REVIEWING", agent="Reviewer Agent")
        try:
            output = self._backend.review(
                self.state.request,
                self.state.selected_route,
                self._writer_output,
                self._trace,
                review_round=self.state.review_round,
            )
        except Exception as error:
            self._trace.record(
                "reviewer_failed",
                "FAILED",
                {
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", "REVIEW_FAILED"),
                    "error": redact_sensitive_text(str(error))[:1000],
                },
                kind="agent",
            )
            self.state.error_codes.append(
                str(getattr(error, "code", "REVIEW_FAILED"))
            )
            return None
        if not isinstance(output, ReviewAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._review_output = output
        self.state.review_decision = output.decision
        return output

    @start()
    def execute(self) -> OrchestrationResult:
        with self._trace.activate():
            try:
                self._transition("ROUTING", agent="Scholar Supervisor Agent")
                supervisor = self._backend.route(self.state.request, self._trace)
                if (
                    not isinstance(supervisor, SupervisorResult)
                    or supervisor.selected_route is None
                ):
                    return self._fail("ROUTING_CONTRACT_INVALID")
                self.state.selected_route = supervisor.selected_route
                route = supervisor.selected_route

                if route in {"RESEARCH", "SUPPORT_CLAIM"}:
                    research = self._research()
                    if research is None:
                        return self._fail(
                            self.state.error_codes[-1]
                            if self.state.error_codes
                            else "RESEARCH_FAILED"
                        )
                    final = self._backend.finalize_research(
                        self.state.request, research, self._trace
                    )
                    answer = (
                        final.final_answer or research.research_summary
                        if isinstance(final, SupervisorResult)
                        else research.research_summary
                    )
                    self._transition("COMPLETED")
                    return self._result(
                        "COMPLETED", final_answer=answer, value=research.domain_result
                    )

                if route == "REVIEW":
                    # Existing review is only exposed when a patch reference is
                    # present.  It still cannot apply or approve the patch.
                    review = self._backend.review_existing(
                        self.state.request, self._trace
                    )
                    if review is None:
                        return self._fail("REVIEW_REQUEST_INVALID")
                    self._review_output = review
                    self.state.review_decision = review.decision
                    self._transition("COMPLETED")
                    return self._result("COMPLETED", value=review.review_report)

                if route not in {
                    "WRITE_INTRODUCTION",
                    "WRITE_CONCLUSION",
                    "WRITE_ABSTRACT",
                }:
                    return self._fail("UNSUPPORTED_ROUTE")
                if route == "WRITE_INTRODUCTION":
                    research = self._research()
                    if research is None:
                        return self._fail(
                            self.state.error_codes[-1]
                            if self.state.error_codes
                            else "RESEARCH_FAILED"
                        )
                    # Evidence gaps are not a valid handoff to Writer.  The
                    # Research route may still return a partial answer, but
                    # Introduction writing must fail closed rather than turn
                    # an unsupported gap into a DraftPatch.
                    if research.status != "COMPLETED":
                        return self._fail(
                            str(
                                research.diagnostics.get("error_code")
                                or research.status
                            )
                        )
                # Conclusion/Abstract deliberately skip _research().  Their
                # CapabilityProfile has no research permission and all missing
                # upstream state therefore fails closed in the Writing Runtime.
                while self.state.review_round <= self._max_review_rounds:
                    writer = self._write()
                    if writer is None:
                        # Preserve the failed WritingResult as a diagnostic
                        # value.  In particular, Introduction source-coverage
                        # failures carry the actual N/8 metadata and the
                        # verified evidence packs needed by the Console. The
                        # value is read-only; no DraftPatch is created here.
                        failed_writer_value = (
                            self._writer_output.domain_result
                            if self._writer_output is not None
                            else None
                        )
                        return self._fail(
                            self.state.error_codes[-1]
                            if self.state.error_codes
                            else "WRITING_FAILED",
                            value=failed_writer_value,
                        )
                    reviewer = self._review()
                    if reviewer is None:
                        return self._fail(
                            self.state.error_codes[-1]
                            if self.state.error_codes
                            else "REVIEW_FAILED"
                        )
                    if reviewer.decision == "PASS":
                        patch = writer.draft_patch
                        patch_id = str(getattr(patch, "patch_id", ""))
                        base_hash = str(getattr(patch, "base_hash", ""))
                        if not patch_id or not base_hash:
                            return self._fail("DRAFT_PATCH_INVALID")
                        pending = {
                            "type": "HUMAN_APPROVAL",
                            "patch_id": patch_id,
                            "expected_base_hash": base_hash,
                            "safe_apply": "PatchApprovalService only",
                        }
                        self.state.pending_approval = pending
                        self._transition("WAITING_HUMAN_APPROVAL")
                        return self._result(
                            "WAITING_HUMAN_APPROVAL",
                            pending_action=pending,
                            value=writer.domain_result,
                        )
                    if reviewer.decision == "REJECT":
                        return self._fail("REVIEW_REJECTED", value=writer.domain_result)
                    if self.state.review_round >= self._max_review_rounds:
                        return self._fail(
                            "REVIEW_BUDGET_EXCEEDED", value=writer.domain_result
                        )
                    self.state.review_round += 1
                    self._transition("REVISION_REQUIRED", agent="Writer Agent")
                return self._fail("REVIEW_BUDGET_EXCEEDED")
            except Exception as error:
                if isinstance(error, JobCancelled):
                    # Cancellation is a Worker lifecycle decision. Do not
                    # turn a cooperative stop into ORCHESTRATION_FAILED or
                    # persist a misleading failed Run before the Worker can
                    # finalize the paired Job as CANCELLED.
                    raise
                self._trace.record(
                    "flow_failed",
                    "FAILED",
                    {
                        "error_type": type(error).__name__,
                        "error": redact_sensitive_text(str(error))[:1000],
                    },
                    kind="flow",
                )
                return self._fail("ORCHESTRATION_FAILED")
