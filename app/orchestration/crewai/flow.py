"""Deterministic CrewAI Flow with a persistent Manager decision loop."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from crewai.flow import Flow, start
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.generation.security import redact_sensitive_text
from app.jobs.worker import JobCancelled
from app.orchestration.contracts import (
    DomainResultReference,
    ManagerAction,
    ManagerDecision,
    OrchestrationRequest,
    OrchestrationResult,
    ResearchAgentOutput,
    ReviewAgentOutput,
    RunBudget,
    RunGoal,
    RunState,
    WriterAgentOutput,
)
from app.orchestration.crewai.tracing import CrewAITraceAdapter
from app.orchestration.manager import ManagerDecisionRejected, ManagerDecisionValidator


FlowLifecycle = Literal[
    "ROUTING", "RESEARCHING", "WRITING", "REVIEWING", "REVISION_REQUIRED",
    "WAITING_HUMAN_APPROVAL", "APPLYING", "COMPLETED", "FAILED", "INTERRUPTED",
]


class FlowState(RunState):
    """The CrewAI Flow's durable state is the RunState contract itself."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    project_id: str
    session_id: str | None = None
    run_id: str
    thread_id: str
    trace_id: str
    request: OrchestrationRequest
    goal: RunGoal
    budget: RunBudget = Field(default_factory=RunBudget)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {str(key): _jsonable(child) for key, child in value.model_dump(mode="python").items()}
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
    """Manager cognition inside the deterministic Scholar state machine."""

    _skip_auto_memory = True
    _backend: Any = PrivateAttr()
    _trace: CrewAITraceAdapter = PrivateAttr()
    _validator: ManagerDecisionValidator = PrivateAttr()
    _research_output: ResearchAgentOutput | None = PrivateAttr(default=None)
    _writer_output: WriterAgentOutput | None = PrivateAttr(default=None)
    _review_output: ReviewAgentOutput | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        backend: Any,
        state: FlowState,
        trace: CrewAITraceAdapter,
        max_research_attempts: int = 2,
        max_review_rounds: int = 2,
        max_manager_steps: int = 12,
    ) -> None:
        super().__init__(initial_state=state, memory=None, tracing=False, max_method_calls=100)
        self._backend = backend
        self._trace = trace
        self._validator = ManagerDecisionValidator(
            max_research_attempts=max_research_attempts,
            max_review_calls=max(1, max_review_rounds + 1),
            max_manager_steps=max_manager_steps,
        )
        if self.state.selected_route == "WRITE_INTRODUCTION" and not self.state.pending_gaps:
            self.state.pending_gaps = ["verified evidence for the complete Introduction"]
        self._update_success_criteria()

    def _transition(self, status: FlowLifecycle, *, agent: str | None = None) -> None:
        self.state.status = status
        self.state.stage = status
        self.state.current_agent = agent
        self._trace.transition(status, route=self.state.selected_route)

    def _update_success_criteria(self) -> None:
        """Refresh criterion status from deterministic state, never LLM prose."""
        route = self.state.selected_route or self.state.goal.task_type
        research_ok = self.state.research_status == "COMPLETED" and not self.state.pending_gaps
        draft_ok = bool(self.state.draft_patch_reference)
        review_ok = self.state.review_decision == "PASS"
        if route in {"RESEARCH", "SUPPORT_CLAIM"}:
            values = [research_ok]
        elif route == "REVIEW":
            values = [review_ok]
        else:
            values = [draft_ok, research_ok if route == "WRITE_INTRODUCTION" else True, review_ok]
        self.state.success_criteria_status = {
            criterion: values[index] if index < len(values) else values[-1]
            for index, criterion in enumerate(self.state.goal.success_criteria)
        }

    def _result(
        self, status: Any, *, final_answer: str | None = None,
        pending_action: dict[str, Any] | None = None, value: Any | None = None,
        error_codes: tuple[str, ...] = (),
    ) -> OrchestrationResult:
        result = OrchestrationResult(
            status=status, request_id=self.state.request.request_id,
            project_id=self.state.project_id, session_id=self.state.session_id,
            run_id=self.state.run_id, thread_id=self.state.thread_id,
            trace_id=self.state.trace_id, selected_route=self.state.selected_route,
            final_answer=final_answer, pending_action=pending_action,
            specialist_results={
                key: _jsonable(value) for key, value in {
                    "research": self._research_output,
                    "writer": self._writer_output,
                    "reviewer": self._review_output,
                }.items() if value is not None
            },
            approval_required=pending_action is not None,
            error_codes=list(dict.fromkeys((*self.state.error_codes, *error_codes))),
            diagnostics={
                "flow_state": self.state.status,
                "current_agent": self.state.current_agent,
                "manager_steps": self.state.manager_steps,
                "research_attempts": self.state.research_round,
                "review_calls": self.state.review_calls,
                "review_round": self.state.review_round,
                "manager_decisions": [_jsonable(item) for item in self.state.manager_decisions],
                "domain_result_refs": _jsonable(self.state.domain_result_refs),
                "research_result_reference": self.state.research_result_reference,
                "draft_patch_id": self.state.draft_patch_id,
                "review_decision": self.state.review_decision,
                "pending_approval": self.state.pending_approval,
                "original_goal": _jsonable(self.state.goal),
                "run_state": _jsonable(self.state),
                "trace": self._trace.diagnostics(),
            },
            value=value, backend="crewai",
        )
        self._backend.persist_result(result)
        return result

    def _fail(self, code: str, *, value: Any | None = None) -> OrchestrationResult:
        self.state.error_codes.append(code)
        self._transition("FAILED")
        return self._result("FAILED", value=value, error_codes=(code,))

    def _checkpoint_before_action(self, action: ManagerAction) -> None:
        if action in {"CALL_RESEARCH", "REQUEST_MORE_EVIDENCE"}:
            recovery_action: ManagerAction = "CALL_RESEARCH"
        elif action in {"CALL_WRITER", "REQUEST_REVISION"}:
            # REQUEST_REVISION is a planning label; the replayable capability
            # is still the Writer. Replaying Research here could consume a
            # fresh evidence round after an interruption and turn a resumable
            # revision into an avoidable budget failure.
            recovery_action = "CALL_WRITER"
        else:
            recovery_action = "CALL_WRITER"
        self.state.recovery_action = recovery_action
        self._backend.persist_checkpoint(self.state)
        self._trace.record(
            "manager_checkpoint_saved", "COMPLETED",
            {"recovery_action": recovery_action, "manager_steps": self.state.manager_steps},
            kind="checkpoint",
        )

    def _research(self) -> ResearchAgentOutput | None:
        self._transition("RESEARCHING", agent="Research Agent")
        try:
            output = self._backend.research(self.state.request, self.state.selected_route, self._trace)
        except JobCancelled:
            raise
        except Exception as error:
            self._trace.record(
                "research_failed", "FAILED",
                {"error_type": type(error).__name__, "error_code": getattr(error, "code", "RESEARCH_FAILED"), "error": redact_sensitive_text(str(error))[:1000]},
                kind="agent",
            )
            self.state.error_codes.append(str(getattr(error, "code", "RESEARCH_FAILED")))
            return None
        if not isinstance(output, ResearchAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._research_output = output
        self.state.research_round += 1
        self.state.research_status = output.status
        self.state.writer_ready = output.status == "COMPLETED"
        self.state.last_agent = "research"
        self.state.last_specialist_status = output.status
        self.state.current_agent = "Research Agent"
        self.state.completed_steps.append(
            f"Research round {self.state.research_round}: {output.status}"
        )
        self.state.latest_specialist_result = {
            "agent": "research",
            "status": output.status,
            "summary": output.research_summary[:700],
            "evidence_count": len(output.evidence),
            "unresolved_claims": output.unresolved_claims[:8],
        }
        self.state.last_specialist_result = self.state.latest_specialist_result
        evidence_ids = [
            str(item.get("evidence_id") or item.get("chunk_id") or item.get("paper_id"))
            for item in output.evidence[:32]
            if isinstance(item, dict)
        ]
        fingerprint = "|".join(sorted(value for value in evidence_ids if value)) or output.research_summary[:200]
        if fingerprint and fingerprint == self.state.last_progress_fingerprint:
            self.state.no_progress_rounds += 1
        else:
            self.state.no_progress_rounds = 0
            self.state.last_progress_fingerprint = fingerprint or None
        self.state.pending_gaps = list(dict.fromkeys(output.unresolved_claims + output.contradictions))
        self.state.evidence_references = [
            {
                "evidence_id": item.get("evidence_id") or item.get("chunk_id"),
                "paper_id": item.get("paper_id"),
                "content_hash": item.get("content_hash"),
            }
            for item in output.evidence[:32]
            if isinstance(item, dict)
        ]
        self.state.research_result_reference = {
            "status": output.status, "evidence_count": len(output.evidence),
            "citation_count": len(output.citations), "unresolved_count": len(output.unresolved_claims),
            "freshness_status": output.freshness_status,
        }
        self.state.domain_result_refs["research"] = DomainResultReference(
            result_id=f"{self.state.run_id}:research:{self.state.research_round}",
            result_type=type(output.domain_result).__name__ if output.domain_result is not None else "ResearchResult",
            producer="research", status=output.status,
            summary=output.research_summary[:500],
            metadata={"evidence_count": len(output.evidence), "unresolved_count": len(output.unresolved_claims)},
        )
        if output.status == "FAILED":
            self.state.error_codes.append(str(output.diagnostics.get("error_code") or "RESEARCH_FAILED"))
            return None
        self._update_success_criteria()
        return output

    def _write(self) -> WriterAgentOutput | None:
        self._transition("WRITING", agent="Writer Agent")
        try:
            output = self._backend.write(
                self.state.request, self.state.selected_route, self._research_output,
                self._trace, review_round=self.state.review_round,
            )
        except Exception as error:
            self._trace.record(
                "writer_failed", "FAILED",
                {"error_type": type(error).__name__, "error_code": getattr(error, "code", "WRITING_FAILED"), "error": redact_sensitive_text(str(error))[:1000]},
                kind="agent",
            )
            self.state.error_codes.append(str(getattr(error, "code", "WRITING_FAILED")))
            return None
        if not isinstance(output, WriterAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._writer_output = output
        self.state.last_agent = "writer"
        self.state.last_specialist_status = output.status
        patch = output.draft_patch
        self.state.draft_patch_id = str(getattr(patch, "patch_id", "")) if patch is not None else None
        self.state.writer_ready = output.status == "READY" and patch is not None
        self.state.current_agent = "Writer Agent"
        self.state.completed_steps.append(
            f"Writer round {self.state.review_round + 1}: {output.status}"
        )
        self.state.latest_specialist_result = {
            "agent": "writer",
            "status": output.status,
            "draft_patch_id": self.state.draft_patch_id,
            "research_required": output.research_required,
            "missing_context": output.missing_context[:8],
        }
        self.state.last_specialist_result = self.state.latest_specialist_result
        if output.status in {"FAILED", "INSUFFICIENT_MANUSCRIPT_STATE"} or output.research_required:
            self.state.error_codes.extend(output.missing_context)
            self.state.pending_gaps = list(dict.fromkeys(output.missing_context))
            return None
        if patch is None:
            self.state.error_codes.append("DRAFT_PATCH_REQUIRED")
            return None
        self.state.domain_result_refs["writer"] = DomainResultReference(
            result_id=str(getattr(patch, "patch_id", f"{self.state.run_id}:writer")),
            result_type=type(output.domain_result).__name__ if output.domain_result is not None else "WritingResult",
            producer="writer", status=output.status, summary="DraftPatch proposal created",
            metadata={"patch_id": self.state.draft_patch_id},
        )
        self.state.draft_patch_reference = {
            "patch_id": self.state.draft_patch_id,
            "status": output.status,
            "producer": "writer",
        }
        self.state.pending_gaps = list(dict.fromkeys(output.missing_context))
        self._update_success_criteria()
        return output

    def _review(self) -> ReviewAgentOutput | None:
        self._transition("REVIEWING", agent="Reviewer Agent")
        try:
            output = self._backend.review(
                self.state.request, self.state.selected_route, self._writer_output,
                self._trace, review_round=self.state.review_round,
            )
        except Exception as error:
            self._trace.record(
                "reviewer_failed", "FAILED",
                {"error_type": type(error).__name__, "error_code": getattr(error, "code", "REVIEW_FAILED"), "error": redact_sensitive_text(str(error))[:1000]},
                kind="agent",
            )
            self.state.error_codes.append(str(getattr(error, "code", "REVIEW_FAILED")))
            return None
        if not isinstance(output, ReviewAgentOutput):
            self.state.error_codes.append("CONTRACT_VALIDATION_FAILED")
            return None
        self._review_output = output
        self.state.review_calls += 1
        self.state.last_agent = "reviewer"
        self.state.last_specialist_status = output.decision
        self.state.review_decision = output.decision
        self.state.current_agent = "Reviewer Agent"
        self.state.completed_steps.append(
            f"Reviewer round {self.state.review_calls}: {output.decision}"
        )
        self.state.latest_specialist_result = {
            "agent": "reviewer",
            "status": output.decision,
            "issue_count": len(output.issues),
            "revision_instructions": output.revision_instructions[:8],
        }
        self.state.last_specialist_result = self.state.latest_specialist_result
        if output.decision == "PASS":
            self.state.pending_gaps = []
        else:
            issues = [
                str(item.get("message") or item.get("description") or item)
                for item in output.issues[:8]
            ]
            self.state.pending_gaps = list(
                dict.fromkeys(
                    issues
                    + output.unsupported_claims
                    + output.citation_issues
                    + output.revision_instructions
                )
            )
        self.state.domain_result_refs["reviewer"] = DomainResultReference(
            result_id=f"{self.state.run_id}:review:{self.state.review_calls}",
            result_type=type(output.review_report).__name__ if output.review_report is not None else "ReviewReport",
            producer="reviewer", status=output.decision,
            summary=f"Reviewer decision={output.decision}", metadata={"issue_count": len(output.issues)},
        )
        self._update_success_criteria()
        return output

    def _manager_context(self, phase: str) -> dict[str, Any]:
        context = (
            self._backend.build_manager_context(self.state)
            if hasattr(self._backend, "build_manager_context")
            else {
                "original_goal": _jsonable(self.state.goal),
                "current_run_state": _jsonable(self.state),
            }
        )
        context.update(
            {
                "phase": phase,
                "route": self.state.selected_route,
                "request": {
                    "instruction": self.state.request.instruction,
                    "task_type": self.state.request.task_type,
                },
                "manager_steps": self.state.manager_steps,
                "research_attempts": self.state.research_round,
                "review_calls": self.state.review_calls,
                "research_status": self.state.research_status,
                "writer_ready": self.state.writer_ready,
                "review_decision": self.state.review_decision,
                "last_agent": self.state.last_agent,
                "last_specialist_status": self.state.last_specialist_status,
                "recovery_action": self.state.recovery_action,
                "pending_gaps": self.state.pending_gaps,
                "success_criteria_status": self.state.success_criteria_status,
                "allowed_actions": [
                    "CALL_RESEARCH",
                    "CALL_WRITER",
                    "CALL_REVIEWER",
                    "REQUEST_MORE_EVIDENCE",
                    "REQUEST_REVISION",
                    "COMPLETE",
                ],
            }
        )
        return context

    def _manager(self) -> ManagerDecision | None:
        phase = "RECOVERY" if self.state.recovery_action is not None else "INITIAL" if self.state.last_agent is None else f"AFTER_{self.state.last_agent.upper()}"
        self._transition("ROUTING", agent="Scholar Manager Agent")
        try:
            backend_tool_calls = getattr(self._backend, "_tool_calls_used", None)
            if isinstance(backend_tool_calls, int):
                self.state.tool_calls = backend_tool_calls
            total_tokens = self._trace.diagnostics().get("total_tokens")
            if (
                self.state.budget.max_tokens is not None
                and isinstance(total_tokens, (int, float))
                and total_tokens >= self.state.budget.max_tokens
            ):
                raise ManagerDecisionRejected("TOKEN_BUDGET_EXCEEDED: token budget exceeded")
            decision = self._backend.manager_decide(self._manager_context(phase), self._trace)
            if not isinstance(decision, ManagerDecision):
                raise ManagerDecisionRejected("Manager output is not ManagerDecision")
            validation = self._validator.validate(
                decision,
                goal=self.state.goal,
                state=self.state,
                budget=self.state.budget,
                research_status=self.state.research_status,
                writer_ready=self.state.writer_ready,
                review_decision=self.state.review_decision,
                recovery_action=self.state.recovery_action,
                context_refs=self.state.domain_result_refs,
            )
            if not validation.allowed:
                error = ManagerDecisionRejected(f"{validation.code}: {validation.reason}")
                error.code = validation.code or ManagerDecisionRejected.code
                raise error
        except (ManagerDecisionRejected, ValueError) as error:
            code = getattr(error, "code", "MANAGER_DECISION_REJECTED")
            self._trace.record(
                "manager_decision_rejected", "FAILED",
                {"error_code": code, "error": redact_sensitive_text(str(error))[:500]}, kind="manager",
            )
            self.state.error_codes.append(code)
            return None
        self.state.manager_steps += 1
        self.state.manager_decisions.append(decision.model_dump(mode="python"))
        self.state.recovery_action = None
        self._trace.record(
            "manager_decision", "COMPLETED",
            {"next_action": decision.next_action, "target_agent": decision.target_agent, "completion_status": decision.completion_status, "reason": decision.reason[:500]}, kind="manager",
        )
        return decision

    def _complete(self, decision: ManagerDecision) -> OrchestrationResult:
        if decision.completion_status in {"WAITING_HUMAN_APPROVAL", "READY_TO_COMPLETE"} and self.state.selected_route not in {"RESEARCH", "SUPPORT_CLAIM", "REVIEW"}:
            patch = self._writer_output.draft_patch if self._writer_output else None
            patch_id, base_hash = str(getattr(patch, "patch_id", "")), str(getattr(patch, "base_hash", ""))
            if not patch_id or not base_hash:
                return self._fail("DRAFT_PATCH_INVALID")
            pending = {"type": "HUMAN_APPROVAL", "patch_id": patch_id, "expected_base_hash": base_hash, "safe_apply": "PatchApprovalService only"}
            self.state.pending_approval = pending
            self._transition("WAITING_HUMAN_APPROVAL")
            return self._result("WAITING_HUMAN_APPROVAL", pending_action=pending, value=self._writer_output.domain_result if self._writer_output else None)
        self._transition("COMPLETED")
        if self.state.selected_route in {"RESEARCH", "SUPPORT_CLAIM"}:
            answer = self._research_output.research_summary if self._research_output else ""
            value = self._research_output.domain_result if self._research_output else None
        elif self.state.selected_route == "REVIEW":
            answer, value = f"Review completed: {self.state.review_decision}", self._review_output.review_report if self._review_output else None
        else:
            answer, value = "DraftPatch proposal is ready for human approval.", self._writer_output.domain_result if self._writer_output else None
        return self._result("COMPLETED", final_answer=answer, value=value)

    @start()
    def execute(self) -> OrchestrationResult:
        with self._trace.activate():
            try:
                if self.state.selected_route is None:
                    return self._fail("ROUTE_REQUIRED")
                while self.state.manager_steps < self._validator.max_manager_steps:
                    decision = self._manager()
                    if decision is None:
                        return self._fail(self.state.error_codes[-1] if self.state.error_codes else "MANAGER_DECISION_REJECTED")
                    if decision.next_action == "COMPLETE":
                        return self._complete(decision)
                    self._checkpoint_before_action(decision.next_action)
                    if decision.next_action in {"CALL_RESEARCH", "REQUEST_MORE_EVIDENCE"}:
                        research_output = self._research()
                        if research_output is None:
                            return self._fail(self.state.error_codes[-1] if self.state.error_codes else "RESEARCH_FAILED")
                        if (
                            research_output.status == "INSUFFICIENT_EVIDENCE"
                            and self.state.research_round >= self._validator.max_research_attempts
                        ):
                            return self._fail("INSUFFICIENT_EVIDENCE")
                    elif decision.next_action in {"CALL_WRITER", "REQUEST_REVISION"}:
                        is_revision = (
                            decision.next_action == "REQUEST_REVISION"
                            or self.state.review_decision == "REVISE"
                        )
                        if is_revision:
                            self.state.review_round += 1
                        if self._write() is None:
                            failed_value = self._writer_output.domain_result if self._writer_output else None
                            return self._fail(self.state.error_codes[-1] if self.state.error_codes else "WRITING_FAILED", value=failed_value)
                        if decision.next_action == "REQUEST_REVISION":
                            self._transition("REVISION_REQUIRED", agent="Writer Agent")
                    elif decision.next_action == "CALL_REVIEWER":
                        if self.state.selected_route == "REVIEW":
                            review = self._backend.review_existing(self.state.request, self._trace)
                            if review is None:
                                return self._fail("REVIEW_REQUEST_INVALID")
                            self._review_output = review
                            self.state.review_calls += 1
                            self.state.last_agent, self.state.last_specialist_status = "reviewer", review.decision
                            self.state.review_decision = review.decision
                            self.state.latest_specialist_result = {
                                "agent": "reviewer",
                                "status": review.decision,
                                "issue_count": len(review.issues),
                            }
                            self.state.last_specialist_result = self.state.latest_specialist_result
                            self.state.pending_gaps = [] if review.decision == "PASS" else list(review.revision_instructions)
                            self._update_success_criteria()
                        elif self._review() is None:
                            return self._fail(self.state.error_codes[-1] if self.state.error_codes else "REVIEW_FAILED")
                    else:
                        return self._fail("UNSUPPORTED_MANAGER_ACTION")
                    # The in-memory action completed. The persisted checkpoint
                    # intentionally still contains its replay action so a
                    # process restart can safely repeat the capability.
                    self.state.recovery_action = None
                return self._fail("MANAGER_STEP_BUDGET_EXCEEDED")
            except JobCancelled:
                raise
            except Exception as error:
                self._trace.record(
                    "flow_failed", "FAILED",
                    {"error_type": type(error).__name__, "error": redact_sensitive_text(str(error))[:1000]}, kind="flow",
                )
                return self._fail("ORCHESTRATION_FAILED")
