from __future__ import annotations

from pathlib import Path
from typing import Any

from app.orchestration.contracts import (
    ManagerDecision,
    OrchestrationRequest,
    ResearchAgentOutput,
    ReviewAgentOutput,
    RunBudget,
    RunGoal,
    RunState,
    WriterAgentOutput,
    default_run_goal,
)
from app.orchestration.crewai.flow import CrewAIOrchestrationFlow, FlowState
from app.orchestration.crewai.tracing import CrewAITraceAdapter
from app.orchestration.manager import ManagerDecisionValidator
from app.session import ConversationContextBuilder, ManagerContextBuilder, SessionManager


class ScriptedLongChainBackend:
    """Deterministic specialist doubles behind the real CrewAI Flow."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.research_calls = 0
        self.writer_calls = 0
        self.review_calls = 0
        self.checkpoints: list[dict[str, Any]] = []
        self.results: list[Any] = []
        self.revise_with_research = True

    def build_manager_context(self, state: FlowState) -> dict[str, Any]:
        return {
            "original_goal": state.goal.model_dump(mode="python"),
            "current_run_state": state.model_dump(mode="python"),
            "pending_gaps": state.pending_gaps,
        }

    def manager_decide(self, context: dict[str, Any], _trace: Any) -> ManagerDecision:
        route = str(context["route"])
        last_agent = context.get("last_agent")
        review_decision = context.get("review_decision")
        if context.get("recovery_action"):
            action = str(context["recovery_action"])
        elif last_agent is None:
            action = "CALL_RESEARCH"
        elif last_agent == "research":
            action = "CALL_WRITER"
        elif last_agent == "writer":
            action = "CALL_REVIEWER"
        elif last_agent == "reviewer" and review_decision == "REVISE":
            action = (
                "REQUEST_MORE_EVIDENCE"
                if self.revise_with_research
                else "REQUEST_REVISION"
            )
        else:
            action = "COMPLETE"
        target = {
            "CALL_RESEARCH": "research",
            "REQUEST_MORE_EVIDENCE": "research",
            "CALL_WRITER": "writer",
            "REQUEST_REVISION": "writer",
            "CALL_REVIEWER": "reviewer",
            "COMPLETE": None,
        }[action]
        return ManagerDecision(
            next_action=action,
            target_agent=target,
            goal_alignment=f"This {route.lower().replace('_', ' ')} action advances the original goal.",
            remaining_gap=(
                "; ".join(context.get("pending_gaps") or [])
                or "complete the remaining success criteria"
            ),
            reason="scripted deterministic manager decision",
            completion_status="READY_TO_COMPLETE" if action == "COMPLETE" else "IN_PROGRESS",
        )

    def research(self, _request: Any, _route: Any, _trace: Any) -> ResearchAgentOutput:
        self.research_calls += 1
        evidence_id = f"E{self.research_calls}"
        return ResearchAgentOutput(
            status="COMPLETED",
            research_summary=f"verified introduction evidence round {self.research_calls}",
            evidence=[{"evidence_id": evidence_id, "paper_id": f"P{self.research_calls}"}],
        )

    def write(self, _request: Any, _route: Any, _research: Any, _trace: Any, *, review_round: int) -> WriterAgentOutput:
        self.writer_calls += 1
        patch_id = f"intro-patch-{self.writer_calls}"
        patch = type(
            "Patch",
            (),
            {"patch_id": patch_id, "base_hash": "base", "project_id": "project"},
        )()
        return WriterAgentOutput(status="READY", draft_patch=patch, domain_result={"patch_id": patch_id})

    def review(self, _request: Any, _route: Any, _writer: Any, _trace: Any, *, review_round: int) -> ReviewAgentOutput:
        self.review_calls += 1
        if self.review_calls == 1:
            return ReviewAgentOutput(
                decision="REVISE",
                issues=[{"message": "third paragraph needs supporting evidence"}],
                revision_instructions=["add evidence for the third paragraph"],
            )
        return ReviewAgentOutput(decision="PASS")

    def persist_checkpoint(self, state: FlowState) -> None:
        self.checkpoints.append(state.model_dump(mode="python"))

    def persist_result(self, result: Any) -> None:
        self.results.append(result)


class EvidenceRecoveryBackend(ScriptedLongChainBackend):
    """Return one insufficient research result, then a verified result."""

    def manager_decide(self, context: dict[str, Any], _trace: Any) -> ManagerDecision:
        last_agent = context.get("last_agent")
        research_status = context.get("research_status")
        if last_agent is None:
            action = "CALL_RESEARCH"
        elif last_agent == "research" and research_status == "INSUFFICIENT_EVIDENCE":
            action = "REQUEST_MORE_EVIDENCE"
        else:
            action = "COMPLETE"
        return ManagerDecision(
            next_action=action,
            target_agent="research" if action == "REQUEST_MORE_EVIDENCE" or action == "CALL_RESEARCH" else None,
            goal_alignment="The research action advances the original research goal.",
            remaining_gap="More verified evidence is required." if action != "COMPLETE" else "none",
            reason="recovery test manager decision",
            completion_status="READY_TO_COMPLETE" if action == "COMPLETE" else "IN_PROGRESS",
        )

    def research(self, _request: Any, _route: Any, _trace: Any) -> ResearchAgentOutput:
        self.research_calls += 1
        if self.research_calls == 1:
            return ResearchAgentOutput(
                status="INSUFFICIENT_EVIDENCE",
                research_summary="the first bounded search was insufficient",
                unresolved_claims=["missing supporting evidence"],
            )
        return ResearchAgentOutput(
            status="COMPLETED",
            research_summary="the second bounded search found verified evidence",
            evidence=[{"evidence_id": "E-recovered", "paper_id": "P-recovered"}],
        )


class IllegalManagerActionBackend(ScriptedLongChainBackend):
    def manager_decide(self, _context: dict[str, Any], _trace: Any) -> ManagerDecision:
        # model_construct simulates a malformed provider payload after the
        # outer Pydantic adapter; the deterministic validator must still win.
        return ManagerDecision.model_construct(
            next_action="APPLY_PATCH",
            target_agent=None,
            goal_alignment="The research action advances the original research goal.",
            remaining_gap="none",
            reason="malicious provider proposal",
            required_context_refs=[],
            completion_status="IN_PROGRESS",
        )


class AlwaysReviseBackend(ScriptedLongChainBackend):
    def review(
        self,
        _request: Any,
        _route: Any,
        _writer: Any,
        _trace: Any,
        *,
        review_round: int,
    ) -> ReviewAgentOutput:
        self.review_calls += 1
        return ReviewAgentOutput(
            decision="REVISE",
            issues=[{"message": "the same blocking issue remains"}],
            revision_instructions=["make a bounded revision"],
        )


def _flow_for(
    backend: ScriptedLongChainBackend,
    *,
    route: str = "RESEARCH",
    instruction: str = "Find verified evidence.",
    max_research_rounds: int = 2,
    max_manager_steps: int = 8,
) -> Any:
    request = OrchestrationRequest(
        request_id=f"{route.lower()}-test",
        project_id="project",
        instruction=instruction,
        task_type=route,
    )
    state = FlowState(
        project_id="project",
        run_id="run-test",
        thread_id="thread-test",
        trace_id="trace-test",
        request=request,
        goal=default_run_goal(instruction, route),
        budget=RunBudget(
            max_manager_steps=max_manager_steps,
            max_research_rounds=max_research_rounds,
            max_review_rounds=3,
        ),
        selected_route=route,
    )
    trace = CrewAITraceAdapter(
        project_id="project", session_id=None, run_id="run-test", trace_id="trace-test"
    )
    return CrewAIOrchestrationFlow(
        backend=backend,
        state=state,
        trace=trace,
        max_research_attempts=max_research_rounds,
        max_manager_steps=max_manager_steps,
    ).kickoff()


def _long_chain_flow(
    tmp_path: Path,
    *,
    revise_with_research: bool = True,
    backend: ScriptedLongChainBackend | None = None,
) -> tuple[ScriptedLongChainBackend, Any]:
    backend = backend or ScriptedLongChainBackend(tmp_path)
    backend.revise_with_research = revise_with_research
    request = OrchestrationRequest(
        request_id="intro-e2e",
        project_id="project",
        instruction="Write a complete evidence-grounded Introduction.",
        task_type="WRITE_INTRODUCTION",
    )
    goal = default_run_goal(request.instruction, request.task_type)
    state = FlowState(
        project_id="project",
        run_id="run-e2e",
        thread_id="thread-e2e",
        trace_id="trace-e2e",
        request=request,
        goal=goal,
        budget=RunBudget(max_manager_steps=12, max_research_rounds=3, max_review_rounds=3),
        selected_route="WRITE_INTRODUCTION",
    )
    trace = CrewAITraceAdapter(
        project_id="project", session_id=None, run_id="run-e2e", trace_id="trace-e2e"
    )
    return backend, CrewAIOrchestrationFlow(backend=backend, state=state, trace=trace).kickoff()


def test_long_chain_keeps_original_introduction_goal_after_local_reviewer_gap(tmp_path: Path) -> None:
    backend, result = _long_chain_flow(tmp_path)

    assert result.status == "WAITING_HUMAN_APPROVAL"
    assert backend.research_calls == 2
    assert backend.writer_calls == 2
    assert backend.review_calls == 2
    diagnostics = result.diagnostics
    assert diagnostics["original_goal"]["task_type"] == "WRITE_INTRODUCTION"
    assert diagnostics["original_goal"]["original_instruction"].startswith("Write a complete")
    assert diagnostics["review_decision"] == "PASS"
    assert diagnostics["manager_decisions"][-1]["completion_status"] == "READY_TO_COMPLETE"
    assert "third paragraph" not in diagnostics["original_goal"]["original_instruction"]


def test_reviewer_revise_can_enter_writer_without_forcing_research(
    tmp_path: Path,
) -> None:
    backend, result = _long_chain_flow(tmp_path, revise_with_research=False)

    assert result.status == "WAITING_HUMAN_APPROVAL"
    assert backend.research_calls == 1
    assert backend.writer_calls == 2
    assert backend.review_calls == 2
    actions = [item["next_action"] for item in result.diagnostics["manager_decisions"]]
    assert "REQUEST_REVISION" in actions


def test_continuous_reviewer_revise_hits_review_budget_and_fails_closed(
    tmp_path: Path,
) -> None:
    backend, result = _long_chain_flow(
        tmp_path,
        revise_with_research=False,
        backend=AlwaysReviseBackend(tmp_path),
    )

    assert result.status == "FAILED"
    assert result.error_codes == ["REVIEW_BUDGET_EXCEEDED"]
    assert backend.review_calls == 3


def test_manager_researches_after_insufficient_evidence_and_then_completes(
    tmp_path: Path,
) -> None:
    backend = EvidenceRecoveryBackend(tmp_path)

    result = _flow_for(backend)

    assert result.status == "COMPLETED"
    assert backend.research_calls == 2
    assert [item["next_action"] for item in result.diagnostics["manager_decisions"]] == [
        "CALL_RESEARCH",
        "REQUEST_MORE_EVIDENCE",
        "COMPLETE",
    ]
    assert result.diagnostics["research_attempts"] == 2


def test_research_budget_exhaustion_fails_closed(tmp_path: Path) -> None:
    backend = EvidenceRecoveryBackend(tmp_path)

    result = _flow_for(backend, max_research_rounds=1)

    assert result.status == "FAILED"
    assert result.error_codes == ["INSUFFICIENT_EVIDENCE"]
    assert backend.research_calls == 1


def test_review_budget_exhaustion_is_rejected_before_reviewer_capability() -> None:
    validator = ManagerDecisionValidator()
    goal = default_run_goal("Write a complete Introduction", "WRITE_INTRODUCTION")
    state = RunState(
        selected_route="WRITE_INTRODUCTION",
        writer_ready=True,
        draft_patch_reference={"patch_id": "patch-1"},
        review_calls=1,
    )
    decision = ManagerDecision(
        next_action="CALL_REVIEWER",
        target_agent="reviewer",
        goal_alignment="The reviewer action advances the original Introduction goal.",
        remaining_gap="validate the draft",
        reason="review budget test",
    )

    result = validator.validate(
        decision,
        goal=goal,
        state=state,
        budget=RunBudget(max_review_rounds=1),
    )

    assert result.allowed is False
    assert result.code == "REVIEW_BUDGET_EXCEEDED"


def test_manager_cannot_propose_apply_patch_even_if_provider_payload_is_malformed(
    tmp_path: Path,
) -> None:
    result = _flow_for(IllegalManagerActionBackend(tmp_path))

    assert result.status == "FAILED"
    assert result.error_codes == ["UNSUPPORTED_ACTION"]


def test_validator_rejects_premature_completion_and_goal_drift() -> None:
    goal = default_run_goal("Write a complete Introduction", "WRITE_INTRODUCTION")
    validator = ManagerDecisionValidator()
    state = RunState(
        selected_route="WRITE_INTRODUCTION",
        pending_gaps=["third paragraph needs evidence"],
        manager_steps=3,
        research_round=1,
        review_calls=1,
        review_decision="REVISE",
    )
    premature = ManagerDecision(
        next_action="COMPLETE",
        target_agent=None,
        goal_alignment="The third paragraph citation is fixed.",
        remaining_gap="none",
        reason="local issue fixed",
        completion_status="READY_TO_COMPLETE",
    )
    rejected = validator.validate(premature, goal=goal, state=state)
    assert rejected.allowed is False
    assert rejected.code == "GOAL_DRIFT"

    aligned = premature.model_copy(
        update={
            "goal_alignment": "The Introduction research loop serves the original goal.",
            "remaining_gap": "none",
        }
    )
    rejected_again = validator.validate(aligned, goal=goal, state=state)
    assert rejected_again.allowed is False
    assert rejected_again.code == "PREMATURE_COMPLETION"


def test_goal_state_checkpoint_survives_runtime_reopen_and_new_run_is_isolated(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.create("multi-turn", session_id="multi-turn")
    runtime = manager.open(session.session_id)
    goal = default_run_goal("Write a complete Introduction", "WRITE_INTRODUCTION")
    runtime.create_run(
        goal.original_instruction,
        run_id="RUN_INTRO",
        thread_id="THREAD_INTRO",
        task_type=goal.task_type,
        goal=goal,
    )
    runtime.append_message("user", "第三段短一点", run_id="RUN_INTRO")
    preference_context = ConversationContextBuilder().build(
        runtime, "第三段短一点", project_id="project"
    )
    assert preference_context["conversation_summary"]["preferences"] == ["第三段短一点"]
    assert runtime.get_run_goal("RUN_INTRO") == goal
    runtime.save_checkpoint(
        "RUN_INTRO",
        RunState(
            selected_route="WRITE_INTRODUCTION",
            pending_gaps=["third paragraph needs evidence"],
            review_round=2,
            last_specialist_result={"agent": "reviewer", "status": "REVISE"},
        ).model_dump(mode="python"),
    )

    reopened = SessionManager(tmp_path).open("multi-turn")
    assert reopened.get_run_goal("RUN_INTRO") == goal
    checkpoint = reopened.load_checkpoint("RUN_INTRO")
    assert checkpoint is not None
    assert checkpoint["run_goal"]["original_instruction"] == goal.original_instruction
    assert checkpoint["run_state"]["pending_gaps"] == ["third paragraph needs evidence"]
    assert checkpoint["run_state"]["review_round"] == 2

    other_goal = default_run_goal("Find papers about Doppler measurements", "RESEARCH")
    reopened.create_run(
        other_goal.original_instruction,
        run_id="RUN_RESEARCH",
        thread_id="THREAD_RESEARCH",
        task_type=other_goal.task_type,
        goal=other_goal,
    )
    assert reopened.get_run_goal("RUN_RESEARCH").task_type == "RESEARCH"
    assert reopened.get_run_goal("RUN_INTRO").task_type == "WRITE_INTRODUCTION"


def test_manager_context_prioritizes_authoritative_state_and_omits_raw_history() -> None:
    builder = ManagerContextBuilder(context_budget_chars=1800)
    context = builder.build_manager_context(
        original_goal=RunGoal(
            original_instruction="Write a complete Introduction",
            task_type="WRITE_INTRODUCTION",
            success_criteria=("complete Introduction",),
            hard_constraints=("do not apply patches",),
        ),
        run_state=RunState(
            selected_route="WRITE_INTRODUCTION",
            pending_gaps=["third paragraph evidence"],
            review_round=1,
            latest_specialist_result={"status": "REVISE", "issue": "third paragraph"},
        ),
        recent_conversation_summary={
            "preferences": ["第三段短一点"],
            "requirements": ["irrelevant historical chatter " * 50],
        },
    )
    assert context["original_goal"]["task_type"] == "WRITE_INTRODUCTION"
    assert context["pending_gaps"] == ["third paragraph evidence"]
    assert "recent_messages" not in context
    assert len(str(context)) <= 1800
