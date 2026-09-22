"""Fail-closed validation around Manager Agent cognition.

The Manager is allowed to propose one bounded cognitive action. This module
owns the parts that must not be delegated to an LLM: goal anchoring, lifecycle
transitions, completion gates, budgets, and loop detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.orchestration.contracts import (
    ManagerAction,
    ManagerDecision,
    RunBudget,
    RunGoal,
    RunState,
    Route,
)


class ManagerDecisionRejected(ValueError):
    """Raised by ``assert_valid`` when a proposal must not execute."""

    code = "MANAGER_DECISION_REJECTED"


@dataclass(frozen=True, slots=True)
class DecisionValidationResult:
    allowed: bool
    reason: str = ""
    code: str = ""
    next_action: ManagerAction | None = None


def _budget_value(
    budget: RunBudget | Mapping[str, Any] | None,
    name: str,
    default: int | None,
) -> int | None:
    if budget is None:
        return default
    mapping = budget.model_dump(mode="python") if isinstance(budget, RunBudget) else budget
    value = mapping.get(name)
    return int(value) if value is not None else default


class ManagerDecisionValidator:
    """Validate one Manager proposal against the authoritative Run state."""

    _TARGETS: dict[str, str | None] = {
        "CALL_RESEARCH": "research",
        "REQUEST_MORE_EVIDENCE": "research",
        "CALL_WRITER": "writer",
        "REQUEST_REVISION": "writer",
        "CALL_REVIEWER": "reviewer",
        "COMPLETE": None,
    }

    def __init__(
        self,
        *,
        max_research_attempts: int = 2,
        max_review_calls: int = 3,
        max_manager_steps: int = 12,
        max_no_progress_rounds: int = 2,
    ) -> None:
        self.max_research_attempts = max(0, int(max_research_attempts))
        self.max_review_calls = max(0, int(max_review_calls))
        self.max_manager_steps = max(1, int(max_manager_steps))
        self.max_no_progress_rounds = max(1, int(max_no_progress_rounds))

    @staticmethod
    def _ok(decision: ManagerDecision) -> DecisionValidationResult:
        return DecisionValidationResult(True, next_action=decision.next_action)

    @staticmethod
    def _bad(code: str, reason: str) -> DecisionValidationResult:
        return DecisionValidationResult(False, reason=reason, code=code)

    @staticmethod
    def _goal_anchors(goal: RunGoal) -> set[str]:
        text = f"{goal.task_type} {goal.original_instruction}".casefold()
        anchors = {
            "introduction": "introduction",
            "intro": "introduction",
            "引言": "introduction",
            "conclusion": "conclusion",
            "结论": "conclusion",
            "abstract": "abstract",
            "摘要": "abstract",
            "review": "review",
            "审阅": "review",
            "research": "research",
            "研究": "research",
            "claim": "claim",
            "证据": "evidence",
            "文献": "evidence",
        }
        return {canonical for token, canonical in anchors.items() if token in text}

    def _validate_goal_alignment(
        self, decision: ManagerDecision, goal: RunGoal, state: RunState
    ) -> DecisionValidationResult | None:
        alignment = decision.goal_alignment.strip().casefold()
        if not alignment:
            return self._bad("GOAL_DRIFT", "goal_alignment 不能为空。")
        anchors = self._goal_anchors(goal)
        alignment_anchors = anchors & {
            token
            for token in (
                "introduction",
                "conclusion",
                "abstract",
                "review",
                "research",
                "claim",
                "evidence",
            )
            if token in alignment
        }
        local_only = any(
            marker in alignment
            for marker in (
                "third paragraph",
                "第三段",
                "one citation",
                "single citation",
                "local issue",
                "局部",
            )
        )
        if local_only and not alignment_anchors:
            return self._bad(
                "GOAL_DRIFT",
                "goal_alignment 只描述局部问题，没有重新锚定 OriginalGoal。",
            )
        if (
            anchors
            and not alignment_anchors
            and "original goal" not in alignment
            and "原始目标" not in alignment
            and "success criteria" not in alignment
            and "成功标准" not in alignment
            and "requested task" not in alignment
            and "最终任务" not in alignment
        ):
            return self._bad(
                "GOAL_DRIFT",
                "goal_alignment 未说明该动作如何服务 OriginalGoal。",
            )
        if state.selected_route and state.selected_route != goal.task_type:
            return self._bad(
                "GOAL_ROUTE_MISMATCH",
                f"RunState route={state.selected_route} 与 RunGoal task_type={goal.task_type} 不一致。",
            )
        return None

    def validate(
        self,
        decision: ManagerDecision,
        *,
        goal: RunGoal | None = None,
        state: RunState | None = None,
        budget: RunBudget | Mapping[str, Any] | None = None,
        # Compatibility seam for callers that used the pre-RunState API.
        route: Route | None = None,
        phase: str | None = None,
        manager_steps: int | None = None,
        research_attempts: int | None = None,
        review_calls: int | None = None,
        research_status: str | None = None,
        writer_ready: bool = False,
        review_decision: str | None = None,
        recovery_action: ManagerAction | None = None,
        context_refs: Mapping[str, Any] | None = None,
    ) -> DecisionValidationResult:
        del phase, context_refs
        if goal is None:
            goal = RunGoal(
                original_instruction="default manager request",
                task_type=route or "RESEARCH",
                success_criteria=("complete the requested route",),
            )
        if state is None:
            state = RunState(
                selected_route=route,
                manager_steps=manager_steps or 0,
                research_round=research_attempts or 0,
                review_calls=review_calls or 0,
                recovery_action=recovery_action,
                review_decision=(
                    review_decision
                    if review_decision in {"PASS", "REVISE", "REJECT"}
                    else None
                ),
            )
        current_route = state.selected_route or route or goal.task_type
        state = state.model_copy(update={"selected_route": current_route})
        max_steps = _budget_value(budget, "max_manager_steps", self.max_manager_steps)
        max_research = _budget_value(
            budget, "max_research_rounds", self.max_research_attempts
        )
        max_reviews = _budget_value(budget, "max_review_rounds", self.max_review_calls)
        max_tools = _budget_value(budget, "max_tool_calls", None)

        if max_steps is not None and state.manager_steps >= max_steps:
            return self._bad(
                "MANAGER_STEP_BUDGET_EXCEEDED", "manager step budget exceeded"
            )
        if max_tools is not None and state.tool_calls > max_tools:
            return self._bad("TOOL_BUDGET_EXCEEDED", "capability tool budget exceeded")
        if state.no_progress_rounds >= self.max_no_progress_rounds and decision.next_action in {
            "CALL_RESEARCH",
            "REQUEST_MORE_EVIDENCE",
            "REQUEST_REVISION",
        }:
            return self._bad("NO_PROGRESS_LOOP", "连续多轮执行没有产生新的有效进展。")

        goal_error = self._validate_goal_alignment(decision, goal, state)
        if goal_error is not None:
            return goal_error
        expected_target = self._TARGETS.get(decision.next_action)
        if expected_target is None and decision.next_action != "COMPLETE":
            return self._bad(
                "UNSUPPORTED_ACTION", f"unsupported action: {decision.next_action}"
            )
        if decision.target_agent != expected_target:
            return self._bad(
                "INVALID_TARGET",
                f"action {decision.next_action} requires target_agent={expected_target}",
            )
        if decision.completion_status == "BLOCKED":
            return self._bad(
                "MANAGER_BLOCKED",
                decision.remaining_gap or "Manager reported a blocking gap.",
            )
        if (
            decision.next_action != "COMPLETE"
            and decision.completion_status != "IN_PROGRESS"
        ):
            return self._bad(
                "INVALID_COMPLETION_STATUS", "non-terminal action must use IN_PROGRESS"
            )
        if (
            decision.next_action == "COMPLETE"
            and decision.completion_status != "READY_TO_COMPLETE"
        ):
            return self._bad(
                "INVALID_COMPLETION_STATUS", "COMPLETE must use READY_TO_COMPLETE"
            )

        if recovery_action is not None and decision.next_action != recovery_action:
            return self._bad(
                "RECOVERY_ACTION_MISMATCH",
                f"recovery requires replaying {recovery_action}, got {decision.next_action}",
            )

        if decision.next_action in {"CALL_RESEARCH", "REQUEST_MORE_EVIDENCE"}:
            if current_route not in {"RESEARCH", "SUPPORT_CLAIM", "WRITE_INTRODUCTION"}:
                return self._bad(
                    "ILLEGAL_TRANSITION", f"research is not allowed for route={current_route}"
                )
            if max_research is not None and state.research_round >= max_research:
                return self._bad("RESEARCH_BUDGET_EXCEEDED", "research budget exceeded")
            if (
                decision.next_action == "REQUEST_MORE_EVIDENCE"
                and state.last_specialist_result is None
                and state.last_agent not in {"research", "reviewer"}
                and state.recovery_action is None
            ):
                return self._bad(
                    "ILLEGAL_TRANSITION",
                    "more evidence requires a bounded specialist result",
                )
            return self._ok(decision)

        if decision.next_action in {"CALL_WRITER", "REQUEST_REVISION"}:
            if current_route not in {
                "WRITE_INTRODUCTION",
                "WRITE_CONCLUSION",
                "WRITE_ABSTRACT",
            }:
                return self._bad(
                    "ILLEGAL_TRANSITION", f"writing is not allowed for route={current_route}"
                )
            if (
                decision.next_action == "REQUEST_REVISION"
                and state.review_decision != "REVISE"
            ):
                return self._bad(
                    "ILLEGAL_TRANSITION", "revision requires reviewer decision=REVISE"
                )
            if decision.next_action == "CALL_WRITER" and state.last_agent == "writer":
                return self._bad(
                    "STATE_REGRESSION", "Writer cannot immediately replace its own DraftPatch without a review transition"
                )
            if decision.next_action == "CALL_WRITER" and state.review_decision == "PASS":
                return self._bad(
                    "STATE_REGRESSION", "a passing review cannot transition back to Writer"
                )
            if (
                current_route == "WRITE_INTRODUCTION"
                and not writer_ready
                and state.research_round == 0
            ):
                return self._bad(
                    "ILLEGAL_TRANSITION",
                    "Introduction Writer requires completed evidence",
                )
            return self._ok(decision)

        if decision.next_action == "CALL_REVIEWER":
            if max_reviews is not None and state.review_calls >= max_reviews:
                return self._bad("REVIEW_BUDGET_EXCEEDED", "review budget exceeded")
            if current_route == "REVIEW":
                return self._ok(decision)
            if current_route not in {
                "WRITE_INTRODUCTION",
                "WRITE_CONCLUSION",
                "WRITE_ABSTRACT",
            }:
                return self._bad(
                    "ILLEGAL_TRANSITION", f"review is not allowed for route={current_route}"
                )
            if not writer_ready:
                return self._bad(
                    "ILLEGAL_TRANSITION", "Reviewer requires a Writer DraftPatch"
                )
            if state.last_agent == "reviewer":
                return self._bad(
                    "STATE_REGRESSION", "Reviewer must not call itself without a new Writer result"
                )
            return self._ok(decision)

        # COMPLETE is only legal when deterministic state says that every
        # criterion is satisfied. The model's prose is never authoritative.
        if state.pending_gaps:
            return self._bad(
                "PREMATURE_COMPLETION",
                "pending_gap remains: " + "; ".join(state.pending_gaps[:3]),
            )
        criteria = state.success_criteria_status
        if criteria and not all(criteria.values()):
            return self._bad(
                "PREMATURE_COMPLETION", "one or more success criteria are not satisfied"
            )
        if current_route in {"RESEARCH", "SUPPORT_CLAIM"}:
            if research_status != "COMPLETED":
                return self._bad("PREMATURE_COMPLETION", "research route is not verified")
        elif current_route == "REVIEW":
            if (state.review_decision or review_decision) != "PASS":
                return self._bad(
                    "PREMATURE_COMPLETION", "review route requires Reviewer PASS"
                )
        else:
            if (state.review_decision or review_decision) != "PASS":
                return self._bad(
                    "PREMATURE_COMPLETION", "writing route requires Reviewer PASS"
                )
            if not state.draft_patch_reference and not writer_ready:
                return self._bad(
                    "PREMATURE_COMPLETION", "writing route lacks DraftPatch reference"
                )
        return self._ok(decision)

    def assert_valid(self, decision: ManagerDecision, **kwargs: Any) -> ManagerDecision:
        result = self.validate(decision, **kwargs)
        if not result.allowed:
            error = ManagerDecisionRejected(f"{result.code}: {result.reason}")
            error.code = result.code or ManagerDecisionRejected.code
            raise error
        return decision
