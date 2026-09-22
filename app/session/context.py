"""Separate Conversation Context, Workflow State, and Project Memory.

The manager context is intentionally assembled by priority. Raw messages are
never the source of truth for progress and are not included in the manager
projection by default.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.session.runtime import SessionRuntime


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_dump(child) for child in value]
    return value


class ConversationContextBuilder:
    """Build bounded user-conversation context without copying Run state."""

    def __init__(self, recent_limit: int = 8, *, context_budget_chars: int = 6000) -> None:
        self.recent_limit = max(1, recent_limit)
        self.context_budget_chars = max(1000, context_budget_chars)

    @staticmethod
    def _bounded(value: Any, limit: int) -> str:
        text = str(value)
        return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"

    def summarize_messages(self, messages: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Retain only future-relevant user requirements and open items.

        This is deliberately extractive and deterministic. It is a summary,
        not a second agent memory: the authoritative goal/state remain in the
        Run contracts.
        """

        requirements: list[str] = []
        preferences: list[str] = []
        confirmed_facts: list[str] = []
        unresolved: list[str] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = self._bounded(message.get("content", ""), 480).strip()
            if not content:
                continue
            metadata = message.get("metadata")
            metadata_text = str(metadata).casefold()
            if role == "user":
                lowered = content.casefold()
                if any(token in lowered for token in ("shorter", "简短", "短一点", "不要", "prefer", "偏好")):
                    preferences.append(content)
                else:
                    requirements.append(content)
            elif "confirmed" in metadata_text or "fact" in metadata_text:
                confirmed_facts.append(content)
            elif "pending" in metadata_text or "unresolved" in metadata_text:
                unresolved.append(content)
        return {
            "requirements": requirements[-6:],
            "confirmed_facts": confirmed_facts[-6:],
            "preferences": preferences[-6:],
            "unresolved_items": unresolved[-6:],
        }

    def build(
        self,
        runtime: SessionRuntime,
        query: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        recent = runtime.recent_messages(self.recent_limit)
        return {
            "current_query": query,
            "recent_messages": recent,
            "conversation_summary": self.summarize_messages(recent),
            "workflow_state": None,
            "project_memory": {"project_id": project_id},
            "research_state": None,
            "project_id": project_id,
            "shared_knowledge": "resolved by Research Engine",
        }

    def build_manager_context(
        self,
        *,
        original_goal: Any,
        run_state: Any,
        project_facts: Mapping[str, Any] | None = None,
        recent_conversation_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the manager projection in explicit authority order."""

        goal = _dump(original_goal)
        state = _dump(run_state)
        state = state if isinstance(state, dict) else {"value": state}
        # Never allow a specialist-shaped state value or an unbounded decision
        # log to grow the prompt. These are the only workflow fields the
        # Manager needs; raw Flow metadata and domain objects stay out.
        state_keys = {
            "stage", "status", "current_agent", "selected_route", "research_status",
            "writer_ready", "last_agent", "last_specialist_status", "completed_steps",
            "pending_gaps", "success_criteria_status", "evidence_references",
            "research_result_reference", "domain_result_refs", "draft_patch_reference",
            "draft_patch_id", "research_round", "review_round", "manager_steps",
            "review_calls", "tool_calls", "last_specialist_result",
            "latest_specialist_result", "review_decision", "recovery_action",
            "pending_approval", "no_progress_rounds", "user_preferences",
            "manager_decisions",
        }
        bounded_state = {key: state[key] for key in state_keys if key in state}
        for key in ("completed_steps", "evidence_references", "user_preferences"):
            if isinstance(bounded_state.get(key), list):
                bounded_state[key] = bounded_state[key][-32:]
        if isinstance(bounded_state.get("manager_decisions"), list):
            bounded_state["manager_decisions"] = bounded_state["manager_decisions"][-6:]
        for key in ("last_specialist_result", "latest_specialist_result"):
            if isinstance(bounded_state.get(key), dict):
                bounded_state[key] = {
                    str(child_key): self._bounded(child_value, 1000)
                    for child_key, child_value in list(bounded_state[key].items())[:16]
                }
        project_memory = {
            str(key): _dump(value)
            for key, value in list((project_facts or {}).items())[:32]
        }
        summary = dict(recent_conversation_summary or {})
        context = {
            "original_goal": goal,
            "success_criteria": goal.get("success_criteria", []) if isinstance(goal, dict) else [],
            "hard_constraints": goal.get("hard_constraints", []) if isinstance(goal, dict) else [],
            "current_run_state": bounded_state,
            "completed_steps": bounded_state.get("completed_steps", []),
            "pending_gaps": bounded_state.get("pending_gaps", []),
            "relevant_project_facts": project_memory,
            "evidence_references": bounded_state.get("evidence_references", []),
            "draft_patch_reference": bounded_state.get("draft_patch_reference"),
            "latest_specialist_result": bounded_state.get("latest_specialist_result"),
            "recent_conversation_summary": summary,
            "authority_order": [
                "original_goal",
                "hard_constraints",
                "current_run_state",
                "relevant_project_facts",
                "latest_specialist_result",
                "recent_conversation_summary",
            ],
        }
        # Check the assembled projection, not raw history, against the budget.
        while len(str(context)) > self.context_budget_chars:
            recent = context["recent_conversation_summary"]
            if isinstance(recent, dict) and recent:
                key = next(reversed(recent))
                recent.pop(key, None)
                continue
            latest = context.get("latest_specialist_result")
            if isinstance(latest, dict) and latest:
                latest.pop(next(reversed(latest)), None)
                continue
            break
        return context


class ManagerContextBuilder(ConversationContextBuilder):
    """Named façade for call sites that assemble Manager-only context."""
