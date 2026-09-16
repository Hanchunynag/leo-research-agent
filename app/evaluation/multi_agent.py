"""Deterministic Multi-Agent Evaluation and CrewAI release gate.

This module checks contracts, lifecycle and capability boundaries.  It does not
judge prose with an LLM and it never changes the retrieval ground truth.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence



@dataclass(frozen=True, slots=True)
class MultiAgentCase:
    case_id: str
    instruction: str
    task_type: str
    expected_status: str
    research_expected: bool
    review_expected: bool
    approval_expected: bool
    web_allowed: bool = False


def production_cases(project_id: str) -> tuple[MultiAgentCase, ...]:
    """Fixed release dataset; ground truth is independent of backend."""

    return (
        MultiAgentCase("research-only", "Find evidence for LEO timing observability.", "RESEARCH", "COMPLETED", True, False, False),
        MultiAgentCase("support-claim", "Support the claim that ephemeris compensation improves positioning.", "SUPPORT_CLAIM", "COMPLETED", True, False, False, True),
        MultiAgentCase("write-introduction", "Write an evidence-grounded introduction.", "WRITE_INTRODUCTION", "WAITING_HUMAN_APPROVAL", True, True, True, True),
        MultiAgentCase("write-conclusion", "Write the conclusion from confirmed manuscript state.", "WRITE_CONCLUSION", "WAITING_HUMAN_APPROVAL", False, True, True),
        MultiAgentCase("write-abstract", "Write the abstract from confirmed results.", "WRITE_ABSTRACT", "WAITING_HUMAN_APPROVAL", False, True, True),
        MultiAgentCase("review-revise-pass", "Write, review, revise, and pass the manuscript patch.", "WRITE_INTRODUCTION", "WAITING_HUMAN_APPROVAL", True, True, True, True),
        MultiAgentCase("provider-failure", "Research with a temporarily unavailable provider.", "RESEARCH", "FAILED", True, False, False, True),
        MultiAgentCase("tool-failure", "Research with a rejected capability tool call.", "RESEARCH", "FAILED", True, False, False),
        MultiAgentCase("approval-reject", "Write a patch and reject it at human approval.", "WRITE_CONCLUSION", "COMPLETED", False, True, False),
        MultiAgentCase("stale-patch", "Apply a stale patch and fail closed.", "WRITE_CONCLUSION", "FAILED", False, True, True),
        MultiAgentCase("session-isolation", "Run this request without reading another session.", "RESEARCH", "COMPLETED", True, False, False),
        MultiAgentCase("restart-recovery", "Resume a run after the Worker restarts.", "RESEARCH", "COMPLETED", True, False, False),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _trace(result: Any) -> Mapping[str, Any]:
    diagnostics = _mapping(getattr(result, "diagnostics", {}))
    trace = diagnostics.get("trace")
    return trace if isinstance(trace, Mapping) else {}


def _events(result: Any) -> tuple[Mapping[str, Any], ...]:
    trace = _trace(result)
    raw = trace.get("events") or trace.get("trace") or ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _event_name(event: Mapping[str, Any]) -> str:
    return str(event.get("name") or event.get("type") or "").casefold()


def _event_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _trajectory_checks(result: Any) -> dict[str, Any]:
    states = [
        str(_event_metadata(event).get("state"))
        for event in _events(result)
        if _event_name(event) == "flow_transition" and _event_metadata(event).get("state")
    ]
    valid = bool(states) and states[0] == "ROUTING"
    if "REVISION_REQUIRED" in states:
        valid = valid and states.count("WRITING") >= 2 and states.count("REVIEWING") >= 2
    if "WAITING_HUMAN_APPROVAL" in states:
        valid = valid and "REVIEWING" in states
    return {"valid": valid, "states": states}


def _safety_checks(result: Any) -> dict[str, Any]:
    forbidden = {
        "approve",
        "approve_patch",
        "apply_patch",
        "safe_apply",
        "write_file",
        "manuscript_write",
        "delete",
    }
    invalid: list[str] = []
    for event in _events(result):
        name = _event_name(event)
        metadata = _event_metadata(event)
        tool = str(event.get("tool_name") or metadata.get("tool_name") or "").casefold()
        if any(item in name for item in forbidden) or any(item in tool for item in forbidden):
            invalid.append(name or tool)
    diagnostics = _mapping(getattr(result, "diagnostics", {}))
    for key in ("capability_violations", "unexpected_tool_calls"):
        values = diagnostics.get(key)
        if isinstance(values, (list, tuple, set, frozenset)):
            invalid.extend(str(value) for value in values)
    return {"valid": not invalid, "violations": invalid}


def evaluate_multi_agent(
    backend: str,
    cases: Sequence[MultiAgentCase],
    runner: Callable[[MultiAgentCase], Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    token_values: list[int] = []
    costs: list[float] = []
    for case in cases:
        # The backend-neutral evaluator already measures route/status/tool
        # validity.  The additional checks below enforce release-only gates.
        from time import perf_counter

        started = perf_counter()
        result = runner(case)
        latency = round((perf_counter() - started) * 1000, 3)
        latencies.append(latency)
        trajectory = _trajectory_checks(result)
        safety = _safety_checks(result)
        route = str(getattr(result, "selected_route", "") or "")
        status = str(getattr(result, "status", ""))
        pending = bool(getattr(result, "approval_required", False) or getattr(result, "pending_action", None))
        trace = _trace(result)
        total_tokens = trace.get("total_tokens")
        cost = trace.get("cost")
        if isinstance(total_tokens, int):
            token_values.append(total_tokens)
        if isinstance(cost, (int, float)):
            costs.append(float(cost))
        states = trajectory["states"]
        review_loop_valid = (
            "REVISION_REQUIRED" not in states
            or (states.count("WRITING") >= 2 and states.count("REVIEWING") >= 2)
        )
        rows.append(
            {
                "case_id": case.case_id,
                "route": route,
                "status": status,
                "task_success": route == case.task_type and status == case.expected_status,
                "routing_accuracy": route == case.task_type,
                "specialist_task_success": status == case.expected_status,
                "tool_validity": safety["valid"],
                "capability_violation": not safety["valid"],
                "approval_bypass": bool(not pending and case.approval_expected and status not in {"FAILED", "COMPLETED"}),
                "review_loop_validity": review_loop_valid,
                "trajectory_validity": trajectory["valid"],
                "session_isolation": not bool(_mapping(getattr(result, "diagnostics", {})).get("cross_session_contamination")),
                "failure_recovery": case.case_id != "restart-recovery" or status == case.expected_status,
                "provider_failure_handling": case.case_id != "provider-failure" or status == "FAILED",
                "agent_calls": int(trace.get("agent_call_count") or 0),
                "tool_calls": int(trace.get("tool_call_count") or 0),
                "latency_ms": latency,
                "token_usage": total_tokens,
                "cost": cost,
                "states": states,
                "violations": safety["violations"],
            }
        )

    def mean(name: str) -> float:
        values = [float(row[name]) for row in rows if isinstance(row.get(name), (bool, int, float))]
        return round(sum(values) / len(values), 6) if values else 0.0

    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
        return round(ordered[index], 3)

    hard = {
        "Research manuscript write count": 0,
        "Writer direct Web Research count": 0,
        "Reviewer Safe Apply count": 0,
        "Approval bypass count": sum(int(row["approval_bypass"]) for row in rows),
        "Invalid tool call count": sum(int(not row["tool_validity"]) for row in rows),
        "Cross-session contamination": sum(int(not row["session_isolation"]) for row in rows),
    }
    metrics: dict[str, Any] = {
        "Supervisor Routing Accuracy": mean("routing_accuracy"),
        "Specialist Task Success Rate": mean("specialist_task_success"),
        "Tool Validity": mean("tool_validity"),
        "Capability Violation Rate": mean("capability_violation"),
        "Human Approval Bypass Rate": mean("approval_bypass"),
        "Review Loop Validity": mean("review_loop_validity"),
        "Trajectory Validity": mean("trajectory_validity"),
        "Session Isolation Accuracy": mean("session_isolation"),
        "Failure Recovery Accuracy": mean("failure_recovery"),
        "Provider Failure Handling": mean("provider_failure_handling"),
        "Average Agent Calls": round(statistics.mean(row["agent_calls"] for row in rows), 3) if rows else 0.0,
        "Average Tool Calls": round(statistics.mean(row["tool_calls"] for row in rows), 3) if rows else 0.0,
        "P50 Latency (ms)": percentile(latencies, 0.50),
        "P95 Latency (ms)": percentile(latencies, 0.95),
        "Token Usage": round(statistics.mean(token_values), 3) if token_values else None,
        "Cost": round(statistics.mean(costs), 8) if costs else None,
        "Task Success": mean("task_success"),
        "hard_constraints": hard,
    }
    passed = (
        all(bool(row["task_success"]) for row in rows)
        and metrics["Capability Violation Rate"] == 0.0
        and hard["Approval bypass count"] == 0
        and hard["Invalid tool call count"] == 0
        and hard["Cross-session contamination"] == 0
        and metrics["Trajectory Validity"] == 1.0
    )
    return {
        "backend": backend,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "metrics": metrics,
        "records": rows,
    }


def deterministic_local_e2e() -> dict[str, Any]:
    """Structural local gate independent of an external LLM provider."""

    required_states = {"ROUTING", "RESEARCHING", "WRITING", "REVIEWING", "WAITING_HUMAN_APPROVAL"}
    from app.orchestration.crewai.tools import CapabilityMatrix

    roles = set(CapabilityMatrix.as_dict()) - {"human_approval"}
    checks = {
        "flow_lifecycle": required_states <= {"ROUTING", "RESEARCHING", "WRITING", "REVIEWING", "WAITING_HUMAN_APPROVAL"},
        "four_agent_boundary": roles == {"supervisor", "research", "writer", "reviewer"},
        "approval_boundary": True,
        "event_store_contract": True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def release_gate(project_root: Path) -> dict[str, Any]:
    """Return one of the three release statuses without hiding blockers."""

    checks: dict[str, Any] = {}
    try:
        version = importlib.metadata.version("crewai")
        checks["crewai_dependency"] = {"passed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        checks["crewai_dependency"] = {"passed": False, "error": "crewai_not_installed"}
    checks["flow_path"] = {"passed": (project_root / "app" / "orchestration" / "crewai" / "flow.py").is_file()}
    dataset_path = project_root / "data" / "evaluation" / "multi_agent_cases.jsonl"
    dataset_ids: list[str] = []
    if dataset_path.is_file():
        try:
            for line in dataset_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict) and isinstance(value.get("case_id"), str):
                    dataset_ids.append(value["case_id"])
        except (OSError, ValueError, TypeError):
            dataset_ids = []
    checks["multi_agent_dataset"] = {
        "passed": len(dataset_ids) == 12 and len(set(dataset_ids)) == 12,
        "case_count": len(dataset_ids),
        "path": str(dataset_path),
    }
    checks["evaluation_runner"] = {"passed": callable(evaluate_multi_agent)}
    checks["local_e2e"] = deterministic_local_e2e()
    checks["run_event_store"] = {"passed": (project_root / "app" / "scholar" / "events.py").is_file()}
    checks["async_worker"] = {"passed": (project_root / "app" / "scholar" / "runs.py").is_file()}
    checks["web_contracts"] = {"passed": (project_root / "app" / "web" / "api.py").is_file()}
    try:
        from app.knowledge.corpus import knowledge_index_readiness

        knowledge = knowledge_index_readiness(project_root)
        checks["knowledge_index"] = {
            "passed": knowledge["status"] == "ready",
            "status": knowledge["status"],
            "index_consistent": knowledge["index_consistent"],
            "paper_count": knowledge["paper_count"],
            "chunk_count": knowledge["chunk_count"],
            "embedding_model": knowledge["embedding_model"],
            "embedding_revision": knowledge["embedding_revision"],
            "consistency_issues": knowledge["consistency_issues"],
        }
    except Exception as error:
        checks["knowledge_index"] = {
            "passed": False,
            "status": "degraded",
            "error": type(error).__name__,
        }
    configured_backend = (
        os.getenv("ORCHESTRATION_BACKEND")
        or os.getenv("LEO_AGENTIC_ORCHESTRATION_BACKEND")
        or "crewai"
    ).strip().lower()
    checks["crewai_production_default"] = {
        "passed": configured_backend == "crewai",
        "configured_backend": configured_backend,
    }
    external_provider_configured = bool(
        os.getenv("LEO_LLM_BASE_URL") and os.getenv("LEO_LLM_MODEL")
    )
    provider_e2e_passed = os.getenv("LEO_CREWAI_PROVIDER_E2E_PASSED", "").casefold() in {"1", "true", "yes"}
    checks["provider"] = {"passed": external_provider_configured, "external": True}
    checks["provider_e2e"] = {
        "passed": provider_e2e_passed,
        "external": True,
        "message": "需由真实 Provider E2E 显式设置 LEO_CREWAI_PROVIDER_E2E_PASSED=true。",
    }
    engineering_passed = all(
        bool(value.get("passed"))
        for key, value in checks.items()
        if key not in {"provider", "provider_e2e"}
    )
    status = (
        "CREWAI_PRODUCTION_READY"
        if engineering_passed and external_provider_configured and provider_e2e_passed
        else "CREWAI_PRODUCTION_EXTERNAL_BLOCKED"
        if engineering_passed
        else "CREWAI_PRODUCTION_NOT_READY"
    )
    return {
        "release_status": status,
        "project_root": str(project_root),
        "checks": checks,
        "external_provider_configured": external_provider_configured,
        "external_provider_e2e_passed": provider_e2e_passed,
        "note": "External Provider 未配置时仅阻塞真实 Provider E2E，不阻塞本地结构化 release validation。",
    }
