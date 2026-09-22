"""Evaluation of the CrewAI Scholar orchestration contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from app.orchestration.contracts import OrchestrationRequest


@dataclass(frozen=True, slots=True)
class OrchestrationEvaluationCase:
    case_id: str
    instruction: str
    task_type: str
    expected_status: str
    research_expected: bool
    review_expected: bool
    approval_expected: bool
    web_allowed: bool = False


@dataclass(frozen=True, slots=True)
class OrchestrationEvaluationReport:
    backend: str
    evaluated_at: str
    records: tuple[dict[str, Any], ...]
    metrics: Mapping[str, float | int | None]

    @property
    def passed(self) -> bool:
        return all(bool(record.get("task_success")) for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "evaluated_at": self.evaluated_at,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "records": [dict(record) for record in self.records],
        }


def default_orchestration_cases(project_id: str) -> tuple[OrchestrationEvaluationCase, ...]:
    # The fixed four-route dataset shared by local E2E and parity runs.
    return (
        OrchestrationEvaluationCase(
            "support-claim",
            "Can local LEO literature support the claim that explicit ephemeris-error compensation improves positioning estimates?",
            "SUPPORT_CLAIM",
            "COMPLETED",
            True,
            False,
            False,
            True,
        ),
        OrchestrationEvaluationCase(
            "introduction",
            "Write an evidence-grounded introduction about ephemeris-error compensation for LEO signals-of-opportunity positioning.",
            "WRITE_INTRODUCTION",
            "WAITING_HUMAN_APPROVAL",
            True,
            True,
            True,
            True,
        ),
        OrchestrationEvaluationCase(
            "conclusion",
            "Write a conclusion using only the current manuscript, confirmed facts, and confirmed contribution.",
            "WRITE_CONCLUSION",
            "WAITING_HUMAN_APPROVAL",
            False,
            True,
            True,
            False,
        ),
        OrchestrationEvaluationCase(
            "abstract",
            "Write an abstract using only the latest full manuscript state and confirmed numerical or qualitative results.",
            "WRITE_ABSTRACT",
            "WAITING_HUMAN_APPROVAL",
            False,
            True,
            True,
            False,
        ),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else {}
    if hasattr(value, "__dataclass_fields__"):
        dumped = asdict(value)
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _metadata(result: Any) -> Mapping[str, Any]:
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and diagnostics.get("trace") is not None:
        return diagnostics
    metadata = getattr(result, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _route(result: Any) -> str | None:
    value = getattr(result, "selected_route", None) or getattr(result, "task_type", None)
    if isinstance(value, str):
        return value.upper().replace("-", "_")
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        routing = metadata.get("routing")
        if isinstance(routing, Mapping) and isinstance(routing.get("task_type"), str):
            return str(routing["task_type"]).upper().replace("-", "_")
    return None


def _trace(result: Any) -> Mapping[str, Any]:
    value = _metadata(result).get("trace")
    return value if isinstance(value, Mapping) else {}


def _events(result: Any) -> tuple[Mapping[str, Any], ...]:
    trace = _trace(result)
    raw = trace.get("events") or trace.get("trace") or ()
    return tuple(value for value in raw if isinstance(value, Mapping))


def _tool_name(event: Mapping[str, Any]) -> str:
    metadata = _mapping(event.get("metadata"))
    return str(event.get("tool_name") or metadata.get("tool_name") or event.get("name") or "")


def _usage(result: Any) -> Mapping[str, Any]:
    usage = _trace(result).get("usage")
    if isinstance(usage, Mapping):
        return usage
    value = _metadata(result).get("usage")
    return value if isinstance(value, Mapping) else {}


def _research_present(result: Any) -> bool:
    specialist = getattr(result, "specialist_results", {})
    if isinstance(specialist, Mapping) and specialist.get("research") is not None:
        return True
    names = {_tool_name(event) for event in _events(result)}
    return bool(
        names & {"research_capability", "research_evidence", "LOCAL_RESEARCH", "WEB_DISCOVERY"}
    )


def _review_present(result: Any) -> bool:
    specialist = getattr(result, "specialist_results", {})
    if isinstance(specialist, Mapping) and specialist.get("reviewer") is not None:
        return True
    names = {_tool_name(event) for event in _events(result)}
    return bool(names & {"review_capability", "review_draft", "REVIEWING", "reviewer_specialist"})


def _evidence_and_citations(result: Any) -> tuple[set[str], set[str], list[Mapping[str, Any]]]:
    value = _mapping(getattr(result, "value", None))
    evidence_ids: set[str] = set()
    citation_ids: set[str] = set()
    for pack_value in value.get("evidence_packs", ()):
        pack = _mapping(pack_value)
        for item in pack.get("evidence", ()):
            evidence = _mapping(item)
            if evidence.get("evidence_id"):
                evidence_ids.add(str(evidence["evidence_id"]))
    for item in value.get("evidence", ()):
        evidence = _mapping(item)
        if evidence.get("evidence_id"):
            evidence_ids.add(str(evidence["evidence_id"]))
    patch = _mapping(value.get("patch"))
    citation_ids.update(str(item) for item in patch.get("used_evidence_ids", ()) if item)
    report = _mapping(value.get("review_report"))
    issues = [
        _mapping(item)
        for item in report.get("issues", ())
        if isinstance(item, Mapping) or hasattr(item, "model_dump")
    ]
    return evidence_ids, citation_ids, issues


def _trajectory_valid(case: OrchestrationEvaluationCase, result: Any) -> bool:
    states = [
        str(_mapping(event.get("metadata")).get("state"))
        for event in _events(result)
        if event.get("name") == "flow_transition"
        and _mapping(event.get("metadata")).get("state")
    ]
    if states:
        if states[0] != "ROUTING":
            return False
        if not case.research_expected and "RESEARCHING" in states:
            return False
        if case.research_expected and "RESEARCHING" not in states:
            return False
        if case.review_expected and "REVIEWING" not in states:
            return False
        if case.approval_expected and "WAITING_HUMAN_APPROVAL" not in states:
            return False
        if "WAITING_HUMAN_APPROVAL" in states and "REVIEWING" not in states:
            return False
        return True
    names = {_tool_name(event) for event in _events(result)}
    return not (not case.research_expected and names & {"research_evidence", "WEB_DISCOVERY"})


def evaluate_backend(
    backend: str,
    cases: Sequence[OrchestrationEvaluationCase],
    runner: Callable[[OrchestrationEvaluationCase], Any],
) -> OrchestrationEvaluationReport:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        result = runner(case)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        route = _route(result)
        status = str(getattr(result, "status", ""))
        research = _research_present(result)
        reviewed = _review_present(result)
        pending = getattr(result, "pending_action", None)
        approval = bool(getattr(result, "approval_required", False) or pending)
        evidence_ids, citation_ids, review_issues = _evidence_and_citations(result)
        citation_complete = 1.0 if not citation_ids else float(citation_ids <= evidence_ids)
        citation_scope = float(
            not any("CITATION" in str(issue.get("code", "")).upper() for issue in review_issues)
        )
        events = _events(result)
        tool_names = {
            _tool_name(event)
            for event in events
            if str(event.get("kind", "")).lower() == "tool"
        }
        allowed_tools = {
            "research_capability",
            "writer_capability",
            "review_capability",
            "research_evidence",
            "review_draft",
            "execute_scholar_skill",
            "get_project_context",
            "get_patch_status",
            "read_file",
        }
        diagnostics = getattr(result, "diagnostics", {})
        metadata = getattr(result, "metadata", {})
        forbidden = set()
        for source in (diagnostics, metadata):
            if isinstance(source, Mapping):
                forbidden.update(str(item) for item in source.get("capability_violations", ()))
                forbidden.update(
                    str(item)
                    for item in source.get("unexpected_tool_calls", ())
                    if str(item) in {"approve_patch", "apply_patch", "safe_apply", "write_file", "delete"}
                )
        trace = _trace(result)
        usage = _usage(result)
        agent_calls = trace.get("agent_call_count")
        tool_calls = trace.get("tool_call_count")
        if agent_calls is None:
            agent_calls = len(
                {
                    _mapping(event.get("metadata")).get("agent_role")
                    for event in events
                    if event.get("name") == "AgentExecutionStartedEvent"
                }
                - {None}
            )
        if tool_calls is None:
            tool_calls = len(tool_names)
        total_tokens = trace.get("total_tokens", usage.get("total_tokens"))
        cost = trace.get("cost", usage.get("cost"))
        rows.append(
            {
                "case_id": case.case_id,
                "route": route,
                "status": status,
                "task_success": float(
                    route == case.task_type
                    and status == case.expected_status
                    and research == case.research_expected
                    and reviewed == case.review_expected
                    and approval == case.approval_expected
                ),
                "routing_accuracy": float(route == case.task_type),
                "retrieval_sufficiency": float(research == case.research_expected),
                "citation_completeness": citation_complete,
                "citation_scope_precision": citation_scope,
                "tool_validity": float(tool_names <= allowed_tools),
                "capability_violation": float(bool(forbidden)),
                "trajectory_validity": float(_trajectory_valid(case, result)),
                "agent_calls": int(agent_calls or 0),
                "tool_calls": int(tool_calls or 0),
                "latency_ms": elapsed_ms,
                "token_usage": int(total_tokens) if total_tokens is not None else None,
                "cost": float(cost) if cost is not None else None,
                "tool_names": sorted(tool_names),
                "error_codes": list(getattr(result, "error_codes", ()) or ()),
            }
        )

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return round(sum(values) / len(values), 6) if values else None

    metrics: dict[str, float | int | None] = {
        "Task Success": mean("task_success"),
        "Routing Accuracy": mean("routing_accuracy"),
        "Retrieval Sufficiency": mean("retrieval_sufficiency"),
        "Citation Completeness": mean("citation_completeness"),
        "Citation Scope Precision": mean("citation_scope_precision"),
        "Tool Validity": mean("tool_validity"),
        "Capability Violation": sum(int(row["capability_violation"]) for row in rows),
        "Trajectory Validity": mean("trajectory_validity"),
        "Average Agent Calls": mean("agent_calls"),
        "Average Tool Calls": mean("tool_calls"),
        "Latency (ms)": mean("latency_ms"),
        "Token Usage": mean("token_usage"),
        "Cost": mean("cost"),
    }
    return OrchestrationEvaluationReport(
        backend=backend,
        evaluated_at=datetime.now().astimezone().isoformat(),
        records=tuple(rows),
        metrics=metrics,
    )


def request_for_case(
    case: OrchestrationEvaluationCase, project_id: str
) -> OrchestrationRequest:
    return OrchestrationRequest(
        request_id=f"EVAL_{case.case_id}",
        project_id=project_id,
        instruction=case.instruction,
        task_type=case.task_type,
    )
