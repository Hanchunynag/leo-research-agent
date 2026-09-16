"""Deterministic evaluation of Scholar Harness decisions.

The suite evaluates routing, visibility and domain boundaries from the existing
Harness metadata/trace.  It does not judge prose quality and does not create a
second agent or evaluation runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class HarnessEvaluationCase:
    case_id: str
    instruction: str
    project_id: str
    expected_skill: str
    research_expected: bool | None
    web_allowed: bool
    expected_result_type: str | None = None
    review_expected: bool | None = None
    resume_expected: bool | None = None
    forbidden_tools: frozenset[str] = frozenset(
        {
            "approve_patch",
            "apply_patch",
            "write_file",
            "edit_file",
            "delete",
            "execute",
            "sqlite_query",
            "qdrant",
            "http_request",
        }
    )
    task_type: str | None = None


@dataclass(frozen=True, slots=True)
class HarnessEvaluationRecord:
    case_id: str
    passed: bool
    failures: tuple[str, ...] = ()
    selected_skill: str | None = None
    research_invoked: bool = False
    web_invoked: bool = False
    reviewer_invoked: bool = False
    forbidden_tool_calls: int = 0
    context_isolation_violations: int = 0
    result_type: str | None = None
    domain_result_valid: bool = False
    context_tokens: int = 0
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class HarnessEvaluationReport:
    records: tuple[HarnessEvaluationRecord, ...]
    metrics: Mapping[str, float | int]

    @property
    def passed(self) -> bool:
        return all(record.passed for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "records": [
                {
                    "case_id": record.case_id,
                    "passed": record.passed,
                    "failures": list(record.failures),
                    "selected_skill": record.selected_skill,
                    "research_invoked": record.research_invoked,
                    "web_invoked": record.web_invoked,
                    "reviewer_invoked": record.reviewer_invoked,
                    "forbidden_tool_calls": record.forbidden_tool_calls,
                    "context_isolation_violations": record.context_isolation_violations,
                    "result_type": record.result_type,
                    "domain_result_valid": record.domain_result_valid,
                    "context_tokens": record.context_tokens,
                    "resumed": record.resumed,
                }
                for record in self.records
            ],
        }


def _trace_events(result: Any) -> tuple[Mapping[str, Any], ...]:
    metadata = getattr(result, "metadata", {})
    trace = metadata.get("trace", {}) if isinstance(metadata, Mapping) else {}
    events = trace.get("trace", ()) if isinstance(trace, Mapping) else ()
    return tuple(value for value in events if isinstance(value, Mapping))


def _trace_tool_names(events: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect both legacy event names and CrewAI tool metadata names."""

    names: set[str] = set()
    for event in events:
        if event.get("kind") != "tool":
            continue
        name = event.get("name")
        if name:
            names.add(str(name))
        metadata = event.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("tool_name"):
            names.add(str(metadata["tool_name"]))
        if event.get("tool_name"):
            names.add(str(event["tool_name"]))
    return names


def _skill_key(value: Any) -> str:
    """Normalize Legacy skill names and CrewAI route names for comparison."""

    return str(value or "").strip().casefold().replace("_", "-")


def _domain_result_is_valid(result: Any, expected_type: str | None) -> bool:
    """Validate the outer Scholar result without judging prose quality.

    A failed/ interrupted result can still carry the domain value type from a
    partially completed execution.  It must not count as a valid release
    result merely because that type happens to match the fixture expectation.
    """

    if expected_type is None or getattr(result, "result_type", None) != expected_type:
        return False
    status = getattr(result, "status", None)
    return status not in {"FAILED", "INTERRUPTED"}


class ScholarHarnessEvaluationSuite:
    """Run fixed Harness cases using an injected Scholar request callable."""

    def evaluate_result(
        self,
        case: HarnessEvaluationCase,
        result: Any,
    ) -> HarnessEvaluationRecord:
        metadata = getattr(result, "metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        events = _trace_events(result)
        tool_names = _trace_tool_names(events)
        selected = metadata.get("selected_skill")
        research = bool(
            tool_names
            & {
                "research_capability",
                "research_evidence",
                "LOCAL_RESEARCH",
                "WEB_DISCOVERY",
            }
            or any(
                event.get("name") == "LOCAL_RESEARCH"
                and event.get("status") == "succeeded"
                for event in events
            )
        )
        web = bool(
            tool_names & {"literature.search", "WEB_DISCOVERY", "web_research"}
            or any(
                event.get("name") == "WEB_DISCOVERY"
                and event.get("status") == "succeeded"
                for event in events
            )
        )
        reviewed = bool(tool_names & {"review_draft", "review_capability"})
        resumed = bool(metadata.get("resumed", False))
        observed_forbidden = set(metadata.get("unexpected_tool_calls", ())) if isinstance(metadata.get("unexpected_tool_calls"), (list, tuple, set, frozenset)) else set()
        forbidden_calls = len((tool_names | {str(value) for value in observed_forbidden}) & set(case.forbidden_tools))
        visible = metadata.get("visible_tools", {})
        visible_names = {
            str(name)
            for values in visible.values()
            if isinstance(values, (list, tuple, set, frozenset))
            for name in values
        } if isinstance(visible, Mapping) else set()
        isolation_violations = len(visible_names & set(case.forbidden_tools))
        if not case.web_allowed and web:
            isolation_violations += 1
        failures: list[str] = []
        if _skill_key(selected) != _skill_key(case.expected_skill):
            failures.append("TASK_ROUTING_MISMATCH")
        if case.research_expected is True and not research:
            failures.append("REQUIRED_RESEARCH_MISSED")
        if case.research_expected is False and research:
            failures.append("UNEXPECTED_RESEARCH")
        if case.review_expected is True and not reviewed:
            failures.append("REQUIRED_REVIEW_MISSED")
        if case.review_expected is False and reviewed:
            failures.append("UNEXPECTED_REVIEW")
        if case.resume_expected is True and not resumed:
            failures.append("RESUME_MISSED")
        if case.expected_result_type and not _domain_result_is_valid(result, case.expected_result_type):
            failures.append("INVALID_DOMAIN_RESULT")
        if forbidden_calls:
            failures.append("FORBIDDEN_TOOL_CALL")
        if isolation_violations:
            failures.append("CONTEXT_ISOLATION_VIOLATION")
        trace = metadata.get("trace", {})
        usage = trace.get("usage", {}) if isinstance(trace, Mapping) else {}
        context_tokens = int(usage.get("context_tokens", 0)) if isinstance(usage, Mapping) else 0
        return HarnessEvaluationRecord(
            case_id=case.case_id,
            passed=not failures,
            failures=tuple(failures),
            selected_skill=str(selected) if selected is not None else None,
            research_invoked=research,
            web_invoked=web,
            reviewer_invoked=reviewed,
            forbidden_tool_calls=forbidden_calls,
            context_isolation_violations=isolation_violations,
            result_type=getattr(result, "result_type", None),
            domain_result_valid=_domain_result_is_valid(result, case.expected_result_type),
            context_tokens=context_tokens,
            resumed=resumed,
        )

    def run(
        self,
        cases: Iterable[HarnessEvaluationCase],
        runner: Callable[[HarnessEvaluationCase], Any],
    ) -> HarnessEvaluationReport:
        case_values = tuple(cases)
        records = tuple(self.evaluate_result(case, runner(case)) for case in case_values)
        count = len(records)
        failures = sum(not record.passed for record in records)
        forbidden = sum(record.forbidden_tool_calls for record in records)
        isolation = sum(record.context_isolation_violations for record in records)
        unexpected_research = sum(
            1
            for record, case in zip(records, case_values, strict=False)
            if case.research_expected is False and record.research_invoked
        )
        required_miss = sum(
            1
            for record, case in zip(records, case_values, strict=False)
            if case.research_expected is True and not record.research_invoked
        )
        resume_cases = [
            record for record, case in zip(records, case_values, strict=False)
            if case.resume_expected is not None
        ]
        return HarnessEvaluationReport(
            records,
            {
                "Task Routing Accuracy": sum(
                    record.selected_skill == case.expected_skill
                    for record, case in zip(records, case_values, strict=False)
                ) / count if count else 1.0,
                "Forbidden Tool Call Count": forbidden,
                "Unexpected Research Rate": unexpected_research / count if count else 0.0,
                "Required Research Miss Rate": required_miss / count if count else 0.0,
                "Domain Result Validity": sum(
                    record.domain_result_valid
                    for record, case in zip(records, case_values, strict=False)
                    if case.expected_result_type is not None
                ) / max(1, sum(case.expected_result_type is not None for case in case_values)),
                "Context Isolation Violation Count": isolation,
                "Resume Success Rate": (
                    sum(record.resumed for record in resume_cases) / len(resume_cases)
                    if resume_cases else 0.0
                ),
                "Capability Violation Count": forbidden + isolation,
                "failed_cases": failures,
            },
        )


HarnessEvaluationSuite = ScholarHarnessEvaluationSuite


def default_harness_cases(project_id: str) -> tuple[HarnessEvaluationCase, ...]:
    """The stable four-skill decision fixture used by CI/evaluation tooling."""

    return (
        HarnessEvaluationCase("support-claim", "support this claim", project_id, "support-claim", True, True, "ClaimSupportResult", False, task_type="SUPPORT_CLAIM"),
        HarnessEvaluationCase("introduction", "write introduction", project_id, "write-introduction", None, True, None, True, task_type="WRITE_INTRODUCTION"),
        HarnessEvaluationCase("conclusion", "write conclusion", project_id, "write-conclusion", False, False, None, True, task_type="WRITE_CONCLUSION"),
        HarnessEvaluationCase("abstract", "write abstract", project_id, "write-abstract", False, False, None, True, task_type="WRITE_ABSTRACT"),
    )
