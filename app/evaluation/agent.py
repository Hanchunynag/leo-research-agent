"""LangGraph/Research Harness 轨迹评测，不执行自由形式 Tool Calling。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.1"
ALLOWED_TOOLS = frozenset({
    "knowledge.retrieve",
    "workspace.read_scope",
    "workspace.list_documents",
    "literature.resolve_publication_date",
    "literature.search",
    "literature.get_metadata",
    "literature.download",
    "job.get_status",
})


def _required_text(value: Mapping[str, Any], field: str) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} 必须是非空字符串。")
    return raw.strip()


def _string_list(value: Mapping[str, Any], field: str) -> tuple[str, ...]:
    raw = value.get(field, [])
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise ValueError(f"{field} 必须是字符串数组。")
    return tuple(dict.fromkeys(item.strip() for item in raw))


def load_agent_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
        question_id = _required_text(raw, "question_id")
        if question_id in seen:
            raise ValueError(f"{path}:{line_number}: question_id 重复。")
        seen.add(question_id)
        expected_task = _required_text(raw, "expected_task_type")
        expected_tools = _string_list(raw, "required_tools")
        if any(tool not in ALLOWED_TOOLS for tool in expected_tools):
            raise ValueError(f"{path}:{line_number}: required_tools 含未知固定工具。")
        output.append({
            "question_id": question_id,
            "query": _required_text(raw, "query"),
            "expected_task_type": expected_task,
            "expected_retrieval_sufficient": raw.get("expected_retrieval_sufficient"),
            "expected_paper_ids": list(_string_list(raw, "expected_paper_ids")),
            "expected_evidence_ids": list(_string_list(raw, "expected_evidence_ids")),
            "required_tools": list(expected_tools),
            "max_tool_calls": int(raw.get("max_tool_calls", 12)),
            "max_retrieval_rounds": int(raw.get("max_retrieval_rounds", 3)),
        })
    if not output:
        raise ValueError("Agent 评测集不能为空。")
    return output


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} 每行必须是 JSON 对象。")
        values.append(value)
    return values


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _list_ids(value: Any, key: str | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        raw = item.get(key) if key and isinstance(item, Mapping) else item
        if isinstance(raw, str) and raw:
            result.append(raw)
    return result


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key) or 0.0) for row in rows) / max(1, len(rows)), 6)


_TERMINAL_FAILURE_OUTCOMES = frozenset(
    {"generation_failed", "insufficient_evidence", "budget_exhausted", "validation_failed"}
)
_TERMINAL_STATES = frozenset({"completed", "interrupted", "failed", "refused"})
_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"context_preparing"}),
    "context_preparing": frozenset({"planning", "failed", "refused"}),
    "planning": frozenset({"executing", "failed", "refused"}),
    "executing": frozenset({"evaluating", "recovering", "committing", "interrupted", "failed", "refused"}),
    "evaluating": frozenset({"recovering", "committing", "interrupted", "failed", "refused"}),
    "recovering": frozenset({"executing", "committing", "failed", "refused"}),
    "committing": frozenset({"completed", "failed", "refused"}),
    "interrupted": frozenset(),
    "completed": frozenset(),
    "failed": frozenset(),
    "refused": frozenset(),
}


def _trace_contract_validation(
    prediction: Mapping[str, Any],
    harness: Mapping[str, Any],
    trace: Sequence[Any],
) -> tuple[bool, list[str]]:
    """Validate trace order without treating an expected refusal as a bug."""

    issues: list[str] = []
    ordinals = [
        int(value["ordinal"])
        for value in trace
        if isinstance(value, Mapping) and isinstance(value.get("ordinal"), int)
    ]
    if ordinals and ordinals != list(range(1, len(ordinals) + 1)):
        issues.append("trace_ordinals_not_contiguous")

    state_history = harness.get("state_history")
    if isinstance(state_history, list) and state_history:
        states = [str(value) for value in state_history]
        if states[0] != "created":
            issues.append("state_history_must_start_created")
        for current, target in zip(states, states[1:]):
            if target not in _STATE_TRANSITIONS.get(current, frozenset()):
                issues.append(f"illegal_state_transition:{current}->{target}")
        if states[-1] not in _TERMINAL_STATES:
            issues.append("state_history_has_no_terminal_state")

    outcome_code = str((prediction.get("outcome") or {}).get("code") or "")
    failed_positions: list[int] = []
    commit_positions: list[int] = []
    active_tools: set[str] = set()
    approvals: set[str] = set()
    open_actions: set[str] = set()
    for position, value in enumerate(trace):
        if not isinstance(value, Mapping):
            issues.append(f"trace_item_not_mapping:{position + 1}")
            continue
        name = str(value.get("name") or value.get("event_type") or "").upper()
        status = str(value.get("status") or "").casefold()
        kind = str(value.get("kind") or "").casefold()
        details = value.get("details") if isinstance(value.get("details"), Mapping) else {}
        if status in {"failed", "error"}:
            failed_positions.append(position)
        if name == "COMMIT_SAFE_STATE" and status in {"succeeded", "success", "completed"}:
            commit_positions.append(position)

        # ResearchRunHarness.record_tool is an atomic completed observation.
        # Explicit lifecycle events, when supplied by a projection, are
        # checked as paired TOOL_STARTED/TOOL_COMPLETED events.
        is_tool_started = name == "TOOL_STARTED" or name.endswith("_TOOL_STARTED")
        is_tool_completed = name == "TOOL_COMPLETED" or name.endswith("_TOOL_COMPLETED")
        tool_key = str(value.get("tool_name") or details.get("tool_name") or name)
        if is_tool_started:
            active_tools.add(tool_key)
        elif is_tool_completed:
            if tool_key not in active_tools:
                issues.append(f"tool_completed_before_started:{tool_key}")
            else:
                active_tools.remove(tool_key)
        elif kind == "tool" and status in {"succeeded", "success", "completed"}:
            # The internal Research Harness records the tool call atomically;
            # requiring a synthetic start event would falsify its trace.
            pass

        if name in {"APPROVAL_GRANTED", "HUMAN_APPROVAL_GRANTED"}:
            approvals.add(str(details.get("patch_id") or value.get("patch_id") or "*"))
        if name in {"APPLY_COMPLETED", "PATCH_APPLY_COMPLETED"}:
            patch_id = str(details.get("patch_id") or value.get("patch_id") or "*")
            if patch_id not in approvals and "*" not in approvals:
                issues.append(f"apply_without_approval:{patch_id}")
        if name in {"REQUIRED_ACTION_OPENED", "ACTION_REQUIRED"}:
            open_actions.add(str(details.get("action_id") or value.get("action_id") or position))
        if name in {"REQUIRED_ACTION_CLOSED", "ACTION_RESOLVED"}:
            open_actions.discard(str(details.get("action_id") or value.get("action_id") or position))
        if name == "RUN_COMPLETED" and open_actions:
            issues.append("run_completed_with_open_required_action")

    # A failed generation/retrieval path is valid when the harness records the
    # failure, commits the safe state, and exposes a matching terminal outcome.
    if failed_positions:
        if outcome_code not in _TERMINAL_FAILURE_OUTCOMES:
            issues.append("unexpected_failed_trace_without_terminal_failure_outcome")
        if not any(position > failed_positions[-1] for position in commit_positions):
            issues.append("failed_trace_missing_safe_commit")
    if active_tools:
        issues.append("unclosed_tool_lifecycle")

    terminal_state = str(harness.get("state") or "")
    if outcome_code == "answered" and terminal_state != "completed":
        issues.append("answered_outcome_without_completed_state")
    if outcome_code in _TERMINAL_FAILURE_OUTCOMES and terminal_state not in {"refused", "failed", "interrupted"}:
        issues.append("failure_outcome_without_failure_state")
    return not issues, issues


def evaluate_agent_predictions(
    questions: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    prediction_map = {
        str(value.get("question_id")): value
        for value in predictions
        if value.get("question_id")
    }
    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        prediction = prediction_map.get(question_id, {})
        langgraph = _nested(prediction, "diagnostics", "langgraph")
        langgraph = langgraph if isinstance(langgraph, Mapping) else {}
        harness = _nested(prediction, "diagnostics", "harness")
        harness = harness if isinstance(harness, Mapping) else {}
        task_type = str(langgraph.get("task_type") or "")
        planner_ok = float(task_type == str(question["expected_task_type"]))

        candidate_papers = _list_ids(
            prediction.get("candidate_paper_ids")
            or _nested(prediction, "workflow_details", "candidate_paper_ids")
            or langgraph.get("candidate_paper_ids"),
        )
        expected_papers = set(str(value) for value in question.get("expected_paper_ids", []))
        paper_recall = (
            len(set(candidate_papers) & expected_papers) / max(1, len(expected_papers))
            if expected_papers else None
        )
        evidence = prediction.get("selected_evidence")
        evidence = evidence if isinstance(evidence, list) else []
        selected_evidence_ids = _list_ids(evidence, "evidence_id")
        selected_evidence_ids.extend(_list_ids(evidence, "chunk_id"))
        selected_evidence_ids = list(dict.fromkeys(selected_evidence_ids))
        expected_evidence = set(str(value) for value in question.get("expected_evidence_ids", []))
        evidence_recall = (
            len(set(selected_evidence_ids) & expected_evidence) / max(1, len(expected_evidence))
            if expected_evidence else None
        )

        tool_calls = langgraph.get("tool_calls")
        tool_calls = tool_calls if isinstance(tool_calls, list) else []
        tool_names = [
            str(value.get("tool_name"))
            for value in tool_calls
            if isinstance(value, Mapping) and value.get("tool_name")
        ]
        required_tools = [str(value) for value in question.get("required_tools", [])]
        tool_validity = float(all(name in ALLOWED_TOOLS for name in tool_names))
        tool_coverage = float(all(name in tool_names for name in required_tools))
        usage = harness.get("usage") if isinstance(harness.get("usage"), Mapping) else {}
        policy = harness.get("policy") if isinstance(harness.get("policy"), Mapping) else {}
        tool_budget = int(usage.get("tool_calls") or len(tool_names)) <= int(question.get("max_tool_calls", 12))
        round_budget = int(usage.get("retrieval_rounds") or 0) <= int(question.get("max_retrieval_rounds", 3))
        policy_budget = all(
            int(usage.get(name) or 0) <= int(policy.get(limit) or 10**9)
            for name, limit in (
                ("steps", "max_steps"),
                ("llm_calls", "max_llm_calls"),
                ("tool_calls", "max_tool_calls"),
                ("retrieval_rounds", "max_retrieval_rounds"),
            )
        )
        trace = harness.get("trace") if isinstance(harness.get("trace"), list) else []
        trace_status_ok, trace_issues = _trace_contract_validation(
            prediction, harness, trace
        )
        generation_failed = str(
            _nested(prediction, "outcome", "code") or ""
        ) == "generation_failed"
        evidence_preserved_on_failure = float(
            not generation_failed or bool(prediction.get("selected_evidence"))
        )
        citations = prediction.get("citations")
        citations = citations if isinstance(citations, list) else []
        selected_set = set(selected_evidence_ids)
        citation_complete = float(
            all(
                isinstance(value, Mapping)
                and str(value.get("evidence_id") or "") in selected_set
                and value.get("page_start") is not None
                and value.get("page_end") is not None
                for value in citations
            )
        )
        row: dict[str, Any] = {
            "question_id": question_id,
            "planner_task_accuracy": planner_ok,
            "paper_recall": paper_recall,
            "evidence_recall": evidence_recall,
            "tool_validity": tool_validity,
            "required_tool_coverage": tool_coverage,
            "trajectory_budget_compliance": float(tool_budget and round_budget and policy_budget),
            "trajectory_trace_validity": float(trace_status_ok),
            "trajectory_trace_issues": trace_issues,
            "evidence_preserved_on_generation_failure": evidence_preserved_on_failure,
            "citation_completeness": citation_complete,
            "tool_call_count": len(tool_names),
            "retrieval_rounds": int(usage.get("retrieval_rounds") or 0),
            "outcome_code": str(prediction.get("outcome", {}).get("code") or ""),
        }
        expected_sufficient = question.get("expected_retrieval_sufficient")
        if isinstance(expected_sufficient, bool):
            actual_sufficient = bool(prediction.get("selected_evidence"))
            row["retrieval_sufficiency_accuracy"] = float(actual_sufficient == expected_sufficient)
        rows.append(row)

    numeric_names = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"tool_call_count", "retrieval_rounds"}
    })
    report: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluator": "langgraph_research_harness",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "metrics": {name: _mean(rows, name) for name in numeric_names},
        "per_question": rows,
        "trace_contract": {
            "valid_question_count": sum(
                row["trajectory_trace_validity"] == 1.0 for row in rows
            ),
            "failure_cases": [
                {
                    "question_id": row["question_id"],
                    "outcome_code": row["outcome_code"],
                    "issues": row["trajectory_trace_issues"],
                }
                for row in rows
                if row["trajectory_trace_issues"]
            ],
            "known_limitations": [
                "ResearchRunHarness 的 atomic tool trace 记录一次成功调用；只有显式 TOOL_STARTED/TOOL_COMPLETED 事件才执行成对顺序校验。",
                "该评测验证 Research Harness 轨迹，不代替 Scholar Web Console RunEvent/SSE 的端到端验证。",
            ],
        },
        "allowed_tools": sorted(ALLOWED_TOOLS),
    }
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        report["output_path"] = str(resolved)
        write_json_atomic(resolved, report)
    return report


def evaluate_agent_files(
    questions_path: Path,
    predictions_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    return evaluate_agent_predictions(
        load_agent_questions(questions_path),
        load_jsonl(predictions_path),
        output_path=output_path,
    )
