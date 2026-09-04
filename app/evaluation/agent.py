"""LangGraph/Research Harness 轨迹评测，不执行自由形式 Tool Calling。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.0"
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
        trace_status_ok = all(
            not isinstance(value, Mapping) or value.get("status") not in {"failed", "error"}
            for value in trace
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
