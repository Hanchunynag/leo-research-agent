"""Read-only projections for the Scholar Web Console.

The Console consumes existing Session/Project/Harness contracts.  This module
does not create workflow state; it maps the persisted Harness trace into a
small event vocabulary suitable for a browser and SSE replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from app.scholar.evaluation import HarnessEvaluationCase, ScholarHarnessEvaluationSuite
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.project import ScholarProjectStore
from app.session import SessionManager


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _status(value: Any) -> str:
    raw = str(value or "succeeded").casefold()
    return {
        "succeeded": "COMPLETED",
        "success": "COMPLETED",
        "failed": "FAILED",
        "running": "RUNNING",
        "waiting_user": "WAITING_USER",
        "interrupted": "INTERRUPTED",
    }.get(raw, raw.upper())


def _is_verified_evidence(value: Mapping[str, Any]) -> bool:
    """Accept only a positively validated evidence projection.

    Result payloads also contain CitationRequirements and raw discovery
    candidates. Those mappings may carry an ``evidence_id`` but are not
    evidence that the Console is allowed to display as verified.
    """

    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    status = str(value.get("validation_status") or metadata.get("validation_status") or "").casefold()
    if status in {"", "candidate", "unverified", "rejected", "invalid", "failed"}:
        return False
    content = value.get("content") or value.get("text") or metadata.get("content") or metadata.get("text")
    if not value.get("content_hash") or not content:
        return False
    source_type = value.get("source_type") or metadata.get("source_type")
    if source_type == "WEB_LITERATURE":
        return bool(value.get("canonical_id") and value.get("source_locator"))
    return bool(
        (value.get("paper_id") or metadata.get("paper_id"))
        and (value.get("chunk_id") or value.get("block_id") or value.get("section_id") or metadata.get("chunk_id") or metadata.get("section_id"))
    )


def _event_type(kind: str, name: str, *, terminal: bool = False, resumed: bool = False) -> str:
    if terminal:
        return "RUN_COMPLETED"
    if resumed:
        return "RUN_RESUMED"
    normalized = name.casefold()
    if normalized in {"get_project_context", "read_file"}:
        return "TOOL_COMPLETED"
    if normalized == "research_evidence":
        return "RESEARCH_COMPLETED"
    if normalized == "review_draft":
        return "REVIEW_COMPLETED"
    if normalized == "execute_scholar_skill":
        return "PATCH_CREATED" if kind == "tool" else "DRAFT_CREATED"
    if normalized.startswith("web_") or normalized.startswith("evidence_"):
        return "EVIDENCE_VERIFIED" if "validate" in normalized or "verified" in normalized else "RESEARCH_COMPLETED"
    if normalized in {"harness_context", "harness_plan"}:
        return "SKILL_SELECTED" if normalized == "harness_plan" else "RUN_STARTED"
    if "checkpoint" in normalized:
        return "CHECKPOINT_SAVED"
    return "TOOL_COMPLETED" if kind == "tool" else "SUBAGENT_COMPLETED"


class ScholarConsoleProjection:
    """Deterministic read model backed by existing runtime stores."""

    def __init__(self, project_root: Path, *, project_store: ScholarProjectStore | None = None, session_manager: SessionManager | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.session_manager = session_manager or SessionManager(self.project_root)

    def _find_run(self, run_id: str) -> tuple[Any, Any, dict[str, Any]]:
        for session in self.session_manager.list(include_deleted=False):
            runtime = self.session_manager.open(session.session_id)
            for run in runtime.list_runs():
                if run.run_id != run_id:
                    continue
                result = next((item for item in runtime.list_results() if item["run_id"] == run_id), {})
                return run, session, result
        raise KeyError(f"Run 不存在：{run_id}")

    @staticmethod
    def _metadata(result: Mapping[str, Any]) -> dict[str, Any]:
        metadata = result.get("metadata")
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    def run_snapshot(self, run_id: str) -> dict[str, Any]:
        run, session, result = self._find_run(run_id)
        metadata = self._metadata(result)
        harness = metadata.get("harness")
        if not isinstance(harness, Mapping):
            harness = {}
        harness = dict(harness)
        for key in ("visible_capabilities", "visible_tools", "unexpected_tool_calls", "capability_violations", "context_budget"):
            if key in metadata:
                harness[key] = _jsonable(metadata[key])
        value: Any = None
        answer = result.get("answer")
        if isinstance(answer, str) and answer:
            try:
                value = json.loads(answer)
            except json.JSONDecodeError:
                value = None
        return {
            "run": _jsonable(run),
            "session": _jsonable(session),
            "result": {
                "status": result.get("status", run.status),
                "result_type": metadata.get("result_type"),
                "value": value,
                "error": metadata.get("error") or run.failure_message,
            },
            "routing": {
                "task_type": metadata.get("task_type"),
                "selected_skill": metadata.get("selected_skill"),
                "resumed": bool(metadata.get("resumed")),
            },
            "harness": _jsonable(harness),
            "termination_reason": metadata.get("termination_reason") or harness.get("termination_reason"),
        }

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        snapshot = self.run_snapshot(run_id)
        run = snapshot["run"]
        metadata = snapshot["routing"]
        events: list[dict[str, Any]] = [{
            "event_id": f"{run_id}:0",
            "run_id": run_id,
            "session_id": run.get("session_id"),
            "timestamp": run.get("created_at") or _now(),
            "type": "RUN_STARTED",
            "node": "Supervisor",
            "status": "COMPLETED" if run.get("status") != "RUNNING" else "RUNNING",
            "summary": "Scholar Run started",
            "metadata": {"selected_skill": metadata.get("selected_skill")},
        }]
        if metadata.get("resumed"):
            events.append({
                "event_id": f"{run_id}:resumed",
                "run_id": run_id,
                "session_id": run.get("session_id"),
                "timestamp": run.get("started_at") or _now(),
                "type": "RUN_RESUMED",
                "node": "Supervisor",
                "status": "COMPLETED",
                "summary": "Scholar Run resumed from persistent checkpoint",
                "metadata": {"thread_id": run.get("thread_id")},
            })
        harness = snapshot.get("harness", {})
        trace = harness.get("trace", []) if isinstance(harness, Mapping) else []
        for ordinal, item in enumerate(trace if isinstance(trace, list) else [], 1):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("stage") or "step")
            kind = str(item.get("kind") or "step")
            event_type = _event_type(kind, name)
            node = {
                "get_project_context": "Supervisor",
                "read_file": "Skill",
                "research_evidence": "Research Subagent",
                "review_draft": "Reviewer",
                "execute_scholar_skill": "Writing Runtime",
            }.get(name, "Evidence / Runtime" if name.upper().startswith(("WEB_", "EVIDENCE_")) else name.replace("_", " ").title())
            events.append({
                "event_id": f"{run_id}:{ordinal}",
                "run_id": run_id,
                "session_id": run.get("session_id"),
                "timestamp": run.get("started_at") or _now(),
                "type": event_type,
                "node": node,
                "status": _status(item.get("status")),
                "summary": name.replace("_", " ").title(),
                "duration_ms": item.get("elapsed_ms"),
                "metadata": _jsonable(item.get("details") or {}),
            })
        result = snapshot.get("result", {})
        value = result.get("value") if isinstance(result, Mapping) else None
        patch = value.get("patch") if isinstance(value, Mapping) else None
        if isinstance(patch, Mapping):
            patch_status = "AWAITING_APPROVAL"
            try:
                stored_patch = self.project_store.get_patch(str(patch.get("patch_id") or ""))
                patch_status = str(getattr(stored_patch, "status", patch_status))
            except (KeyError, ValueError):
                pass
            approval_status, approval_summary = {
                "APPLIED": ("COMPLETED", "DraftPatch applied by human approval"),
                "REJECTED": ("COMPLETED", "DraftPatch rejected by human approval"),
                "CONFLICT": ("FAILED", "DraftPatch approval encountered a conflict"),
                "FAILED": ("FAILED", "DraftPatch apply failed"),
            }.get(patch_status, ("WAITING_USER", "DraftPatch awaits human approval"))
            events.append({
                "event_id": f"{run_id}:patch",
                "run_id": run_id,
                "session_id": run.get("session_id"),
                "timestamp": run.get("completed_at") or _now(),
                "type": "WAITING_USER" if approval_status == "WAITING_USER" else "PATCH_CREATED",
                "node": "Human Approval",
                "status": approval_status,
                "summary": approval_summary,
                "metadata": {"patch_id": patch.get("patch_id"), "patch_status": patch_status},
            })
        terminal = "RUN_INTERRUPTED" if run.get("status") == "INTERRUPTED" else "RUN_FAILED" if run.get("status") == "FAILED" else "RUN_COMPLETED"
        events.append({
            "event_id": f"{run_id}:terminal",
            "run_id": run_id,
            "session_id": run.get("session_id"),
            "timestamp": run.get("completed_at") or _now(),
            "type": terminal,
            "node": "Scholar Run",
            "status": _status(run.get("status")),
            "summary": snapshot.get("termination_reason") or terminal,
            "metadata": {"termination_reason": snapshot.get("termination_reason")},
        })
        return events

    def project_state(self, project_id: str) -> dict[str, Any]:
        if project_id != self.project_store.project_id:
            raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
        synchronizer = ManuscriptSynchronizer(self.project_root)
        state = synchronizer.scan(previous=self.project_store.load_manuscript_state())
        patches = self.project_store.list_patches()
        patch_by_section: dict[str, Any] = {}
        for stored in patches:
            patch = getattr(stored, "patch", None)
            if patch is not None:
                patch_by_section[patch.target_section] = stored
        return {
            "project_id": project_id,
            "root_tex": state.root_tex,
            "project_hash": state.project_hash,
            "version": state.version,
            "stale_sections": list(state.stale_sections),
            "sections": [
                {
                    **_jsonable(section),
                    "last_patch": _jsonable(patch_by_section.get(name)),
                }
                for name, section in sorted(state.sections.items())
            ],
            "facts": _jsonable(self.project_store.list_facts()),
            "contributions": _jsonable(self.project_store.list_contributions()),
            "patches": _jsonable(patches),
        }

    def evidence_view(self, project_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        if project_id != self.project_store.project_id:
            raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
        bindings = [_jsonable(value) for value in self.project_store.list_citation_bindings()]
        binding_by_evidence: dict[str, dict[str, Any]] = {}
        for binding in bindings:
            evidence_ids = binding.get("evidence_ids", ()) if isinstance(binding, Mapping) else ()
            for evidence_id in evidence_ids:
                binding_by_evidence[str(evidence_id)] = binding
        external = []
        for row in self.project_store.list_external_evidence_projections():
            item = dict(row)
            metadata = item.get("metadata_json")
            if isinstance(metadata, str):
                try:
                    item["metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    item["metadata"] = {}
                item.pop("metadata_json", None)
            if not _is_verified_evidence(item):
                continue
            binding = binding_by_evidence.get(str(item.get("evidence_id")))
            if binding is not None:
                item["citation_binding"] = binding
                item["bibkey"] = binding.get("bibkey")
            external.append(item)
        citation_requirements: list[Any] = []
        if run_id:
            snapshot = self.run_snapshot(run_id)
            value = snapshot.get("result", {}).get("value")
            seen = {str(item.get("evidence_id")) for item in external if item.get("evidence_id")}

            def collect(candidate: Any, *, evidence_context: bool = False) -> None:
                if isinstance(candidate, Mapping):
                    if evidence_context and candidate.get("evidence_id") and _is_verified_evidence(candidate) and candidate.get("evidence_id") not in seen:
                        item = dict(candidate)
                        binding = binding_by_evidence.get(str(item.get("evidence_id")))
                        if binding is not None:
                            item["citation_binding"] = binding
                            item["bibkey"] = binding.get("bibkey")
                        external.append(_jsonable(item))
                        seen.add(str(candidate["evidence_id"]))
                    for key, child in candidate.items():
                        if key == "citation_requirements":
                            if isinstance(child, (list, tuple)):
                                citation_requirements.extend(_jsonable(value) for value in child)
                            continue
                        if key in {"evidence", "supporting_evidence", "counter_evidence", "qualifying_evidence", "verified_evidence"}:
                            collect(child, evidence_context=True)
                        elif key in {"evidence_packs", "value"}:
                            collect(child)
                elif isinstance(candidate, (list, tuple)):
                    for child in candidate:
                        collect(child, evidence_context=evidence_context)

            collect(value)
        return {
            "project_id": project_id,
            "run_id": run_id,
            "verified_evidence": external,
            "citation_bindings": bindings,
            "citation_requirements": citation_requirements,
            "note": "仅展示已验证或已持久化审计投影；Discovery Candidate 不进入此视图。",
        }

    def evaluation(self, run_id: str) -> dict[str, Any]:
        snapshot = self.run_snapshot(run_id)
        task_type = str(snapshot.get("routing", {}).get("task_type") or "")
        expected = {
            "SUPPORT_CLAIM": ("support-claim", True, True, "ClaimSupportResult", False),
            "WRITE_INTRODUCTION": ("write-introduction", None, True, "WritingResult", True),
            "WRITE_CONCLUSION": ("write-conclusion", False, False, "WritingResult", True),
            "WRITE_ABSTRACT": ("write-abstract", False, False, "WritingResult", True),
        }.get(task_type)
        if expected is None:
            raise ValueError("EVALUATION_UNSUPPORTED_TASK")
        case = HarnessEvaluationCase(
            run_id,
            str(snapshot["run"].get("query") or ""),
            self.project_store.project_id,
            expected[0],
            expected[1],
            expected[2],
            expected[3],
            expected[4],
            task_type=task_type,
        )
        class Result:
            result_type = snapshot["result"].get("result_type")
            status = snapshot["result"].get("status")
            metadata = {
                "selected_skill": snapshot["routing"].get("selected_skill"),
                "resumed": snapshot["routing"].get("resumed", False),
                "visible_tools": snapshot.get("harness", {}).get("visible_tools", {}),
                "unexpected_tool_calls": snapshot.get("harness", {}).get("unexpected_tool_calls", []),
                "trace": snapshot.get("harness", {}),
            }
        record = ScholarHarnessEvaluationSuite().evaluate_result(case, Result())
        return {
            "run_id": run_id,
            "metrics": {
                "Task Routing Accuracy": 1.0 if record.selected_skill == case.expected_skill else 0.0,
                "Forbidden Tool Call Count": record.forbidden_tool_calls,
                "Unexpected Research Rate": int("UNEXPECTED_RESEARCH" in record.failures),
                "Required Research Miss Rate": int("REQUIRED_RESEARCH_MISSED" in record.failures),
                "Domain Result Validity": int(record.domain_result_valid),
                "Context Isolation Violation Count": record.context_isolation_violations,
                "Resume Success Rate": int(record.resumed),
                "Capability Violation Count": record.forbidden_tool_calls + record.context_isolation_violations,
            },
            "record": _jsonable(record),
        }


def demo_console_payload() -> dict[str, Any]:
    """Small fixed payload for UI screenshots; never used by Production."""

    run_id = "DEMO_RUN_INTRODUCTION"
    return {
        "demo": True,
        "snapshot": {
            "run": {"run_id": run_id, "session_id": "DEMO_SESSION", "status": "COMPLETED", "query": "Write an evidence-grounded introduction about LEO positioning.", "thread_id": "DEMO_THREAD"},
            "routing": {"task_type": "WRITE_INTRODUCTION", "selected_skill": "write-introduction", "resumed": False},
            "result": {"status": "READY", "result_type": "WritingResult", "value": {"patch": {"patch_id": "DEMO_PATCH_01", "target_section": "introduction"}}},
            "termination_reason": "NEEDS_USER_REVIEW",
            "harness": {"usage": {"steps": 9, "context_tokens": 2210, "total_tokens": 6840}, "trace": []},
        },
        "events": [
            {"event_id": "DEMO:1", "type": "RUN_STARTED", "node": "Supervisor", "status": "COMPLETED", "summary": "Scholar Run started"},
            {"event_id": "DEMO:2", "type": "SKILL_SELECTED", "node": "write-introduction", "status": "COMPLETED", "summary": "Skill selected"},
            {"event_id": "DEMO:3", "type": "RESEARCH_COMPLETED", "node": "Research Subagent", "status": "COMPLETED", "summary": "Local RAG + Web Literature · 6 verified evidence"},
            {"event_id": "DEMO:4", "type": "EVIDENCE_VERIFIED", "node": "Evidence Validation", "status": "COMPLETED", "summary": "6 Verified Evidence"},
            {"event_id": "DEMO:5", "type": "DRAFT_CREATED", "node": "Writing Runtime", "status": "COMPLETED", "summary": "SectionDraft created"},
            {"event_id": "DEMO:6", "type": "REVIEW_COMPLETED", "node": "Reviewer", "status": "COMPLETED", "summary": "Review passed"},
            {"event_id": "DEMO:7", "type": "PATCH_CREATED", "node": "DraftPatch", "status": "COMPLETED", "summary": "DraftPatch proposed"},
            {"event_id": "DEMO:8", "type": "WAITING_USER", "node": "Human Approval", "status": "WAITING_USER", "summary": "Awaiting human approval"},
        ],
        "evidence": [
            {"evidence_id": "EV_LOCAL_01", "claim": "C1", "title": "Ephemeris error compensation for LEO navigation", "source_type": "LOCAL_CORPUS", "locator": "results · chunk C_019", "validation_status": "VERIFIED", "citation": "Khalife2024"},
            {"evidence_id": "EV_WEB_02", "claim": "C1", "title": "Robust signals-of-opportunity positioning", "source_type": "WEB_LITERATURE", "locator": "ABSTRACT · DOI 10.1234/demo", "validation_status": "VERIFIED", "citation": "Smith2023SOO"},
        ],
        "sections": [
            {"name": "introduction", "relative_path": "sections/introduction.tex", "version": 3, "stale": False, "status": "CURRENT"},
            {"name": "method", "relative_path": "sections/method.tex", "version": 2, "stale": False, "status": "CURRENT"},
            {"name": "results", "relative_path": "sections/results.tex", "version": 2, "stale": False, "status": "CURRENT"},
            {"name": "conclusion", "relative_path": "sections/conclusion.tex", "version": 1, "stale": True, "status": "STALE"},
            {"name": "abstract", "relative_path": "sections/abstract.tex", "version": 1, "stale": True, "status": "STALE"},
        ],
        "evaluation": {"metrics": {"Task Routing Accuracy": 1.0, "Forbidden Tool Call Count": 0, "Unexpected Research Rate": 0.0, "Required Research Miss Rate": 0.0, "Domain Result Validity": 1.0, "Context Isolation Violation Count": 0, "Resume Success Rate": 1.0, "Capability Violation Count": 0}},
    }
