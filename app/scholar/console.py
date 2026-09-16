"""Read-only projections for the Scholar Web Console.

The Console consumes existing Session/Project/Harness contracts.  This module
does not create workflow state; it maps the persisted Harness trace into a
small event vocabulary suitable for a browser and SSE replay.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from app.scholar.evaluation import HarnessEvaluationCase, ScholarHarnessEvaluationSuite
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.project import ScholarProjectStore
from app.scholar.events import RunEventStore
from app.session import SessionManager


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
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
        "pending": "PENDING",
        "queued": "PENDING",
        "waiting_user": "WAITING_USER",
        "interrupted": "INTERRUPTED",
        "cancelled": "CANCELLED",
        "skipped": "COMPLETED",
    }.get(raw, raw.upper())


RunEventType = Literal[
    "RUN_STARTED",
    "RUN_RESUMED",
    "SKILL_SELECTED",
    "CONTEXT_ASSEMBLED",
    "SUBAGENT_STARTED",
    "SUBAGENT_COMPLETED",
    "TOOL_STARTED",
    "TOOL_COMPLETED",
    "RESEARCH_PROGRESS",
    "RESEARCH_COMPLETED",
    "EVIDENCE_VERIFIED",
    "DOMAIN_RESULT",
    "DRAFT_CREATED",
    "REVIEW_STARTED",
    "REVIEW_COMPLETED",
    "PATCH_CREATED",
    "CHECKPOINT_SAVED",
    "WAITING_USER",
    "RUN_INTERRUPTED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
]
RunEventStatus = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "WAITING_USER",
    "INTERRUPTED",
    "CANCELLED",
]
_RUN_EVENT_TYPES = frozenset(RunEventType.__args__)
_RUN_EVENT_STATUSES = frozenset(RunEventStatus.__args__)


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Stable read-only event contract projected from persisted Harness trace."""

    event_id: str
    cursor: int
    run_id: str
    session_id: str | None
    timestamp: str
    type: RunEventType
    node: str
    status: RunEventStatus
    summary: str
    duration_ms: float | int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.run_id.strip():
            raise ValueError("RunEvent event_id/run_id 不能为空。")
        if self.cursor < 1:
            raise ValueError("RunEvent cursor 必须从 1 开始。")
        if not self.node.strip() or not self.summary.strip():
            raise ValueError("RunEvent node/summary 不能为空。")
        if self.type not in _RUN_EVENT_TYPES:
            raise ValueError(f"RunEvent type 不受支持：{self.type}")
        if self.status not in _RUN_EVENT_STATUSES:
            raise ValueError(f"RunEvent status 不受支持：{self.status}")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _is_verified_evidence(value: Mapping[str, Any]) -> bool:
    """Accept only a positively validated evidence projection.

    Result payloads also contain CitationRequirements and raw discovery
    candidates. Those mappings may carry an ``evidence_id`` but are not
    evidence that the Console is allowed to display as verified.
    """

    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    status = str(value.get("validation_status") or metadata.get("validation_status") or "").casefold()
    if not (
        status == "verified"
        or "verified" in status
        or "validated" in status
    ):
        return False
    content = value.get("content") or value.get("text") or metadata.get("content") or metadata.get("text")
    if not value.get("content_hash") or not content:
        return False
    if hashlib.sha256(str(content).encode("utf-8")).hexdigest() != str(value["content_hash"]):
        return False
    source_type = value.get("source_type") or metadata.get("source_type")
    if str(source_type).casefold() == "web_literature":
        return bool(value.get("canonical_id") and value.get("source_locator"))
    return bool(
        (value.get("paper_id") or metadata.get("paper_id"))
        and (value.get("chunk_id") or value.get("block_id") or value.get("section_id") or metadata.get("chunk_id") or metadata.get("section_id"))
    )


def _event_type(kind: str, name: str, *, task_type: str | None = None) -> RunEventType:
    normalized = name.casefold()
    if normalized in {"flow_transition", "flowstartedevent", "flowfinishedevent"}:
        return "DOMAIN_RESULT"
    if normalized == "crewkickoffstartedevent":
        return "SUBAGENT_STARTED"
    if normalized == "crewkickoffcompletedevent":
        return "SUBAGENT_COMPLETED"
    if normalized == "agentexecutionstartedevent":
        return "SUBAGENT_STARTED"
    if normalized == "agentexecutioncompletedevent":
        return "SUBAGENT_COMPLETED"
    if normalized in {"taskstartedevent", "taskcompletedevent"}:
        return "DOMAIN_RESULT"
    if normalized == "toolusagestartedevent":
        return "TOOL_STARTED"
    if normalized == "toolusagefinishedevent":
        return "TOOL_COMPLETED"
    if normalized == "harness_context":
        return "CONTEXT_ASSEMBLED"
    if normalized == "harness_plan":
        return "SKILL_SELECTED"
    if normalized == "harness_result":
        return "DOMAIN_RESULT"
    if normalized in {"get_project_context", "read_file", "get_patch_status"}:
        return "TOOL_COMPLETED"
    if normalized == "research_evidence":
        return "RESEARCH_COMPLETED"
    if normalized in {"research_need", "research_stage", "research_coverage_item"}:
        return "RESEARCH_PROGRESS"
    if normalized == "review_draft":
        return "REVIEW_COMPLETED"
    if normalized == "execute_scholar_skill":
        return "DOMAIN_RESULT" if task_type == "SUPPORT_CLAIM" else "DRAFT_CREATED"
    if normalized.startswith("web_") or normalized.startswith("evidence_"):
        return "EVIDENCE_VERIFIED" if "validate" in normalized or "verified" in normalized else "RESEARCH_COMPLETED"
    if "checkpoint" in normalized:
        return "CHECKPOINT_SAVED"
    return "TOOL_COMPLETED" if kind == "tool" else "SUBAGENT_COMPLETED"


class ScholarConsoleProjection:
    """Deterministic read model backed by existing runtime stores."""

    def __init__(self, project_root: Path, *, project_store: ScholarProjectStore | None = None, session_manager: SessionManager | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.session_manager = session_manager or SessionManager(self.project_root)
        self.event_store = RunEventStore(self.project_root)

    def _find_run(self, run_id: str) -> tuple[Any, Any, dict[str, Any]]:
        requested = run_id.strip()
        if not requested:
            raise KeyError("Run ID 不能为空。")
        for session in self.session_manager.list(include_deleted=False):
            session_project = getattr(session, "project_id", None)
            if session_project and session_project != self.project_store.project_id:
                continue
            runtime = self.session_manager.open(session.session_id)
            for run in runtime.list_runs():
                if run.run_id != requested:
                    continue
                if run.project_id and run.project_id != self.project_store.project_id:
                    continue
                result = next((item for item in runtime.list_results() if item["run_id"] == requested), {})
                return run, session, result
        raise KeyError(f"Run 不存在：{requested}")

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
        # CrewAI framework events are projected through the existing trace
        # field.  Keep the Console vocabulary and replay endpoint unchanged.
        if metadata.get("backend") == "crewai" and not harness.get("trace"):
            crew_trace = metadata.get("trace")
            if isinstance(crew_trace, Mapping):
                harness = {**harness, **dict(crew_trace)}
        for key in ("visible_capabilities", "visible_tools", "unexpected_tool_calls", "capability_violations", "context_budget"):
            if key in metadata:
                harness[key] = _jsonable(metadata[key])
        if "visible_tools" not in harness and "visible_capabilities" in harness:
            # CrewAI names this projection ``visible_capabilities`` while the
            # existing evaluator/console contract calls it ``visible_tools``.
            harness["visible_tools"] = harness["visible_capabilities"]
        value: Any = None
        structured_result = metadata.get("structured_result")
        if structured_result is not None:
            value = structured_result
        answer = result.get("answer")
        if value is None and isinstance(answer, str) and answer:
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
            "orchestration_backend": metadata.get("backend") or "legacy",
        }

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        persisted = list(self.event_store.list(run_id))
        if persisted:
            return persisted
        snapshot = self.run_snapshot(run_id)
        run = snapshot["run"]
        routing = snapshot["routing"]
        events: list[dict[str, Any]] = []
        sequence = 0

        def emit(
            event_type: RunEventType,
            *,
            node: str,
            status: str,
            summary: str,
            timestamp: str | None = None,
            duration_ms: float | int | None = None,
            metadata: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            allowed_statuses = {"PENDING", "RUNNING", "COMPLETED", "FAILED", "WAITING_USER", "INTERRUPTED", "CANCELLED"}
            normalized_status = status if status in allowed_statuses else "FAILED"
            events.append(
                RunEvent(
                    event_id=f"{run_id}:{sequence}",
                    cursor=sequence,
                    run_id=run_id,
                    session_id=run.get("session_id"),
                    timestamp=timestamp or run.get("started_at") or run.get("created_at") or _now(),
                    type=event_type,
                    node=node,
                    status=normalized_status,  # type: ignore[arg-type]
                    summary=summary,
                    duration_ms=duration_ms,
                    metadata=_jsonable(dict(metadata or {})),
                ).to_dict()
            )

        run_status = _status(run.get("status"))
        start_status = {
            "RUNNING": "RUNNING",
            "PENDING": "PENDING",
        }.get(run_status, "COMPLETED")
        emit(
            "RUN_STARTED",
            node="Supervisor",
            status=start_status,
            summary="Scholar Run started",
            timestamp=run.get("created_at"),
            metadata={"selected_skill": routing.get("selected_skill")},
        )
        if routing.get("resumed"):
            emit(
                "RUN_RESUMED",
                node="Supervisor",
                status="COMPLETED",
                summary="Scholar Run resumed from persistent checkpoint",
                timestamp=run.get("started_at"),
                metadata={"thread_id": run.get("thread_id")},
            )

        harness = snapshot.get("harness", {})
        trace = harness.get("trace", []) if isinstance(harness, Mapping) else []
        task_type = str(routing.get("task_type") or "")
        for item in trace if isinstance(trace, list) else []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("stage") or "step")
            normalized_name = name.casefold()
            kind = str(item.get("kind") or "step")
            event_status = _status(item.get("status"))
            event_type = _event_type(kind, name, task_type=task_type)
            if event_status == "RUNNING":
                event_type = "REVIEW_STARTED" if normalized_name == "review_draft" else "TOOL_STARTED" if kind == "tool" else "SUBAGENT_STARTED"
            if normalized_name == "harness_plan":
                node = str(routing.get("selected_skill") or "Skill")
            elif normalized_name == "harness_context":
                node = "Supervisor"
            elif normalized_name == "execute_scholar_skill":
                node = "Research Capability" if task_type == "SUPPORT_CLAIM" else "Writing Runtime"
            elif normalized_name == "research_evidence":
                node = "Research Subagent"
            elif normalized_name == "review_draft":
                node = "Reviewer"
            elif normalized_name in {"agentexecutionstartedevent", "agentexecutioncompletedevent"}:
                event_metadata = item.get("metadata")
                role = event_metadata.get("agent_role") if isinstance(event_metadata, Mapping) else None
                node = str(role or "Specialist Agent")
            elif normalized_name in {"toolusagestartedevent", "toolusagefinishedevent"}:
                event_metadata = item.get("metadata")
                tool_name = event_metadata.get("tool_name") if isinstance(event_metadata, Mapping) else None
                node = f"Research Tool: {tool_name}" if tool_name else "Capability Tool"
            elif normalized_name == "flow_transition":
                event_metadata = item.get("metadata")
                state_name = event_metadata.get("state") if isinstance(event_metadata, Mapping) else None
                node = f"CrewAI Flow: {state_name}" if state_name else "CrewAI Flow"
            elif normalized_name == "harness_result":
                node = "Research Capability" if task_type == "SUPPORT_CLAIM" else "Writing Runtime"
            elif normalized_name.startswith(("web_", "evidence_")):
                node = "Evidence Validation" if "valid" in normalized_name or "verif" in normalized_name else "Research Subagent"
            else:
                node = name.replace("_", " ").title()
            details = dict(item.get("details") or {}) if isinstance(item.get("details"), Mapping) else {}
            details.update({"trace_name": name, "kind": kind})
            emit(
                event_type,
                node=node,
                status=event_status,
                summary=name.replace("_", " ").title(),
                timestamp=run.get("started_at"),
                duration_ms=item.get("elapsed_ms"),
                metadata=details,
            )

        result = snapshot.get("result", {})
        value = result.get("value") if isinstance(result, Mapping) else None
        patch = value.get("patch") if isinstance(value, Mapping) else None
        if isinstance(patch, Mapping):
            patch_status = "AWAITING_APPROVAL"
            try:
                stored_patch = self.project_store.get_patch(str(patch.get("patch_id") or ""))
                patch_status = str(getattr(stored_patch, "status", patch_status)).upper()
            except (KeyError, ValueError):
                pass
            approval_status, approval_summary = {
                "APPLIED": ("COMPLETED", "DraftPatch applied by human approval"),
                "REJECTED": ("COMPLETED", "DraftPatch rejected by human approval"),
                "CONFLICT": ("FAILED", "DraftPatch approval encountered a conflict"),
                "FAILED": ("FAILED", "DraftPatch apply failed"),
                "APPROVED": ("RUNNING", "DraftPatch approved; applying changes"),
                "APPLYING": ("RUNNING", "DraftPatch is being applied"),
            }.get(patch_status, ("WAITING_USER", "DraftPatch awaits human approval"))
            emit(
                "WAITING_USER" if approval_status == "WAITING_USER" else "PATCH_CREATED",
                node="Human Approval",
                status=approval_status,
                summary=approval_summary,
                timestamp=run.get("completed_at"),
                metadata={"patch_id": patch.get("patch_id"), "patch_status": patch_status},
            )

        terminal = {
            "INTERRUPTED": ("RUN_INTERRUPTED", "INTERRUPTED"),
            "FAILED": ("RUN_FAILED", "FAILED"),
            "CANCELLED": ("RUN_CANCELLED", "CANCELLED"),
            "WAITING_USER": ("WAITING_USER", "WAITING_USER"),
            "COMPLETED": ("RUN_COMPLETED", "COMPLETED"),
        }.get(str(run.get("status") or ""))
        if terminal is not None:
            event_type, terminal_status = terminal
            emit(
                event_type,  # type: ignore[arg-type]
                node="Scholar Run",
                status=terminal_status,
                summary=str(snapshot.get("termination_reason") or event_type),
                timestamp=run.get("completed_at"),
                metadata={"termination_reason": snapshot.get("termination_reason")},
            )
        return events

    def project_state(self, project_id: str) -> dict[str, Any]:
        if project_id != self.project_store.project_id:
            raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
        synchronizer = ManuscriptSynchronizer(self.project_root)
        try:
            state = synchronizer.scan(previous=self.project_store.load_manuscript_state())
        except FileNotFoundError:
            # A research-only workspace may legitimately have no LaTeX root
            # yet.  Keep the project control plane usable so SUPPORT_CLAIM
            # runs can still be inspected; writing tasks will report their own
            # manuscript-context requirement when they need one.
            patches = self.project_store.list_patches()
            return {
                "project_id": project_id,
                "root_tex": None,
                "project_hash": None,
                "version": 0,
                "stale_sections": [],
                "sections": [],
                "facts": _jsonable(self.project_store.list_facts()),
                "contributions": _jsonable(self.project_store.list_contributions()),
                "patches": _jsonable(patches),
                "manuscript_available": False,
                "message": "当前项目还没有 root .tex 文件。",
            }
        patches = self.project_store.list_patches()
        patch_by_section: dict[str, Any] = {}
        for stored in patches:
            patch = getattr(stored, "patch", None)
            if patch is not None:
                patch_by_section[patch.target_section] = stored
        sections: list[dict[str, Any]] = []
        for name, section in sorted(state.sections.items()):
            stored = patch_by_section.get(name)
            dependencies = tuple(
                sorted(
                    source
                    for source, dependents in ManuscriptSynchronizer.DEPENDENTS.items()
                    if name.casefold() in {value.casefold() for value in dependents}
                )
            )
            review_report = getattr(stored, "review_report", None)
            patch_status = str(getattr(stored, "status", "")) if stored is not None else None
            review_status = (
                "NONE"
                if stored is None
                else "PASSED"
                if review_report is not None and review_report.valid
                else "NEEDS_USER_REVIEW"
                if review_report is not None
                else patch_status or "UNKNOWN"
            )
            if section.stale and dependencies:
                dependency_reason = f"upstream section changed: {', '.join(dependencies)}"
            elif section.stale:
                dependency_reason = "section changed since the previous Project State"
            else:
                dependency_reason = None
            item = _jsonable(section)
            item.update(
                {
                    "dependencies": list(dependencies),
                    "dependency_reason": dependency_reason,
                    "last_patch": _jsonable(stored),
                    "patch_status": patch_status,
                    "review_status": review_status,
                }
            )
            sections.append(item)
        return {
            "project_id": project_id,
            "root_tex": state.root_tex,
            "project_hash": state.project_hash,
            "version": state.version,
            "stale_sections": list(state.stale_sections),
            "sections": sections,
            "facts": _jsonable(self.project_store.list_facts()),
            "contributions": _jsonable(self.project_store.list_contributions()),
            "patches": _jsonable(patches),
        }

    def manuscript_view(self, project_id: str) -> dict[str, Any]:
        """Return a safe, read-only manuscript projection for the web console.

        The persisted project state intentionally contains hashes and metadata
        only.  The browser needs the actual section text to make LaTeX review
        useful, so this endpoint reads only files reachable from the root TeX
        file and always goes through ``ManuscriptSynchronizer._safe_path``.
        It never exposes an absolute filesystem path and never mutates files.
        """

        if project_id != self.project_store.project_id:
            raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
        synchronizer = ManuscriptSynchronizer(self.project_root)
        projection = self.project_state(project_id)
        try:
            state = synchronizer.scan(previous=self.project_store.load_manuscript_state())
        except FileNotFoundError:
            return {
                "project_id": project_id,
                "root_tex": None,
                "project_hash": None,
                "version": 0,
                "stale_sections": [],
                "sections": [],
                "latest_build": None,
                "pdf_available": False,
                "pdf_path": None,
                "pdf_url": None,
                "facts": projection.get("facts", []),
                "contributions": projection.get("contributions", []),
                "patches": projection.get("patches", []),
                "manuscript_available": False,
                "message": "当前项目还没有 root .tex 文件。可以先在项目中创建 main.tex。",
            }
        section_by_name = {
            str(item["name"]): item
            for item in projection.get("sections", [])
            if isinstance(item, Mapping) and item.get("name") is not None
        }
        sections: list[dict[str, Any]] = []
        for name, section in sorted(state.sections.items()):
            item = dict(section_by_name.get(name, _jsonable(section)))
            try:
                content = synchronizer.read_section(state, name)
                read_error = None
            except (OSError, UnicodeError, ValueError, KeyError) as error:
                content = ""
                read_error = str(error)
            item["content"] = content
            item["character_count"] = len(content)
            item["line_count"] = len(content.splitlines())
            if read_error:
                item["read_error"] = read_error
            sections.append(item)

        root_path = synchronizer._safe_path(state.root_tex)
        pdf_path = root_path.with_suffix(".pdf")
        pdf_available = pdf_path.is_file()
        latest_build = self.project_store.latest_build_result(project_id)
        return {
            "project_id": project_id,
            "root_tex": state.root_tex,
            "project_hash": state.project_hash,
            "version": state.version,
            "stale_sections": list(state.stale_sections),
            "sections": sections,
            "latest_build": _jsonable(latest_build) if latest_build is not None else None,
            "pdf_available": pdf_available,
            "pdf_path": state.root_tex.rsplit("/", 1)[-1].replace(".tex", ".pdf") if pdf_available else None,
            "pdf_url": f"/api/scholar/projects/{project_id}/manuscript/pdf" if pdf_available else None,
            "manuscript_available": True,
            "facts": projection.get("facts", []),
            "contributions": projection.get("contributions", []),
            "patches": projection.get("patches", []),
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
        patches_by_evidence: dict[str, list[dict[str, Any]]] = {}
        for stored in self.project_store.list_patches():
            patch = getattr(stored, "patch", None)
            if patch is None:
                continue
            patch_ref = {
                "patch_id": patch.patch_id,
                "target_section": patch.target_section,
                "status": str(getattr(stored, "status", "UNKNOWN")),
            }
            for evidence_id in patch.used_evidence_ids:
                patches_by_evidence.setdefault(str(evidence_id), []).append(patch_ref)

        verified: list[dict[str, Any]] = []
        claims: dict[str, dict[str, Any]] = {}
        for row in self.project_store.list_external_evidence_projections():
            item = dict(row)
            metadata = item.get("metadata_json")
            if isinstance(metadata, str):
                try:
                    item["metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    item["metadata"] = {}
                item.pop("metadata_json", None)
            item.setdefault("source_type", "WEB_LITERATURE")
            if not _is_verified_evidence(item):
                continue
            verified.append(item)
        citation_requirements: list[Any] = []
        if run_id:
            snapshot = self.run_snapshot(run_id)
            value = snapshot.get("result", {}).get("value")
            seen = {str(item.get("evidence_id")) for item in verified if item.get("evidence_id")}

            def collect_claim(candidate: Any) -> None:
                if isinstance(candidate, Mapping):
                    claim_id = candidate.get("claim_id") or candidate.get("subclaim_id")
                    evidence_ids = candidate.get("evidence_ids")
                    if evidence_ids is None:
                        evidence_ids = (
                            *(candidate.get("supporting_evidence_ids") or ()),
                            *(candidate.get("qualifying_evidence_ids") or ()),
                            *(candidate.get("counter_evidence_ids") or ()),
                        )
                    if claim_id:
                        claims[str(claim_id)] = {
                            "claim_id": str(claim_id),
                            "text": str(candidate.get("text") or candidate.get("claim") or ""),
                            "status": candidate.get("status") or candidate.get("support_status"),
                            "evidence_ids": list(dict.fromkeys(str(item) for item in evidence_ids or () if item)),
                        }
                    for child in candidate.values():
                        if isinstance(child, (Mapping, list, tuple)):
                            collect_claim(child)
                elif isinstance(candidate, (list, tuple)):
                    for child in candidate:
                        collect_claim(child)

            def collect(candidate: Any, *, evidence_context: bool = False) -> None:
                if isinstance(candidate, Mapping):
                    evidence_id = str(candidate.get("evidence_id") or "")
                    if evidence_context and evidence_id and _is_verified_evidence(candidate) and evidence_id not in seen:
                        item = dict(candidate)
                        verified.append(_jsonable(item))
                        seen.add(evidence_id)
                    for key, child in candidate.items():
                        if key == "citation_requirements":
                            if isinstance(child, (list, tuple)):
                                citation_requirements.extend(_jsonable(value) for value in child)
                            continue
                        if key in {"claims", "subclaims"}:
                            collect_claim(child)
                            continue
                        if key in {"evidence", "supporting_evidence", "counter_evidence", "qualifying_evidence", "verified_evidence"}:
                            collect(child, evidence_context=True)
                        elif key in {"evidence_packs", "value"}:
                            collect(child)
                elif isinstance(candidate, (list, tuple)):
                    for child in candidate:
                        collect(child, evidence_context=evidence_context)

            collect(value)

        enriched: list[dict[str, Any]] = []
        for original in verified:
            item = dict(original)
            evidence_id = str(item.get("evidence_id") or "")
            binding = binding_by_evidence.get(evidence_id)
            if binding is not None:
                item["citation_binding"] = binding
                item["citation_status"] = binding.get("status")
                item["bibkey"] = binding.get("bibkey")
            else:
                item["citation_binding"] = None
                item["citation_status"] = None
                item["bibkey"] = None
            item["claim_ids"] = [
                claim_id
                for claim_id, claim in claims.items()
                if evidence_id in claim.get("evidence_ids", [])
            ]
            item["claims"] = [claims[claim_id] for claim_id in item["claim_ids"]]
            item["patch_references"] = patches_by_evidence.get(evidence_id, [])
            item["evidence_span"] = item.get("content") or item.get("text") or (
                item.get("metadata", {}).get("content")
                if isinstance(item.get("metadata"), Mapping)
                else None
            )
            enriched.append(_jsonable(item))
        enriched.sort(key=lambda item: str(item.get("evidence_id") or ""))
        citation_requirements = list({
            str(item.get("evidence_id") or item.get("identity_key") or index): item
            for index, item in enumerate(citation_requirements)
            if isinstance(item, Mapping)
        }.values())
        return {
            "project_id": project_id,
            "run_id": run_id,
            "claims": list(claims.values()),
            "verified_evidence": enriched,
            "citation_bindings": bindings,
            "citation_requirements": citation_requirements,
            "note": "仅展示已验证或已持久化审计投影；Discovery Candidate 不进入此视图。",
        }

    def evaluation(self, run_id: str) -> dict[str, Any]:
        snapshot = self.run_snapshot(run_id)
        raw_task_type = snapshot.get("routing", {}).get("task_type")
        task_type = str(raw_task_type or "").strip().upper()
        expected = {
            "SUPPORT_CLAIM": ("support-claim", True, False, "ClaimSupportResult", False),
            "WRITE_INTRODUCTION": ("write-introduction", None, True, "WritingResult", True),
            "WRITE_CONCLUSION": ("write-conclusion", False, False, "WritingResult", True),
            "WRITE_ABSTRACT": ("write-abstract", False, False, "WritingResult", True),
        }.get(task_type)
        if expected is None:
            if task_type:
                return {
                    "run_id": run_id,
                    "task_type": task_type,
                    "supported": False,
                    "reason_code": "EVALUATION_UNSUPPORTED_TASK",
                    "reason": f"任务类型 {task_type} 当前没有对应的质量评估合同。",
                    "metrics": {},
                    "record": None,
                }
            return {
                "run_id": run_id,
                "task_type": None,
                "supported": False,
                "reason_code": "EVALUATION_NOT_ROUTED",
                "reason": "任务尚未完成路由，暂时没有可计算的质量评估。",
                "metrics": {},
                "record": None,
            }
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
            "task_type": task_type,
            "supported": True,
            "metrics": {
                "Task Routing Accuracy": 1.0
                if str(record.selected_skill or "").casefold().replace("_", "-")
                == str(case.expected_skill).casefold().replace("_", "-")
                else 0.0,
                "Forbidden Tool Call Count": record.forbidden_tool_calls,
                "Unexpected Research Rate": int("UNEXPECTED_RESEARCH" in record.failures),
                "Required Research Miss Rate": int("REQUIRED_RESEARCH_MISSED" in record.failures),
                "Domain Result Validity": int(record.domain_result_valid),
                "Context Isolation Violation Count": record.context_isolation_violations,
                "Resume Success Rate": (
                    int(record.resumed)
                    if case.resume_expected is True
                    else 1.0
                ),
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
            {"evidence_id": "EV_LOCAL_01", "claim_ids": ["C1"], "title": "Ephemeris error compensation for LEO navigation", "authors": ["Khalife et al."], "source_type": "LOCAL_CORPUS", "paper_id": "PAPER_LOCAL_01", "locator": "results · chunk C_019", "source_locator": "results · chunk C_019", "publication_date": "2024-01-01", "provider": "local corpus", "evidence_span": "Ephemeris error compensation improves the local positioning estimate.", "validation_status": "VERIFIED", "citation_binding": {"status": "RESOLVED_EXISTING", "bibkey": "Khalife2024"}, "bibkey": "Khalife2024"},
            {"evidence_id": "EV_WEB_02", "claim_ids": ["C1"], "title": "Robust signals-of-opportunity positioning", "authors": ["Smith et al."], "source_type": "WEB_LITERATURE", "canonical_id": "doi:10.1234/demo", "locator": "ABSTRACT · DOI 10.1234/demo", "source_locator": "https://doi.org/10.1234/demo#abstract", "publication_date": "2023-06-01", "retrieved_at": "2026-09-13T00:00:00+00:00", "provider": "Crossref", "evidence_span": "Signals of opportunity provide a robust positioning reference.", "validation_status": "VERIFIED", "citation_binding": {"status": "RESOLVED_EXISTING", "bibkey": "Smith2023SOO"}, "bibkey": "Smith2023SOO"},
        ],
        "sections": [
            {"name": "introduction", "relative_path": "sections/introduction.tex", "content_hash": "demo-intro-hash", "version": 3, "stale": False, "status": "CURRENT", "dependencies": [], "review_status": "PASSED", "patch_status": "AWAITING_APPROVAL"},
            {"name": "method", "relative_path": "sections/method.tex", "content_hash": "demo-method-hash", "version": 2, "stale": False, "status": "CURRENT", "dependencies": [], "review_status": "NONE", "patch_status": None},
            {"name": "results", "relative_path": "sections/results.tex", "content_hash": "demo-results-hash", "version": 2, "stale": False, "status": "CURRENT", "dependencies": [], "review_status": "NONE", "patch_status": None},
            {"name": "conclusion", "relative_path": "sections/conclusion.tex", "content_hash": "demo-conclusion-hash", "version": 1, "stale": True, "status": "STALE", "dependencies": ["results"], "dependency_reason": "upstream section changed: results", "review_status": "NONE", "patch_status": None},
            {"name": "abstract", "relative_path": "sections/abstract.tex", "content_hash": "demo-abstract-hash", "version": 1, "stale": True, "status": "STALE", "dependencies": ["conclusion", "experiment", "method", "results"], "dependency_reason": "upstream section changed: conclusion, experiment, method, results", "review_status": "NONE", "patch_status": None},
        ],
        "evaluation": {"metrics": {"Task Routing Accuracy": 1.0, "Forbidden Tool Call Count": 0, "Unexpected Research Rate": 0.0, "Required Research Miss Rate": 0.0, "Domain Result Validity": 1.0, "Context Isolation Violation Count": 0, "Resume Success Rate": 1.0, "Capability Violation Count": 0}},
    }
