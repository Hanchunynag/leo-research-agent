"""In-process Deep Agents harness for the existing Scholar domain runtime.

Deep Agents owns only user-level routing, planning and context isolation here.
The returned values are still Scholar domain contracts; no Deep Agents state or
message type crosses this module's public boundary.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

from deepagents import HarnessProfile, GeneralPurposeSubagentProfile, register_harness_profile, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.langchain_agent.provider import OpenAICompatibleChatModel, ScholarChatModelAdapter
from app.langchain_agent.checkpoint_factory import checkpointer_has_checkpoint
from app.research.harness import (
    HarnessState,
    ResearchBudgetPolicy,
    ResearchRunHarness,
)
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.context import ScholarContextBudget
from app.scholar.errors import CorrelationConflict, ResumeUnavailable
from app.scholar.models import ReviewReport
from app.scholar.project import ScholarProjectStore
from app.scholar.research import ResearchBudget, ResearchCapabilityService, ResearchRequest
from app.scholar.writing.harness import ScholarSkillRuntime
from app.scholar.writing.models import TaskType, WritingRequest
from app.scholar.writing.runtime import SkillDefinition, SkillRuntimeError
from app.scholar.writing.support import SupportClaimRequest


_SAFE_FS_TOOLS = frozenset({"ls", "write_file", "edit_file", "delete", "glob", "grep", "execute"})
_PROFILE_REGISTERED = False
_NO_RESUME = object()
_FORBIDDEN_AGENT_TOOLS = frozenset(
    {
        "write_file",
        "edit_file",
        "delete",
        "execute",
        "approve_patch",
        "apply_patch",
        "approve",
        "sqlite_query",
        "qdrant",
        "http_request",
    }
)


def _install_safe_harness_profile() -> None:
    """Disable Deep Agents' default general-purpose/filesystem write surface."""

    global _PROFILE_REGISTERED  # noqa: PLW0603
    if _PROFILE_REGISTERED:
        return
    profile = HarnessProfile(
        excluded_tools=_SAFE_FS_TOOLS,
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    # The adapter gives every model a stable provider identity, including test
    # models.  This avoids relying on provider-specific builtin profiles.
    register_harness_profile("scholarchatmodeladapter", profile)
    register_harness_profile("openaicompatiblechatmodel", profile)
    _PROFILE_REGISTERED = True


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _token_estimate(value: Any) -> int:
    return max(1, len(json.dumps(_jsonable(value), ensure_ascii=False, default=str)) // 4)


class DeepAgentsSkillAdapter:
    """Progressive-disclosure adapter over the existing SkillRegistry."""

    def __init__(self, registry: Any, project_root: Path) -> None:
        self.registry = registry
        self.project_root = project_root.expanduser().resolve()

    def metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": definition.name,
                "task_type": definition.task_type,
                "required_context": definition.required_context,
                "capabilities": sorted(definition.capabilities.allowed),
                "skill_path": str(definition.skill_path),
            }
            for definition in self.registry.list()
        )

    def definition(self, task_type: TaskType) -> SkillDefinition:
        return self.registry.get(task_type)

    def load(self, task_type: TaskType) -> str:
        return self.definition(task_type).text

    @property
    def source_root(self) -> Path:
        return self.project_root / "skills"


@dataclass(frozen=True, slots=True)
class ScholarHarnessResult:
    task_type: TaskType | None
    skill_name: str | None
    session_id: str
    run_id: str
    thread_id: str
    trace_id: str
    status: str
    result_type: str | None = None
    value: Any | None = None
    error_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class _SkillInput(BaseModel):
    task_type: str = Field(min_length=1)
    instruction: str = Field(min_length=1)


class _ResearchInput(BaseModel):
    query: str = Field(min_length=1)
    purpose: str = "background"
    target_claim: str | None = None
    freshness_mode: Literal["LOCAL_ONLY", "LOCAL_FIRST", "FRESH_REQUIRED"] = Field(
        default="LOCAL_ONLY",
        description=(
            "Deterministic research mode. Use LOCAL_ONLY unless the selected Skill "
            "explicitly permits external research; use FRESH_REQUIRED only when the "
            "user asks for latest, recent, current, newest, or gives a date range."
        ),
    )
    requested_from: date | None = Field(
        default=None,
        description="Optional inclusive ISO date YYYY-MM-DD. Omit when no date range is requested.",
    )
    requested_to: date | None = Field(
        default=None,
        description="Optional inclusive ISO date YYYY-MM-DD. Omit when no date range is requested.",
    )
    explicit_latest: bool = Field(
        default=False,
        description="Set true only for an explicit latest/recent/current/newest request.",
    )


class _ReviewInput(BaseModel):
    task_type: str = Field(min_length=1)
    draft: str = Field(min_length=1)


class _EmptyInput(BaseModel):
    """Explicit empty schema for read-only no-argument harness tools."""


def _tool(
    func: Callable[..., Any],
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    return_direct: bool = False,
) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
    )


class ScholarHarnessService:
    """User-level in-process Deep Agents entry point.

    ``skill_runtime`` remains the owner of Skill execution.  ``model`` is
    wrapped only at the LangChain boundary; callers receive a
    ``ScholarHarnessResult`` containing the existing ``SkillResult.value``.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        skill_runtime: ScholarSkillRuntime,
        model: BaseChatModel | Any,
        project_store: ScholarProjectStore | None = None,
        research: ResearchCapabilityService | None = None,
        writers: Mapping[str, Any] | None = None,
        reviewer: Callable[[Mapping[str, Any]], Any] | None = None,
        checkpointer: Any | None = None,
        session_manager: Any | None = None,
        interrupt_on: Mapping[str, Any] | None = None,
        max_steps: int | None = None,
        harness_policy: ResearchBudgetPolicy | None = None,
        context_budget: ScholarContextBudget | None = None,
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps 必须大于 0。")
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or skill_runtime.project_store
        self.skill_runtime = skill_runtime
        self.research = research
        self.writers = dict(writers or {})
        self.reviewer = reviewer
        self.checkpointer = checkpointer
        self.session_manager = session_manager
        # This is a generic Harness checkpoint boundary.  It is intentionally
        # independent from DraftPatch approval: no approval/apply tool is ever
        # exposed to the agent.
        self.interrupt_on = dict(interrupt_on or {}) or None
        self.harness_policy = harness_policy or ResearchBudgetPolicy(
            max_steps=max_steps or 24,
            max_llm_calls=16,
            max_tool_calls=16,
            max_retrieval_rounds=4,
            max_external_searches=2,
            max_repairs=2,
            max_context_tokens=16_000,
            max_total_tokens=64_000,
        )
        self.max_steps = self.harness_policy.max_steps
        if max_steps is not None and self.harness_policy.max_steps != max_steps:
            raise ValueError("max_steps 与 harness_policy.max_steps 必须一致。")
        if context_budget is None:
            context_total = self.harness_policy.max_context_tokens
            context_budget = ScholarContextBudget(
                supervisor=min(4_000, context_total),
                research=min(4_000, context_total),
                reviewer=min(8_000, context_total),
                total=context_total,
            )
        self.context_budget = context_budget
        self.worker_id = f"scholar-harness-{secrets.token_hex(4)}"
        self.skill_adapter = DeepAgentsSkillAdapter(skill_runtime.registry, self.project_root)
        self._model = self._adapt_model(model)
        _install_safe_harness_profile()

    @staticmethod
    def _adapt_model(model: Any) -> BaseChatModel:
        if isinstance(model, ScholarChatModelAdapter | OpenAICompatibleChatModel):
            return model
        if isinstance(model, BaseChatModel):
            return ScholarChatModelAdapter(
                inner=model,
                model_name=str(getattr(model, "model_name", type(model).__name__)),
            )
        provider = getattr(model, "provider", model)
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("ScholarHarnessService 需要 BaseChatModel 或 chat_completion Provider。")
        return OpenAICompatibleChatModel(
            provider=provider,
            model_name=str(getattr(provider, "model_name", "scholar-model")),
        )

    def permission_audit(self, task_type: str) -> dict[str, Any]:
        """Startup/test assertion for the effective Harness tool surface."""

        definition = self.skill_adapter.definition(task_type)  # type: ignore[arg-type]
        visible = {
            "main": {"get_project_context", "execute_scholar_skill", "get_patch_status", "read_file"},
            "research": {"research_evidence", "read_file"}
            if definition.capabilities.allows("LOCAL_RESEARCH")
            else set(),
            "reviewer": {"review_draft", "read_file"}
            if definition.task_type != "SUPPORT_CLAIM"
            else set(),
        }
        forbidden = sorted(
            value
            for values in visible.values()
            for value in values
            if value in _FORBIDDEN_AGENT_TOOLS
        )
        if forbidden:
            raise RuntimeError(f"CAPABILITY_VIOLATION: unsafe Harness tools visible: {forbidden}")
        if definition.task_type in {"WRITE_CONCLUSION", "WRITE_ABSTRACT"} and visible["research"]:
            raise RuntimeError("CAPABILITY_VIOLATION: synthesis Skill exposed Research Subagent.")
        return {
            "task_type": definition.task_type,
            "skill": definition.name,
            "visible_tools": {key: sorted(value) for key, value in visible.items()},
            "forbidden_tools": forbidden,
            "approval_visible": False,
        }

    def _project_context(self) -> dict[str, Any]:
        synchronizer = ManuscriptSynchronizer(self.project_root)
        previous = None
        try:
            previous = self.project_store.load_manuscript_state()
        except (KeyError, OSError, ValueError):
            pass
        state = synchronizer.scan(previous=previous)
        return {
            "project_id": self.project_store.project_id,
            "project_hash": state.project_hash,
            "root_tex": state.root_tex,
            "version": state.version,
            "sections": {
                name: {
                    "relative_path": section.relative_path,
                    "content_hash": section.content_hash,
                    "version": section.version,
                    "stale": section.stale,
                }
                for name, section in state.sections.items()
            },
            "stale_sections": state.stale_sections,
            "fact_count": len(self.project_store.list_facts()),
            "confirmed_contribution_ids": [
                item.contribution_id
                for item in self.project_store.list_contributions()
                if item.status == "confirmed" and item.confirmed_by_user
            ],
        }

    def _record_context(self, harness: ResearchRunHarness, audience: str, value: Any) -> int:
        """Record bounded, redacted context metrics without storing its body."""

        estimate = _token_estimate(value)
        current = harness.context_usage.get(audience, 0)
        if current + estimate > self.context_budget.limit_for(audience):
            raise ValueError(
                f"CONTEXT_BUDGET_EXCEEDED: {audience} {current + estimate}>{self.context_budget.limit_for(audience)}"
            )
        harness.record_context(audience, estimate)
        return estimate

    @staticmethod
    def _bounded_reviewer_evidence(value: Any) -> tuple[Mapping[str, Any], ...]:
        """Build the smallest reviewer projection that preserves grounding.

        The deterministic Reviewer consumes evidence identity/locator fields
        and, when configured, a bounded content sample.  Passing every full
        local chunk through the Deep Agents reviewer context made an ordinary
        multi-need Introduction exceed the runtime budget even though the
        Reviewer does not need the entire source text.  Keep all evidence IDs
        so citation/claim membership remains checkable, while bounding source
        text and metadata at this Harness boundary.
        """

        allowed = (
            "evidence_id",
            "candidate_id",
            "request_id",
            "source_type",
            "canonical_id",
            "source_locator",
            "locator_type",
            "publication_date",
            "retrieved_at",
            "provider",
            "validation_status",
            "paper_id",
            "work_id",
            "document_id",
            "section_id",
            "chunk_id",
            "block_ids",
            "content_hash",
            "verification_method",
            "evidence_grade",
            "directness",
        )
        output: list[Mapping[str, Any]] = []
        for item in value if isinstance(value, (list, tuple)) else ():
            if not isinstance(item, Mapping) or not item.get("evidence_id"):
                continue
            bounded = {
                key: item[key]
                for key in allowed
                if key in item and item[key] is not None
            }
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping):
                bounded["metadata"] = {
                    key: metadata[key]
                    for key in (
                        "title",
                        "authors",
                        "venue",
                        "doi",
                        "arxiv_id",
                        "canonical_id",
                        "paper_id",
                        "section_id",
                    )
                    if key in metadata and metadata[key] is not None
                }
            text = item.get("content") or item.get("text")
            if text:
                bounded["content"] = str(text)[:160]
            output.append(bounded)
        return tuple(output)

    def _prepare_session_run(
        self,
        instruction: str,
        project_id: str,
        *,
        session_id: str | None,
        thread_id: str,
        trace_id: str,
        resuming: bool,
    ) -> tuple[str, str, str, Any | None]:
        """Bind the Harness run to SessionRuntime when the production factory supplies it."""

        if self.session_manager is None:
            return session_id or f"SESSION_{secrets.token_hex(8)}", f"RUN_{secrets.token_hex(8)}", trace_id, None
        if resuming:
            if not session_id:
                raise ResumeUnavailable("RESUME_UNAVAILABLE: resume 必须携带 session_id。")
            try:
                record = self.session_manager.get(session_id)
            except (KeyError, ValueError) as error:
                raise ResumeUnavailable("RESUME_UNAVAILABLE: Session 不存在。") from error
            if record.project_id and record.project_id != project_id:
                raise ResumeUnavailable("RESUME_UNAVAILABLE: Session 与 Project 不匹配。")
            runtime = self.session_manager.open(session_id)
            matches = [run for run in runtime.list_runs() if run.thread_id == thread_id]
            if not matches:
                raise ResumeUnavailable("RESUME_UNAVAILABLE: Session 没有对应 thread checkpoint run。")
            stored = matches[-1]
            if stored.project_id and stored.project_id != project_id:
                raise ResumeUnavailable("RESUME_UNAVAILABLE: Run 与 Project 不匹配。")
            if stored.status not in {"INTERRUPTED", "RUNNING", "WAITING_USER"}:
                raise ResumeUnavailable(
                    f"RESUME_UNAVAILABLE: Run 当前状态不可恢复：{stored.status}。"
                )
            self.session_manager.set_active_run(session_id, stored.run_id)
            return session_id, stored.run_id, stored.trace_id or trace_id, stored

        session = self.session_manager.resolve(
            session_id,
            title=instruction[:120],
            project_id=project_id,
        )
        runtime = self.session_manager.open(session.session_id)
        existing_session_threads = {
            run.thread_id
            for run in runtime.list_runs()
        }
        if any(
            run.thread_id == thread_id
            for candidate in self.session_manager.list(include_deleted=False)
            for run in self.session_manager.open(candidate.session_id).list_runs()
        ) or thread_id in existing_session_threads or (
            self.checkpointer is not None
            and checkpointer_has_checkpoint(self.checkpointer, thread_id)
        ):
            raise CorrelationConflict(
                f"RUN_CORRELATION_INVALID: thread_id 已绑定到另一个运行上下文：{thread_id}。"
            )
        run_id = f"RUN_{secrets.token_hex(8)}"
        runtime.create_run(
            instruction,
            run_id=run_id,
            thread_id=thread_id,
            trace_id=trace_id,
            project_id=project_id,
        )
        runtime.start_run(run_id, worker_id=self.worker_id)
        set_checkpoint = getattr(runtime, "set_checkpoint_ref", None)
        if callable(set_checkpoint):
            set_checkpoint(run_id, thread_id)
        self.session_manager.set_active_run(session.session_id, run_id)
        return session.session_id, run_id, trace_id, None

    def _finish_session_run(
        self,
        *,
        session_id: str,
        run_id: str,
        status: str,
        value: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.session_manager is None:
            return
        try:
            runtime = self.session_manager.open(session_id)
            run_status = {
                "INTERRUPTED": "INTERRUPTED",
                "FAILED": "FAILED",
                "WAITING_USER": "WAITING_USER",
            }.get(status, "COMPLETED")
            serialized = _jsonable(value) if value is not None else {}
            answer = json.dumps(serialized, ensure_ascii=False, default=str)
            runtime.complete_run(
                run_id,
                status=run_status,
                answer=answer,
                citations=[],
                evidence=[],
                metadata=dict(metadata or {}),
            )
            self.session_manager.set_active_run(session_id, None)
        except (KeyError, OSError, ValueError):
            # A failed projection must not make an already completed domain
            # result look like a successful Session lifecycle update.
            return

    @staticmethod
    def _failure_reason(error: Exception) -> str:
        code = str(getattr(error, "code", ""))
        message = str(error)
        if code in {"PROVIDER_UNAVAILABLE", "WEB_PROVIDER_UNAVAILABLE", "RESEARCH_FAILED"}:
            return "PROVIDER_FAILED"
        if code in {"RESUME_UNAVAILABLE", "CHECKPOINT_UNAVAILABLE"}:
            return "RESUME_UNAVAILABLE"
        if code in {"CAPABILITY_DENIED", "CAPABILITY_VIOLATION"}:
            return "CAPABILITY_DENIED"
        if code in {"DOMAIN_CONFLICT", "FACT_CONFLICT", "CONTRIBUTION_CONFLICT"}:
            return "DOMAIN_CONFLICT"
        if type(error).__name__ in {
            "HTTPStatusError",
            "RequestError",
            "TimeoutException",
            "ReadTimeout",
            "ConnectError",
        }:
            return "PROVIDER_FAILED"
        if code in {"BUDGET_EXHAUSTED", "CONTEXT_BUDGET_EXCEEDED"} or "CONTEXT_BUDGET_EXCEEDED" in message or "预算耗尽" in message:
            return "BUDGET_EXHAUSTED"
        return f"HARNESS_FAILED:{type(error).__name__}"

    @staticmethod
    def _result_error_codes(value: Any, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
        raw = tuple(getattr(value, "error_codes", ()) or ()) or tuple(fallback)
        metadata = getattr(value, "metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("error_code"):
            raw = (*raw, str(metadata["error_code"]))
        return tuple(dict.fromkeys(str(code) for code in raw if code))

    @staticmethod
    def _termination_reason(value: Any, error_codes: tuple[str, ...]) -> str:
        if value is None:
            return "SKILL_EXECUTION_FAILED"
        status = str(getattr(value, "status", "READY"))
        codes = set(error_codes)
        value_metadata = getattr(value, "metadata", {})
        if isinstance(value_metadata, Mapping):
            metadata_code = value_metadata.get("error_code")
            if metadata_code:
                codes.add(str(metadata_code))
        if status == "NEEDS_USER_REVIEW":
            return "NEEDS_USER_REVIEW"
        if codes & {
            "PROVIDER_UNAVAILABLE",
            "WEB_PROVIDER_UNAVAILABLE",
            "WEB_PROVIDER_TIMEOUT",
            "WEB_PROVIDER_RATE_LIMITED",
            "FRESHNESS_UNAVAILABLE",
            "RESEARCH_FAILED",
        }:
            return "PROVIDER_FAILED"
        if codes & {"WEB_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"}:
            return "BUDGET_EXHAUSTED"
        if "CAPABILITY_DENIED" in codes:
            return "CAPABILITY_DENIED"
        if "INSUFFICIENT_EVIDENCE" in codes or "INSUFFICIENT_MANUSCRIPT_STATE" in codes:
            return "INSUFFICIENT_EVIDENCE"
        if "CONFLICT" in status or codes & {"FACT_CONFLICT", "CONTRIBUTION_CONFLICT", "PROJECT_CONFLICT"}:
            return "DOMAIN_CONFLICT"
        if status == "FAILED" or codes:
            return "SKILL_EXECUTION_FAILED"
        return "COMPLETED"

    @staticmethod
    def _trace_with_correlation(
        harness: ResearchRunHarness,
        session_id: str,
        trace_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        diagnostics = harness.diagnostics()
        diagnostics.update(
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "project_id": project_id,
            }
        )
        return diagnostics

    def _make_writing_request(
        self,
        task_type: TaskType,
        instruction: str,
        *,
        project_id: str,
        session_id: str,
        run_id: str,
    ) -> WritingRequest:
        section = {
            "WRITE_INTRODUCTION": "introduction",
            "WRITE_CONCLUSION": "conclusion",
            "WRITE_ABSTRACT": "abstract",
        }[task_type]
        return WritingRequest(
            request_id=run_id,
            project_id=project_id,
            instruction=instruction,
            target_section=section,
            session_id=session_id,
            task_type=task_type,
        )

    def _execute_skill(
        self,
        task_type: str,
        instruction: str,
        *,
        project_id: str,
        session_id: str,
        harness: ResearchRunHarness,
        result_holder: dict[str, Any],
    ) -> dict[str, Any]:
        if task_type not in {"WRITE_INTRODUCTION", "SUPPORT_CLAIM", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            result_holder["error_codes"] = ("UNSUPPORTED_TASK",)
            return {"error_code": "UNSUPPORTED_TASK", "status": "FAILED"}
        if task_type != result_holder["task_type"]:
            result_holder["error_codes"] = ("TASK_MISMATCH",)
            return {"error_code": "TASK_MISMATCH", "status": "FAILED"}
        if not result_holder.get("context_ready"):
            result_holder["error_codes"] = ("CONTEXT_REQUIRED",)
            return {"error_code": "CONTEXT_REQUIRED", "status": "FAILED"}
        started = perf_counter()
        try:
            if task_type == "SUPPORT_CLAIM":
                request: Any = SupportClaimRequest(
                    request_id=harness.run_id,
                    project_id=project_id,
                    claim=instruction,
                    session_id=session_id,
                    metadata={
                        "_precomputed_evidence_packs": tuple(
                            result_holder.get("research_packs", ())
                        ),
                    }
                    if result_holder.get("research_packs")
                    else {},
                )
                value = self.skill_runtime.execute(request).value
            else:
                request = self._make_writing_request(
                    task_type, instruction, project_id=project_id, session_id=session_id, run_id=harness.run_id
                )
                writer = self.writers.get(task_type) or self.writers.get(request.target_section)
                if writer is None:
                    result_holder["error_codes"] = ("SKILL_NOT_AVAILABLE",)
                    return {"error_code": "SKILL_NOT_AVAILABLE", "status": "FAILED"}
                value = self.skill_runtime.execute(request, writer=writer).value
            result_holder["value"] = value
            harness.record_tool(
                "execute_scholar_skill",
                "succeeded",
                (perf_counter() - started) * 1000,
                {"task_type": task_type, "result_type": type(value).__name__},
            )
            return {"status": "COMPLETED", "result_type": type(value).__name__, "result": _jsonable(value)}
        except SkillRuntimeError as error:
            result_holder["error_codes"] = (error.code,)
            harness.record_tool(
                "execute_scholar_skill",
                "failed",
                (perf_counter() - started) * 1000,
                {"task_type": task_type, "error_code": error.code},
            )
            return {"status": "FAILED", "error_code": error.code, "message": str(error)}
        except Exception as error:
            result_holder["error_codes"] = ("SKILL_EXECUTION_FAILED",)
            harness.record_tool(
                "execute_scholar_skill",
                "failed",
                (perf_counter() - started) * 1000,
                {"task_type": task_type, "error_type": type(error).__name__},
            )
            return {"status": "FAILED", "error_code": "SKILL_EXECUTION_FAILED", "message": str(error)}

    def _research_tool(
        self,
        query: str,
        purpose: str,
        target_claim: str | None,
        freshness_mode: str,
        requested_from: date | None,
        requested_to: date | None,
        explicit_latest: bool,
        *,
        harness: ResearchRunHarness,
        result_holder: dict[str, Any],
        capabilities: Any,
    ) -> dict[str, Any]:
        if self.research is None:
            return {"status": "FAILED", "error_code": "SKILL_NOT_AVAILABLE"}
        if not capabilities.allows("LOCAL_RESEARCH"):
            return {"status": "FAILED", "error_code": "CAPABILITY_DENIED"}
        if not result_holder.get("context_ready"):
            return {"status": "FAILED", "error_code": "CONTEXT_REQUIRED"}
        allow_web = capabilities.allows("WEB_RESEARCH")
        budget = ResearchBudget(max_web_queries=1 if allow_web else 0)
        request = ResearchRequest(
            request_id=f"{harness.run_id}:research:{secrets.token_hex(4)}",
            query=query,
            workspace_id=self.research.workspace_id,
            scope_version=self.research.scope_version,
            purpose=purpose,  # type: ignore[arg-type]
            target_claims=(target_claim or query,),
            freshness_mode=freshness_mode,  # type: ignore[arg-type]
            requested_from=requested_from,
            requested_to=requested_to,
            explicit_latest=explicit_latest,
            budget=budget,
        )
        started = perf_counter()
        self._record_context(
            harness,
            "research_subagent",
            {
                "query": query,
                "purpose": purpose,
                "target_claim": target_claim,
                "freshness_mode": freshness_mode,
                "requested_from": requested_from,
                "requested_to": requested_to,
            },
        )
        try:
            pack = self.research.research(request, allow_web=allow_web, harness=harness)
        except Exception as error:
            harness.record_tool(
                "research_evidence",
                "failed",
                (perf_counter() - started) * 1000,
                {
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                },
            )
            return {"status": "FAILED", "error_code": "RESEARCH_FAILED", "message": str(error)}
        result_holder.setdefault("research_packs", []).append(pack)
        harness.record_tool(
            "research_evidence",
            "succeeded",
            (perf_counter() - started) * 1000,
            {"result_type": "EvidencePack", "web_used": pack.web_used, "evidence_count": len(pack.evidence)},
        )
        return {"status": "COMPLETED", "result_type": "EvidencePack", "result": _jsonable(pack)}

    def _review_tool(
        self,
        task_type: str,
        draft: str,
        *,
        harness: ResearchRunHarness,
        result_holder: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.reviewer is None:
            return {"status": "FAILED", "error_code": "REVIEWER_NOT_INJECTED"}
        started = perf_counter()
        review_context = {
            "task_type": task_type,
            "draft": draft,
            "claim_plan": (
                replace(result_holder["claim_plan"], research_needs=())
                if is_dataclass(result_holder.get("claim_plan"))
                else result_holder.get("claim_plan")
            ),
            "evidence": self._bounded_reviewer_evidence(result_holder.get("evidence")),
            "facts": result_holder.get("facts"),
            "confirmed_contributions": result_holder.get("contributions"),
            "review_policy": task_type,
        }
        self._record_context(harness, "reviewer_subagent", review_context)
        try:
            value = self.reviewer(review_context)
            if not isinstance(value, ReviewReport):
                return {"status": "FAILED", "error_code": "REVIEW_FAILED", "message": "Reviewer adapter 必须返回 ReviewReport。"}
            harness.record_tool("review_draft", "succeeded", (perf_counter() - started) * 1000, {"result_type": type(value).__name__})
            return {"status": "COMPLETED", "result_type": type(value).__name__, "result": _jsonable(value)}
        except Exception as error:
            harness.record_tool("review_draft", "failed", (perf_counter() - started) * 1000, {"error_type": type(error).__name__})
            return {"status": "FAILED", "error_code": "REVIEW_FAILED", "message": str(error)}

    def _build_agent(
        self,
        definition: SkillDefinition,
        *,
        harness: ResearchRunHarness,
        project_id: str,
        session_id: str,
        user_instruction: str,
        result_holder: dict[str, Any],
    ) -> Any:
        # Keep the skill source readable but keep Deep Agents' automatic
        # conversation/large-result artifacts out of the repository.  The
        # root route is the existing skill directory; the artifact route is
        # ephemeral StateBackend data carried by the LangGraph checkpoint.
        # Agent write/edit/execute tools remain excluded by the Harness profile
        # and are not made available by this backend composition.
        backend = CompositeBackend(
            default=StateBackend(),
            routes={
                "/harness/": StateBackend(),
                "/": FilesystemBackend(
                    root_dir=self.skill_adapter.source_root,
                    virtual_mode=True,
                ),
            },
            artifacts_root="/harness",
        )
        def readonly_filesystem() -> FilesystemMiddleware:
            return FilesystemMiddleware(backend=backend, tools=["read_file"])

        def get_project_context() -> dict[str, Any]:
            value = self._project_context()
            result_holder["context_ready"] = True
            harness.record_tool("get_project_context", "succeeded", 0.0, {"section_count": len(value["sections"])})
            return value

        def execute_scholar_skill(task_type: str, instruction: str) -> dict[str, Any]:
            # The selected route and the original user request are the Harness
            # authority.  A model-generated tool argument may summarize or
            # rewrite the request; allowing that text into the Domain Runtime
            # can mismatch precomputed EvidencePacks and changes the claim the
            # user asked to evaluate.  Keep the argument in the tool schema for
            # the framework contract, but execute the original request.
            del instruction
            result = self._execute_skill(
                task_type,
                user_instruction,
                project_id=project_id,
                session_id=session_id,
                harness=harness,
                result_holder=result_holder,
            )
            value = result_holder.get("value")
            if value is not None:
                result_holder["claim_plan"] = getattr(value, "claim_plan", None)
                packs = getattr(value, "evidence_packs", ())
                result_holder["evidence"] = tuple(
                    item
                    for pack in packs
                    for item in getattr(pack, "evidence", ())
                )
                result_holder["facts"] = self.project_store.list_facts()
                result_holder["contributions"] = self.project_store.list_contributions()
            return result

        def get_patch_status() -> dict[str, Any]:
            harness.record_tool("get_patch_status", "succeeded", 0.0, {"approval_tool_visible": False})
            return {"approval_required": True, "approval_tool_visible": False, "apply_tool_visible": False}

        main_tools = [
            _tool(
                get_project_context,
                name="get_project_context",
                description="Read a compact Project/Manuscript summary. Never returns database handles or write access.",
                args_schema=_EmptyInput,
            ),
            _tool(
                execute_scholar_skill,
                name="execute_scholar_skill",
                description="Execute the already selected Scholar Skill and return its existing Domain Result. It never approves or applies a patch.",
                args_schema=_SkillInput,
                # A production bundle injects the isolated Reviewer adapter,
                # so writing runs must remain in the graph long enough for the
                # Supervisor to delegate review.  Lightweight callers that do
                # not inject a reviewer retain the historical direct-result
                # behavior used by unit/local-fast runtimes.
                return_direct=definition.task_type == "SUPPORT_CLAIM" or self.reviewer is None,
            ),
            _tool(
                get_patch_status,
                name="get_patch_status",
                description="Read DraftPatch/approval status only. Approval and Apply are never available to an Agent.",
                args_schema=_EmptyInput,
            ),
        ]

        subagents: list[SubAgent] = []
        if definition.capabilities.allows("LOCAL_RESEARCH"):
            research_capabilities = definition.capabilities

            research_instruction = (
                "This is a Support Claim run: call research_evidence at most once. "
                "Do not retry it, invent query variants, or delegate research again; "
                "an insufficient EvidencePack is a valid bounded result."
                if definition.task_type == "SUPPORT_CLAIM"
                else
                "This is an Introduction run: call research_evidence at most once per "
                "distinct ResearchNeed (at most three calls total), do not retry a need "
                "or invent extra query variants, then return the bounded EvidencePacks."
            )

            def research_evidence(
                query: str,
                purpose: str = "background",
                target_claim: str | None = None,
                freshness_mode: str = "LOCAL_ONLY",
                requested_from: str | None = None,
                requested_to: str | None = None,
                explicit_latest: bool = False,
            ) -> dict[str, Any]:
                return self._research_tool(
                    query,
                    purpose,
                    target_claim,
                    freshness_mode,
                    requested_from,
                    requested_to,
                    explicit_latest,
                    harness=harness,
                    result_holder=result_holder,
                    capabilities=research_capabilities,
                )

            research_tool = _tool(
                research_evidence,
                name="research_evidence",
                description="Research only through ResearchCapabilityService; returns EvidencePack and never writes manuscript state.",
                args_schema=_ResearchInput,
            )
            subagents.append({
                "name": "research",
                "description": "Researches bounded Local/Web evidence through ResearchCapabilityService and returns EvidencePack only.",
                "tools": [research_tool],
                "middleware": [readonly_filesystem()],
                "system_prompt": (
                    "You are the isolated Research Subagent. Use only research_evidence. "
                    f"{research_instruction} "
                    "Never edit files, facts, contributions, bibliography, or patches."
                ),
            })

        if definition.task_type in {"WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            def review_draft(task_type: str, draft: str) -> dict[str, Any]:
                return self._review_tool(
                    task_type,
                    draft,
                    harness=harness,
                    result_holder=result_holder,
                )

            reviewer_tool = _tool(
                review_draft,
                name="review_draft",
                description="Review a proposed draft through the injected SharedManuscriptReviewer adapter; never researches or writes.",
                args_schema=_ReviewInput,
            )
            subagents.append({
                "name": "reviewer",
                "description": "Reviews a selected draft under the current Skill ReviewPolicy and returns ReviewReport only.",
                "tools": [reviewer_tool],
                "middleware": [readonly_filesystem()],
                "system_prompt": "You are the isolated Reviewer Subagent. Use only review_draft. Never research, edit files, mutate Facts/Contributions, or approve patches.",
            })

        review_instruction = (
            "For this writing task, after execute_scholar_skill returns, delegate the proposed draft to "
            "the reviewer subagent with review_draft before giving the final response. "
            if definition.task_type != "SUPPORT_CLAIM"
            else "After execute_scholar_skill returns, report the ClaimSupportResult without review delegation. "
        )
        system_prompt = (
            "You are Scholar Deep Agent, the user-level Harness for an existing Scholar domain runtime. "
            f"The selected task is {definition.task_type} and selected skill is {definition.name}. "
            "Use get_project_context first. Load only the selected SKILL.md progressively with read_file when needed; "
            "do not use the read-only skills filesystem to inspect manuscript files because the authorized Project "
            "Context and Skill Runtime provide manuscript state. "
            "Follow the selected skill's CapabilityProfile and delegate research/review only to the named isolated subagent. "
            "If research is available, call the research subagent exactly once; never repeat the delegation. "
            "Call execute_scholar_skill exactly once after required context is ready. Its result is authoritative. "
            f"{review_instruction}"
            "Never call filesystem write/edit/delete/execute, never write manuscript/bibliography/facts/contributions, "
            "and never approve or apply DraftPatch. Return no invented Domain Result."
        )
        return create_deep_agent(
            model=self._model,
            tools=main_tools,
            system_prompt=system_prompt,
            skills=["/"],
            subagents=subagents,
            middleware=[readonly_filesystem()],
            backend=backend,
            checkpointer=self.checkpointer,
            interrupt_on=self.interrupt_on,
            name="scholar-deep-agent",
        )

    def _run_scholar_request(
        self,
        instruction: str,
        project_id: str,
        *,
        session_id: str | None = None,
        task_type: str | None = None,
        thread_id: str | None = None,
        resume_value: Any = _NO_RESUME,
    ) -> ScholarHarnessResult:
        if not instruction.strip() or not project_id.strip():
            raise ValueError("instruction/project_id 不能为空。")
        if project_id != self.project_store.project_id:
            raise SkillRuntimeError("PROJECT_CONFLICT", "Scholar Harness Project 不一致。")
        decision = self.skill_runtime.route(task_type=task_type, instruction=instruction)
        trace_id = f"TRACE_{secrets.token_hex(8)}"
        thread = thread_id or f"THREAD_{secrets.token_hex(8)}"
        session, run_id, trace_id, _stored_run = self._prepare_session_run(
            instruction,
            project_id,
            session_id=session_id,
            thread_id=thread,
            trace_id=trace_id,
            resuming=resume_value is not _NO_RESUME,
        )
        run_harness = ResearchRunHarness(
            "scholar_deep_agent",
            self.harness_policy,
            run_id=run_id,
        )
        if decision.needs_user_choice or decision.task_type is None:
            run_harness.finish(HarnessState.REFUSED, "task_selection_required")
            self._finish_session_run(
                session_id=session,
                run_id=run_id,
                status="FAILED",
                metadata={"termination_reason": "task_selection_required"},
            )
            return ScholarHarnessResult(
                None,
                None,
                session,
                run_harness.run_id,
                thread,
                trace_id,
                "NEEDS_USER_CHOICE",
                error_codes=("UNSUPPORTED_TASK",),
                    metadata={
                        "routing": asdict(decision),
                        "trace": self._trace_with_correlation(run_harness, session, trace_id, project_id),
                    },
                )

        definition = self.skill_adapter.definition(decision.task_type)
        try:
            permission = self.permission_audit(decision.task_type)
        except Exception as error:
            reason = self._failure_reason(error)
            run_harness.finish(HarnessState.FAILED, reason)
            self._finish_session_run(
                session_id=session,
                run_id=run_id,
                status="FAILED",
                metadata={"error_type": type(error).__name__, "error": str(error), "termination_reason": reason},
            )
            return ScholarHarnessResult(
                decision.task_type,
                self.skill_adapter.definition(decision.task_type).name,
                session,
                run_harness.run_id,
                thread,
                trace_id,
                "FAILED",
                error_codes=("CAPABILITY_VIOLATION",),
                metadata={"error_type": type(error).__name__, "error": str(error), "trace": self._trace_with_correlation(run_harness, session, trace_id, project_id)},
            )
        result_holder: dict[str, Any] = {
            "task_type": decision.task_type,
            # A resumed graph already contains the persisted context tool
            # result; rebuilding the Python closure must not require the
            # model to repeat that read-only step.
            "context_ready": resume_value is not _NO_RESUME,
        }
        try:
            run_harness.transition(HarnessState.CONTEXT_PREPARING)
            with run_harness.step("HARNESS_CONTEXT") as details:
                details.update({
                    "selected_skill": definition.name,
                    "context_audiences": ["supervisor", "skill_metadata"],
                    "full_skill_loaded": False,
                })
                self._record_context(
                    run_harness,
                    "supervisor",
                    {"instruction": instruction, "project_id": project_id},
                )
                self._record_context(
                    run_harness,
                    "supervisor",
                    self.skill_adapter.metadata(),
                )
            run_harness.transition(HarnessState.PLANNING)
            with run_harness.step("HARNESS_PLAN") as details:
                details.update({"task_type": decision.task_type, "skill": definition.name, "confidence": decision.confidence})
            run_harness.transition(HarnessState.EXECUTING)
        except Exception as error:
            reason = self._failure_reason(error)
            run_harness.finish(HarnessState.FAILED, reason)
            self._finish_session_run(
                session_id=session,
                run_id=run_id,
                status="FAILED",
                metadata={"error_type": type(error).__name__, "error": str(error), "termination_reason": reason},
            )
            return ScholarHarnessResult(
                decision.task_type,
                definition.name,
                session,
                run_harness.run_id,
                thread,
                trace_id,
                "FAILED",
                error_codes=("HARNESS_EXECUTION_FAILED",),
                metadata={
                    "routing": asdict(decision),
                    "selected_skill": definition.name,
                    "resumed": resume_value is not _NO_RESUME,
                    "visible_capabilities": sorted(definition.capabilities.allowed),
                    "permission_audit": permission,
                    "subagent_names": [
                        name
                        for name in ("research", "reviewer")
                        if (name == "research" and definition.capabilities.allows("LOCAL_RESEARCH"))
                        or (name == "reviewer" and definition.task_type != "SUPPORT_CLAIM")
                    ],
                    "visible_tools": {
                        "main": ["get_project_context", "execute_scholar_skill", "get_patch_status", "read_file"],
                        "research": ["research_evidence", "read_file"]
                        if definition.capabilities.allows("LOCAL_RESEARCH")
                        else [],
                        "reviewer": ["review_draft", "read_file"]
                        if definition.task_type != "SUPPORT_CLAIM"
                        else [],
                    },
                    "unexpected_tool_calls": [
                        event.name
                        for event in run_harness.trace
                        if event.kind == "tool" and event.name in (_SAFE_FS_TOOLS | _FORBIDDEN_AGENT_TOOLS)
                    ],
                    "capability_violations": [
                        event.name
                        for event in run_harness.trace
                        if event.kind == "tool" and event.name in _FORBIDDEN_AGENT_TOOLS
                    ],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trace": self._trace_with_correlation(run_harness, session, trace_id, project_id),
                },
            )
        try:
            agent = self._build_agent(
                definition,
                harness=run_harness,
                project_id=project_id,
                session_id=session,
                user_instruction=instruction,
                result_holder=result_holder,
            )
            invoke_input: Any
            if resume_value is _NO_RESUME:
                invoke_input = {"messages": [{"role": "user", "content": instruction}]}
            else:
                invoke_input = Command(resume=resume_value)
            state = agent.invoke(invoke_input, config={"configurable": {"thread_id": thread}})
            if isinstance(state, Mapping) and state.get("__interrupt__"):
                run_harness.finish(HarnessState.INTERRUPTED, "checkpoint_interrupt")
                interrupt_metadata = {
                    "routing": asdict(decision),
                    "selected_skill": definition.name,
                    "resumed": resume_value is not _NO_RESUME,
                    "interrupts": _jsonable(state.get("__interrupt__")),
                }
                self._finish_session_run(
                    session_id=session,
                    run_id=run_id,
                    status="INTERRUPTED",
                    metadata=interrupt_metadata,
                )
                return ScholarHarnessResult(
                    decision.task_type,
                    definition.name,
                    session,
                    run_harness.run_id,
                    thread,
                    trace_id,
                    "INTERRUPTED",
                    error_codes=(),
                    metadata={
                        **interrupt_metadata,
                        "trace": self._trace_with_correlation(run_harness, session, trace_id, project_id),
                    },
                )
            run_harness.transition(HarnessState.EVALUATING)
            with run_harness.step("HARNESS_RESULT") as details:
                details.update({"domain_result_present": result_holder.get("value") is not None})
            run_harness.transition(HarnessState.COMMITTING)
            value = result_holder.get("value")
            error_codes = self._result_error_codes(
                value,
                tuple(result_holder.get("error_codes", ())),
            )
            domain_reason = self._termination_reason(value, error_codes)
            run_harness.finish(HarnessState.COMPLETED, domain_reason)
        except Exception as error:
            reason = self._failure_reason(error)
            run_harness.finish(HarnessState.FAILED, reason)
            self._finish_session_run(
                session_id=session,
                run_id=run_id,
                status="FAILED",
                metadata={"error_type": type(error).__name__, "error": str(error), "termination_reason": reason},
            )
            return ScholarHarnessResult(
                decision.task_type,
                definition.name,
                session,
                run_harness.run_id,
                thread,
                trace_id,
                "FAILED",
                error_codes=("HARNESS_EXECUTION_FAILED",),
                metadata={
                    "routing": asdict(decision),
                    "selected_skill": definition.name,
                    "resumed": resume_value is not _NO_RESUME,
                    "visible_capabilities": sorted(definition.capabilities.allowed),
                    "permission_audit": permission,
                    "subagent_names": [
                        name
                        for name in ("research", "reviewer")
                        if (name == "research" and definition.capabilities.allows("LOCAL_RESEARCH"))
                        or (name == "reviewer" and definition.task_type != "SUPPORT_CLAIM")
                    ],
                    "visible_tools": {
                        "main": ["get_project_context", "execute_scholar_skill", "get_patch_status", "read_file"],
                        "research": ["research_evidence", "read_file"]
                        if definition.capabilities.allows("LOCAL_RESEARCH")
                        else [],
                        "reviewer": ["review_draft", "read_file"]
                        if definition.task_type != "SUPPORT_CLAIM"
                        else [],
                    },
                    "unexpected_tool_calls": [
                        event.name
                        for event in run_harness.trace
                        if event.kind == "tool" and event.name in (_SAFE_FS_TOOLS | _FORBIDDEN_AGENT_TOOLS)
                    ],
                    "capability_violations": [
                        event.name
                        for event in run_harness.trace
                        if event.kind == "tool" and event.name in _FORBIDDEN_AGENT_TOOLS
                    ],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trace": self._trace_with_correlation(run_harness, session, trace_id, project_id),
                },
            )
        value = result_holder.get("value")
        skill_status = getattr(value, "status", None) or ("COMPLETED" if value is not None else "FAILED")
        error_codes = self._result_error_codes(
            value,
            tuple(result_holder.get("error_codes", ())),
        )
        if value is None and not error_codes:
            error_codes = ("SKILL_NOT_EXECUTED",)
            skill_status = "FAILED"
        metadata = {
            "routing": asdict(decision),
            "selected_skill": definition.name,
            "resumed": resume_value is not _NO_RESUME,
            "visible_capabilities": sorted(definition.capabilities.allowed),
            "permission_audit": permission,
            "skill_metadata_loaded_progressively": True,
            "subagent_names": [
                name for name in ("research", "reviewer")
                if (name == "research" and definition.capabilities.allows("LOCAL_RESEARCH"))
                or (name == "reviewer" and definition.task_type != "SUPPORT_CLAIM")
            ],
            "visible_tools": {
                "main": ["get_project_context", "execute_scholar_skill", "get_patch_status", "read_file"],
                "research": ["research_evidence", "read_file"]
                if definition.capabilities.allows("LOCAL_RESEARCH")
                else [],
                "reviewer": ["review_draft", "read_file"]
                if definition.task_type != "SUPPORT_CLAIM"
                else [],
            },
            "unexpected_tool_calls": [
                event.name
                for event in run_harness.trace
                if event.kind == "tool" and event.name in (_SAFE_FS_TOOLS | _FORBIDDEN_AGENT_TOOLS)
            ],
            "capability_violations": [
                event.name
                for event in run_harness.trace
                if event.kind == "tool" and event.name in _FORBIDDEN_AGENT_TOOLS
            ],
            "trace": run_harness.diagnostics(),
            "agent_state_keys": sorted(state.keys()) if isinstance(state, Mapping) else (),
        }
        metadata["context_budget"] = asdict(self.context_budget)
        metadata["trace"] = self._trace_with_correlation(run_harness, session, trace_id, project_id)
        self._finish_session_run(
            session_id=session,
            run_id=run_id,
            status="COMPLETED" if str(skill_status) != "FAILED" else "FAILED",
            value=value,
            metadata={
                "termination_reason": run_harness.termination_reason,
                "result_type": type(value).__name__ if value is not None else None,
            },
        )
        return ScholarHarnessResult(
            decision.task_type,
            definition.name,
            session,
            run_harness.run_id,
            thread,
            trace_id,
            str(skill_status),
            type(value).__name__ if value is not None else None,
            value,
            error_codes,
            metadata,
        )

    def scholar_request(
        self,
        instruction: str,
        project_id: str,
        *,
        session_id: str | None = None,
        task_type: str | None = None,
        thread_id: str | None = None,
    ) -> ScholarHarnessResult:
        """Start a new user-level Harness run."""

        return self._run_scholar_request(
            instruction,
            project_id,
            session_id=session_id,
            task_type=task_type,
            thread_id=thread_id,
        )

    def resume(
        self,
        thread_id: str,
        resume_value: Any,
        project_id: str,
        *,
        instruction: str,
        session_id: str | None = None,
        task_type: str | None = None,
    ) -> ScholarHarnessResult:
        """Resume an interrupted Deep Agent thread from persisted state.

        ``resume_value`` is the framework interrupt decision, not a Patch
        Approval decision.  Human Approval remains outside this service.
        The caller must keep the injected Checkpointer open for the duration
        of the call; a new service/checkpointer instance may be used after a
        process restart.
        """

        if not thread_id.strip():
            raise ValueError("thread_id 不能为空。")
        if self.checkpointer is None or not checkpointer_has_checkpoint(self.checkpointer, thread_id):
            raise ResumeUnavailable(
                f"RESUME_UNAVAILABLE: thread {thread_id} 没有可恢复的 persistent checkpoint。"
            )
        return self._run_scholar_request(
            instruction,
            project_id,
            session_id=session_id,
            task_type=task_type,
            thread_id=thread_id,
            resume_value=resume_value,
        )

    def status(
        self,
        thread_id: str,
        project_id: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Read-only runtime status; it never resumes or mutates a run."""

        if project_id != self.project_store.project_id:
            raise SkillRuntimeError("PROJECT_CONFLICT", "Scholar Harness Project 不一致。")
        result: dict[str, Any] = {
            "project_id": project_id,
            "session_id": session_id,
            "thread_id": thread_id,
            "checkpoint_available": checkpointer_has_checkpoint(self.checkpointer, thread_id)
            if self.checkpointer is not None
            else False,
            "resume_available": False,
        }
        if self.session_manager is not None and session_id:
            try:
                session_record = self.session_manager.get(session_id)
                if session_record.project_id and session_record.project_id != project_id:
                    raise CorrelationConflict(
                        "RUN_CORRELATION_INVALID: Session 与 Project 不匹配。"
                    )
                runtime = self.session_manager.open(session_id)
                runs = [run for run in runtime.list_runs() if run.thread_id == thread_id]
            except CorrelationConflict:
                raise
            except (KeyError, ValueError):
                runs = []
            if runs:
                run = runs[-1]
                result["run"] = _jsonable(run)
                result["resume_available"] = bool(
                    result["checkpoint_available"] and run.status in {"INTERRUPTED", "RUNNING", "WAITING_USER"}
                )
        else:
            result["resume_available"] = bool(result["checkpoint_available"])
        return result

    resume_scholar_request = resume
