"""Unified Scholar orchestration entry point and CrewAI backend."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from app.jobs.worker import JobCancelled

from app.orchestration.contracts import (
    ManagerDecision,
    OrchestrationRequest,
    OrchestrationResult,
    ResearchAgentOutput,
    ReviewAgentOutput,
    RunBudget,
    RunGoal,
    Route,
    default_run_goal,
    WriterAgentOutput,
)
from app.scholar.models import EvidencePack, ReviewReport
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.project import ScholarProjectStore
from app.scholar.research import (
    ResearchBudget,
    ResearchCapabilityService,
    ResearchRequest,
)
from app.scholar.writing.models import WritingRequest
from app.scholar.writing.citation_coverage import (
    INTRODUCTION_EVIDENCE_LIMIT,
    INTRODUCTION_MINIMUM_UNIQUE_PAPERS,
    INTRODUCTION_PAPER_RETRIEVAL_LIMIT,
    INTRODUCTION_SECTION_RETRIEVAL_LIMIT,
)
from app.scholar.writing.runtime import SkillRuntimeError
from app.scholar.writing.service import ResearchDelegate
from app.scholar.writing.skill import IntroductionSkill
from app.scholar.writing.support import ClaimSupportResult, SupportClaimRequest
from app.session.context import ManagerContextBuilder


class _LazyCapabilityMatrix:
    """Resolve CrewAI tool definitions only when the CrewAI backend runs."""

    def __getattr__(self, name: str) -> Any:
        from app.orchestration.crewai.tools import CapabilityMatrix as _Matrix

        return getattr(_Matrix, name)


CapabilityMatrix = _LazyCapabilityMatrix()


@contextmanager
def _crewai_import_guard() -> Any:
    """Import CrewAI without allowing it to mutate the host process env.

    CrewAI 1.15 imports ``dotenv.load_dotenv`` from its LLM/event modules at
    import time.  The Scholar application already owns project-scoped config
    loading, so letting that call run would make importing the optional
    framework affect retrieval, tests, and unrelated providers.
    """

    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        dotenv.load_dotenv = original


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="python"))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class CrewAIBackend:
    """Application adapter joining CrewAI cognition to existing domain APIs."""

    def __init__(
        self,
        project_root: Path,
        *,
        model: Any | None,
        skill_runtime: Any,
        research: ResearchCapabilityService,
        writers: Mapping[str, Any] | None = None,
        reviewer: Any | None = None,
        project_store: ScholarProjectStore | None = None,
        session_manager: Any | None = None,
        event_store: Any | None = None,
        max_review_rounds: int = 2,
        max_research_attempts: int = 2,
        max_manager_steps: int = 12,
        max_tool_calls: int = 16,
        max_token_budget: int | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.model = model
        self.skill_runtime = skill_runtime
        self.research_service = research
        self.writers = dict(writers or {})
        self.reviewer = reviewer
        self.project_store = (
            project_store
            or getattr(skill_runtime, "project_store", None)
            or ScholarProjectStore(self.project_root)
        )
        self.session_manager = session_manager
        self.event_store = event_store
        self.max_review_rounds = max(0, min(2, max_review_rounds))
        self.max_research_attempts = max(1, min(8, max_research_attempts))
        self.max_manager_steps = max(3, min(32, max_manager_steps))
        self.max_tool_calls = max(1, max_tool_calls)
        self.max_token_budget = max_token_budget
        self._manager_context_builder = ManagerContextBuilder(
            recent_limit=6, context_budget_chars=7000
        )
        # Keep CrewAI out of module import time. Its dependency graph may load
        # a process-level dotenv file; the application owns project-scoped
        # configuration and must not let a framework import mutate it.
        self.crew: Any | None = None
        self._flow_result: OrchestrationResult | None = None
        self._trace: Any | None = None
        self._request: OrchestrationRequest | None = None
        self._run_id: str | None = None
        self._session_run: Any | None = None
        self._run_goal: RunGoal | None = None
        self._tool_calls_used = 0
        self._active_route: Route | None = None
        self._active_research: ResearchAgentOutput | None = None
        self._active_research_domain_result: Any | None = None
        self._active_writer_domain_result: Any | None = None
        self._active_review_round = 0
        self._cancellation_checker: Callable[[], None] | None = None

    @property
    def agents(self) -> dict[str, Any]:
        # Expose the canonical role/tool matrix for diagnostics without using
        # this shared introspection Crew to execute a Run. Actual execution is
        # always performed by ``_new_run_backend``.
        if self.crew is None:
            self._ensure_crew(None)
        return dict(self.crew.agents) if self.crew is not None else {}

    @staticmethod
    def _route_hint(request: OrchestrationRequest) -> Route:
        task_type = request.task_type
        if task_type in {
            "RESEARCH",
            "SUPPORT_CLAIM",
            "WRITE_INTRODUCTION",
            "WRITE_CONCLUSION",
            "WRITE_ABSTRACT",
            "REVIEW",
        }:
            return task_type  # type: ignore[return-value]
        # Keep the existing TaskRouter as a deterministic hint.  An ordinary
        # question that has no Skill keyword is a Research route rather than a
        # user-level writing choice.
        try:
            decision = request.metadata.get("router")
            if isinstance(decision, str) and decision in {
                "RESEARCH",
                "SUPPORT_CLAIM",
                "WRITE_INTRODUCTION",
                "WRITE_CONCLUSION",
                "WRITE_ABSTRACT",
                "REVIEW",
            }:
                return decision  # type: ignore[return-value]
            routed = request.metadata.get("task_type")
            if isinstance(routed, str) and routed in {
                "RESEARCH",
                "SUPPORT_CLAIM",
                "WRITE_INTRODUCTION",
                "WRITE_CONCLUSION",
                "WRITE_ABSTRACT",
                "REVIEW",
            }:
                return routed  # type: ignore[return-value]
            normalized = request.instruction.casefold()
            if any(
                value in normalized
                for value in ("support claim", "证明", "能否支持", "找文献")
            ):
                return "SUPPORT_CLAIM"
            if any(value in normalized for value in ("conclusion", "结论", "总结结论")):
                return "WRITE_CONCLUSION"
            if any(value in normalized for value in ("abstract", "摘要")):
                return "WRITE_ABSTRACT"
            if any(value in normalized for value in ("introduction", "引言", "intro")):
                return "WRITE_INTRODUCTION"
        except AttributeError:
            pass
        return "RESEARCH"

    def _ensure_crew(self, trace: Any) -> Any:
        with _crewai_import_guard():
            from app.orchestration.crewai.crew import ScholarCrew

        if self.crew is None:
            self.crew = ScholarCrew(
                model=self.model, gateway=self, trace=trace, max_agent_iterations=8
            )
        else:
            self.crew.trace = trace
        return self.crew

    def _consume_tool_call(self, tool_name: str) -> bool:
        """Apply the per-run capability-tool budget before domain work."""

        if self._tool_calls_used >= self.max_tool_calls:
            if self._trace is not None:
                self._trace.record(
                    "tool_budget_exceeded",
                    "FAILED",
                    {
                        "tool_name": tool_name,
                        "max_tool_calls": self.max_tool_calls,
                    },
                    kind="tool",
                )
            return False
        self._tool_calls_used += 1
        if self._trace is not None:
            self._trace.record(
                "capability_tool_call",
                "RUNNING",
                {
                    "tool_name": tool_name,
                    "tool_call_index": self._tool_calls_used,
                    "max_tool_calls": self.max_tool_calls,
                },
                kind="tool",
            )
        return True

    @contextmanager
    def _capability_tool(self, tool_name: str) -> Any:
        """Reserve and close one bounded capability invocation.

        The Flow calls domain services directly so lifecycle decisions remain
        deterministic. This adapter still records the same executable
        capability boundary that CrewAI tools use, including a paired
        completion/failure event and the run-level budget.
        """

        if not self._consume_tool_call(tool_name):
            raise SkillRuntimeError(
                "TOOL_BUDGET_EXCEEDED",
                "本次 orchestration 的 capability tool budget 已耗尽。",
            )
        try:
            if self._cancellation_checker is not None:
                self._cancellation_checker()
            yield
        except JobCancelled:
            if self._trace is not None:
                self._trace.record(
                    "capability_tool_call",
                    "CANCELLED",
                    {
                        "tool_name": tool_name,
                        "tool_call_index": self._tool_calls_used,
                    },
                    kind="tool",
                )
            raise
        except Exception as error:
            if self._trace is not None:
                self._trace.record(
                    "capability_tool_call",
                    "FAILED",
                    {
                        "tool_name": tool_name,
                        "error_type": type(error).__name__,
                        "error_code": getattr(error, "code", None),
                    },
                    kind="tool",
                )
            raise
        else:
            if self._trace is not None:
                self._trace.record(
                    "capability_tool_call",
                    "COMPLETED",
                    {
                        "tool_name": tool_name,
                        "tool_call_index": self._tool_calls_used,
                    },
                    kind="tool",
                )

    def manager_decide(self, context: Mapping[str, Any], trace: Any) -> ManagerDecision:
        """Ask the persistent Manager for one proposal, never execute it.

        Flow owns the subsequent transition validation. Keeping this method
        separate makes it impossible for a Manager Task to reach a domain
        capability directly and gives tests a narrow seam for scripted
        Manager decisions.
        """

        self._trace = trace
        output = self._ensure_crew(trace).run(
            "manager", context, ManagerDecision, task_name="manager_decision"
        )
        if not isinstance(output, ManagerDecision):
            raise SkillRuntimeError(
                "MANAGER_CONTRACT_INVALID", "Manager Agent 未返回 ManagerDecision。"
            )
        return output

    def build_manager_context(self, state: Any) -> dict[str, Any]:
        """Assemble the prioritized Manager projection for one decision.

        Only this bounded projection is sent to the Manager task. The full
        Session message table remains available to the application layer but
        is never used as a workflow progress oracle.
        """

        recent_summary: Mapping[str, Any] | None = None
        if self.session_manager is not None and self._session_run is not None:
            try:
                runtime = self.session_manager.open(self._session_run.session_id)
                messages = runtime.list_messages(limit=100_000)
                recent_summary = self._manager_context_builder.summarize_messages(messages)
            except (KeyError, OSError, ValueError):
                recent_summary = None
        project_facts: dict[str, Any] = {}
        try:
            project_facts = {
                "facts": [
                    {"key": value.key, "value": value.value}
                    for value in self.project_store.list_facts()[:16]
                ],
                "confirmed_contributions": [
                    {
                        "contribution_id": value.contribution_id,
                        "statement": value.statement,
                    }
                    for value in self.project_store.list_contributions()
                    if value.status == "confirmed" and value.confirmed_by_user
                ][:16],
                "manuscript": self.project_store.get_manuscript_state_projection(),
            }
        except (AttributeError, OSError, TypeError, ValueError):
            project_facts = {}
        return self._manager_context_builder.build_manager_context(
            original_goal=state.goal,
            run_state=state,
            project_facts=project_facts,
            recent_conversation_summary=recent_summary,
        )

    def _date_value(self, value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str) and value.strip():
            return date.fromisoformat(value[:10])
        return None

    @staticmethod
    def _evidence_from_packs(packs: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for pack in packs if isinstance(packs, (list, tuple)) else ():
            if not isinstance(pack, EvidencePack):
                continue
            result.extend(dict(item) for item in pack.evidence)
        return result

    @staticmethod
    def _citations_from_evidence(
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fields = (
            "evidence_id",
            "paper_id",
            "canonical_id",
            "source_locator",
            "bibkey",
            "citation_key",
            "metadata",
        )
        return [
            {key: item[key] for key in fields if key in item and item[key] is not None}
            for item in evidence
        ]

    @classmethod
    def _evidence_from_result(cls, value: Any) -> list[dict[str, Any]]:
        """Extract verified domain evidence without replacing its source model.

        Scholar Run storage keeps the user-facing answer separately from the
        structured domain result.  Persist the EvidencePack projection as
        evidence/citations so the Console can replay and verify it after a
        process restart.
        """

        packs: tuple[EvidencePack, ...]
        if isinstance(value, EvidencePack):
            packs = (value,)
        else:
            raw_packs = getattr(value, "evidence_packs", ())
            packs = tuple(
                item for item in raw_packs
                if isinstance(item, EvidencePack)
            ) if isinstance(raw_packs, (list, tuple)) else ()

        evidence = cls._evidence_from_packs(packs)
        for pack in packs:
            evidence.extend(dict(item) for item in pack.counter_evidence)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in evidence:
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            unique.append(item)
        return unique

    def _agent_research_output(
        self,
        context: Mapping[str, Any],
        domain_result: Any,
        packs: tuple[EvidencePack, ...],
        *,
        allow_unresolved_claims: bool = False,
    ) -> ResearchAgentOutput:
        output = self._ensure_crew(self._trace).run(
            "research", context, ResearchAgentOutput, task_name="research_specialist"
        )  # type: ignore[arg-type]
        evidence = self._evidence_from_packs(packs)
        unresolved = [claim for pack in packs for claim in pack.unresolved]
        contradictions = [
            str(value) for pack in packs for value in pack.metadata.get("conflicts", ())
        ]
        error_code = next(
            (
                str(pack.metadata.get("error_code"))
                for pack in packs
                if pack.metadata.get("error_code")
            ),
            None,
        )
        # Introduction research is a source-collection handoff, not the final
        # claim-support decision. Its generated rhetorical claims are broad
        # planning hints and exact token overlap with one retrieved chunk is
        # not a valid reason to prevent Writer from checking the real
        # 8-distinct-paper citation contract. Provider/freshness failures are
        # still fail-closed, while the Writing Service reports precise
        # N/8 coverage and refuses to create a Patch when sources are short.
        status = (
            "INSUFFICIENT_EVIDENCE"
            if error_code or (unresolved and not allow_unresolved_claims)
            else "COMPLETED"
        )
        freshness = "UNKNOWN"
        for pack in packs:
            decision = pack.metadata.get("freshness_decision")
            if isinstance(decision, Mapping):
                freshness = str(
                    decision.get("mode") or decision.get("status") or "UNKNOWN"
                )
        return output.model_copy(
            update={
                "status": status,
                "research_summary": f"Research completed with {len(evidence)} verified evidence item(s); unresolved={len(unresolved)}.",
                "evidence": evidence,
                "citations": self._citations_from_evidence(evidence),
                "unresolved_claims": unresolved,
                "contradictions": contradictions,
                "freshness_status": freshness,
                "diagnostics": {
                    "error_code": error_code,
                    "pack_count": len(packs),
                    "unresolved_count": len(unresolved),
                    "verified_evidence_count": len(evidence),
                    "verified_paper_count": len({
                        str(item.get("paper_id") or item.get("canonical_id") or item.get("work_id"))
                        for item in evidence
                        if item.get("paper_id") or item.get("canonical_id") or item.get("work_id")
                    }),
                    "paper_candidate_count": max(
                        (
                            int(pack.metadata["paper_candidate_count"])
                            for pack in packs
                            if isinstance(pack.metadata.get("paper_candidate_count"), int)
                        ),
                        default=0,
                    ),
                    "content_candidate_paper_count": max(
                        (
                            int(pack.metadata["content_candidate_paper_count"])
                            for pack in packs
                            if isinstance(pack.metadata.get("content_candidate_paper_count"), int)
                        ),
                        default=0,
                    ),
                },
                "evidence_packs": list(packs),
                "domain_result": domain_result,
            }
        )

    def research(
        self,
        request: OrchestrationRequest,
        route: Route | None,
        trace: Any,
    ) -> ResearchAgentOutput:
        if route not in {"RESEARCH", "SUPPORT_CLAIM", "WRITE_INTRODUCTION"}:
            raise SkillRuntimeError(
                "CAPABILITY_DENIED", f"Research Agent 不允许 route={route}。"
            )
        self._trace = trace
        self._active_route = route
        self._active_research_domain_result = None
        if route == "SUPPORT_CLAIM":
            with self._capability_tool("research_capability"):
                support_request = SupportClaimRequest(
                    request_id=self._run_id or request.request_id,
                    project_id=request.project_id,
                    claim=request.instruction,
                    session_id=request.session_id,
                    metadata=dict(request.metadata),
                )
                skill_result = self.skill_runtime.execute(support_request)
            domain_result = getattr(skill_result, "value", None)
            if not isinstance(domain_result, ClaimSupportResult):
                raise SkillRuntimeError(
                    "CONTRACT_VALIDATION_FAILED",
                    "Support Claim Runtime 未返回 ClaimSupportResult。",
                )
            packs = tuple(domain_result.evidence_packs)
            context = {
                "request": {"instruction": request.instruction, "route": route},
                "domain_result": domain_result,
                "verified_evidence": self._evidence_from_packs(packs),
                "capabilities": sorted(CapabilityMatrix.RESEARCH),
            }
            return self._agent_research_output(context, domain_result, packs)

        if route == "WRITE_INTRODUCTION":
            # Reuse the existing Introduction Skill's bounded research plan
            # and delegate. The handoff must contain one EvidencePack per
            # ResearchNeed; passing one aggregate pack would make the shared
            # Writing Runtime reject the request as an incomplete handoff.
            writing_request = self._writing_request(request, route)
            introduction_runtime = getattr(self.skill_runtime, "introduction", None)
            skill = getattr(introduction_runtime, "skill", None)
            if skill is None or not callable(
                getattr(skill, "build_research_needs", None)
            ):
                skill = IntroductionSkill(
                    self.project_root / "skills" / "write-introduction" / "SKILL.md"
                )
            needs = tuple(skill.build_research_needs(writing_request))
            if not needs:
                raise SkillRuntimeError(
                    "RESEARCH_PLANNING_FAILED",
                    "Introduction Skill 没有生成 ResearchNeed。",
                )
            delegate = getattr(introduction_runtime, "delegate", None)
            if delegate is None or not callable(
                getattr(delegate, "research_needs", None)
            ):
                delegate = ResearchDelegate(self.research_service)
            # Research Agent owns the Research handoff. The existing delegate
            # still decides local/web access from the registered Skill
            # capability profile and constructs canonical requests.
            capabilities = getattr(introduction_runtime, "capabilities", None)
            if capabilities is not None:
                delegate.capabilities = capabilities

            def research_progress(
                name: str, status: str, details: Mapping[str, Any]
            ) -> None:
                if self._trace is not None:
                    self._trace.record(
                        name,
                        status,
                        {
                            "agent_role": "Research Agent",
                            "summary": (
                                "研究需求进度"
                                if name == "research_need"
                                else "研究阶段进度"
                                if name == "research_stage"
                                else "论文覆盖补齐进度"
                            ),
                            **dict(details),
                        },
                        kind="research",
                    )

            with self._capability_tool("research_capability"):
                packs = tuple(
                    delegate.research_needs(
                        self._run_id or request.request_id,
                        needs,
                        workspace_id=self.research_service.workspace_id,
                        scope_version=self.research_service.scope_version,
                        request_metadata=request.metadata,
                        progress_callback=research_progress,
                        cancellation_checker=self._cancellation_checker,
                    )
                )
            if len(packs) != len(needs):
                raise SkillRuntimeError(
                    "RESEARCH_FAILED",
                    "Research Agent 未覆盖全部 Introduction ResearchNeed。",
                )
            self._active_research_domain_result = packs
            context = {
                "request": {"instruction": request.instruction, "route": route},
                "research_needs": needs,
                "domain_result": packs,
                "verified_evidence": self._evidence_from_packs(packs),
                "capabilities": sorted(CapabilityMatrix.RESEARCH),
            }
            return self._agent_research_output(
                context,
                packs,
                packs,
                # An unresolved broad planning claim may coexist with useful
                # sources, but an entirely empty pack must still fail closed.
                allow_unresolved_claims=any(pack.evidence for pack in packs),
            )

        target_claims = tuple(
            str(value) for value in request.metadata.get("target_claims", ()) if value
        ) or (request.instruction,)
        allow_web = bool(request.metadata.get("allow_web", False))
        research_request = ResearchRequest(
            request_id=self._run_id or request.request_id,
            query=request.instruction,
            workspace_id=self.research_service.workspace_id,
            scope_version=self.research_service.scope_version,
            purpose="background" if route == "RESEARCH" else "prior_work",
            target_claims=target_claims,
            freshness_mode=str(request.metadata.get("freshness_mode") or "LOCAL_ONLY"),  # type: ignore[arg-type]
            requested_from=self._date_value(request.metadata.get("requested_from")),
            requested_to=self._date_value(request.metadata.get("requested_to")),
            explicit_latest=bool(request.metadata.get("explicit_latest", False)),
            budget=ResearchBudget(max_web_queries=1 if allow_web else 0),
        )
        with self._capability_tool("research_capability"):
            pack = self.research_service.research(
                research_request, allow_web=allow_web
            )
        self._active_research_domain_result = pack
        context = {
            "request": {"instruction": request.instruction, "route": route},
            "domain_result": pack,
            "verified_evidence": list(pack.evidence),
            "capabilities": sorted(CapabilityMatrix.RESEARCH),
        }
        return self._agent_research_output(context, pack, (pack,))

    def _writing_request(
        self, request: OrchestrationRequest, route: Route
    ) -> WritingRequest:
        section = {
            "WRITE_INTRODUCTION": "introduction",
            "WRITE_CONCLUSION": "conclusion",
            "WRITE_ABSTRACT": "abstract",
        }.get(route)
        if section is None:
            raise SkillRuntimeError("UNSUPPORTED_ROUTE", f"不是写作 route：{route}")
        # Introduction research is executed before the Writer capability. A
        # brand-new project therefore needs its empty LaTeX skeleton prepared
        # at the route boundary, otherwise a valid research result can never
        # reach DraftPatch generation. This operation is additive only; it
        # never writes Agent prose or bypasses human approval.
        ManuscriptSynchronizer(self.project_root).ensure_initialized()
        metadata = dict(request.metadata)
        metadata["run_id"] = self._run_id or request.request_id
        if route == "WRITE_INTRODUCTION":
            # Lower caller values cannot weaken the Introduction citation
            # contract. Higher values are allowed for broader surveys.
            try:
                requested_minimum = int(metadata.get("minimum_unique_papers", INTRODUCTION_MINIMUM_UNIQUE_PAPERS))
            except (TypeError, ValueError):
                requested_minimum = INTRODUCTION_MINIMUM_UNIQUE_PAPERS
            metadata["minimum_unique_papers"] = max(INTRODUCTION_MINIMUM_UNIQUE_PAPERS, requested_minimum)
            for key, floor in (
                ("paper_retrieval_limit", INTRODUCTION_PAPER_RETRIEVAL_LIMIT),
                ("section_retrieval_limit", INTRODUCTION_SECTION_RETRIEVAL_LIMIT),
                ("evidence_limit", INTRODUCTION_EVIDENCE_LIMIT),
            ):
                try:
                    metadata[key] = max(floor, int(metadata.get(key, floor)))
                except (TypeError, ValueError):
                    metadata[key] = floor
        return WritingRequest(
            request_id=self._run_id or request.request_id,
            project_id=request.project_id,
            instruction=request.instruction,
            target_section=section,
            session_id=request.session_id,
            task_type=route,  # type: ignore[arg-type]
            metadata=metadata,
        )

    def _execute_write_capability(
        self,
        request: OrchestrationRequest,
        route: Route,
        research: ResearchAgentOutput | None,
    ) -> tuple[WritingRequest, Any] | None:
        """Execute only the existing Writing/Skill Runtime capability.

        This helper is shared by the Flow path and the CrewAI Writer tool.
        It deliberately stops before the Writer Agent structured-output step,
        so a tool call cannot recursively start another Crew.
        """

        writing_request = self._writing_request(request, route)
        if route == "WRITE_INTRODUCTION" and research is None:
            return None
        if route == "WRITE_INTRODUCTION" and research is not None:
            writing_request = replace(
                writing_request,
                metadata={
                    **dict(writing_request.metadata),
                    "_precomputed_evidence_packs": tuple(research.evidence_packs),
                },
            )
        writer = self.writers.get(route)
        if writer is None:
            writer = self.writers.get(writing_request.target_section)
        if writer is None:
            return (writing_request, None)
        set_trace = getattr(writer, "set_trace", None)
        if callable(set_trace):
            set_trace(self._trace)
        skill_result = self.skill_runtime.execute(writing_request, writer=writer)
        return writing_request, getattr(skill_result, "value", None)

    def write(
        self,
        request: OrchestrationRequest,
        route: Route | None,
        research: ResearchAgentOutput | None,
        trace: Any,
        *,
        review_round: int = 0,
    ) -> WriterAgentOutput:
        if route not in {"WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            raise SkillRuntimeError(
                "CAPABILITY_DENIED", f"Writer Agent 不允许 route={route}。"
            )
        self._trace = trace
        self._active_route = route
        self._active_research = research
        self._active_writer_domain_result = None
        self._active_review_round = review_round
        if route == "WRITE_INTRODUCTION" and research is None:
            return WriterAgentOutput(
                status="RESEARCH_REQUIRED",
                research_required=True,
                missing_context=["VERIFIED_EVIDENCE"],
            )
        try:
            with self._capability_tool("writer_capability"):
                capability_result = self._execute_write_capability(request, route, research)
        except FileNotFoundError:
            # A production Scholar project may legitimately contain papers
            # but no LaTeX manuscript yet.  Keep that boundary explicit and
            # machine-readable instead of turning the missing root .tex into
            # a generic WRITING_FAILED result.
            return WriterAgentOutput(
                status="INSUFFICIENT_MANUSCRIPT_STATE",
                research_required=False,
                missing_context=["INSUFFICIENT_MANUSCRIPT_STATE"],
            )
        except SkillRuntimeError as error:
            return WriterAgentOutput(
                status="FAILED",
                research_required=False,
                missing_context=[error.code],
            )
        if capability_result is None:
            return WriterAgentOutput(
                status="RESEARCH_REQUIRED",
                research_required=True,
                missing_context=["VERIFIED_EVIDENCE"],
            )
        _, domain_result = capability_result
        self._active_writer_domain_result = domain_result
        if domain_result is None:
            return WriterAgentOutput(
                status="FAILED", missing_context=["WRITER_NOT_INJECTED"]
            )
        patch = getattr(domain_result, "patch", None)
        domain_status = str(getattr(domain_result, "status", "FAILED"))
        errors = tuple(
            str(value) for value in getattr(domain_result, "error_codes", ()) or ()
        )
        if patch is None or domain_status == "FAILED":
            return WriterAgentOutput(
                status="INSUFFICIENT_MANUSCRIPT_STATE"
                if "INSUFFICIENT_MANUSCRIPT_STATE" in errors
                else "FAILED",
                research_required=False,
                missing_context=list(errors or ("DRAFT_PATCH_NOT_CREATED",)),
                warnings=list(getattr(domain_result, "warnings", ()) or ()),
                domain_result=domain_result,
            )
        evidence = list(getattr(domain_result, "evidence_packs", ()) or ())
        context = {
            "request": {
                "instruction": request.instruction,
                "route": route,
                "review_round": review_round,
            },
            "domain_result": domain_result,
            "draft_patch": patch,
            "verified_evidence": self._evidence_from_packs(evidence),
            "capabilities": sorted(CapabilityMatrix.WRITER),
        }
        output = self._ensure_crew(trace).run(
            "writer", context, WriterAgentOutput, task_name="writer_specialist"
        )
        return output.model_copy(
            update={
                "status": "READY",
                "draft_patch": patch,
                "research_required": False,
                "missing_context": [],
                "used_evidence": self._evidence_from_packs(evidence),
                "warnings": list(getattr(domain_result, "warnings", ()) or ()),
                "domain_result": domain_result,
            }
        )

    @staticmethod
    def _review_report_context(report: ReviewReport) -> dict[str, Any]:
        return {
            "valid": report.valid,
            "issues": [_jsonable(issue) for issue in report.issues],
            "revision_round": report.revision_round,
            "report_id": report.report_id,
        }

    def review(
        self,
        request: OrchestrationRequest,
        route: Route | None,
        writer: WriterAgentOutput | None,
        trace: Any,
        *,
        review_round: int = 0,
    ) -> ReviewAgentOutput:
        if (
            route not in {"WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}
            or writer is None
        ):
            raise SkillRuntimeError(
                "CAPABILITY_DENIED", "Reviewer Agent 只接收 Writer 的 DraftPatch。"
            )
        self._trace = trace
        self._active_route = route
        self._active_review_round = review_round
        with self._capability_tool("review_capability"):
            domain_result = writer.domain_result
            report = getattr(domain_result, "review_report", None)
            if not isinstance(report, ReviewReport):
                raise SkillRuntimeError(
                    "REVIEW_FAILED", "Writing Runtime 未返回 ReviewReport。"
                )
        patch = writer.draft_patch
        expected = "PASS" if report.valid else "REVISE"
        context = {
            "draft_patch": patch,
            "review_report": self._review_report_context(report),
            "review_round": review_round,
            "capabilities": sorted(CapabilityMatrix.REVIEWER),
        }
        output = self._ensure_crew(trace).run(
            "reviewer", context, ReviewAgentOutput, task_name="reviewer_specialist"
        )
        if output.decision != expected:
            raise SkillRuntimeError(
                "CONTRACT_VALIDATION_FAILED",
                f"Reviewer Agent decision={output.decision} 与 deterministic ReviewReport={expected} 不一致。",
            )
        issues = [_jsonable(issue) for issue in report.issues]
        return output.model_copy(
            update={
                "decision": expected,
                "issues": issues,
                "unsupported_claims": [
                    str(issue.get("claim_id"))
                    for issue in issues
                    if issue.get("claim_id")
                ],
                "fact_conflicts": [
                    str(issue.get("code"))
                    for issue in issues
                    if "FACT" in str(issue.get("code", ""))
                ],
                "revision_instructions": [
                    str(issue.get("message"))
                    for issue in issues
                    if expected == "REVISE"
                ],
                "review_report": report,
            }
        )

    def review_existing(
        self, request: OrchestrationRequest, trace: Any
    ) -> ReviewAgentOutput | None:
        patch_id = request.metadata.get("patch_id")
        if not isinstance(patch_id, str) or not patch_id.strip():
            return None
        try:
            with self._capability_tool("review_capability"):
                stored = self.project_store.get_patch(patch_id)
                report = getattr(stored, "review_report", None)
                if not isinstance(report, ReviewReport):
                    return None
        except KeyError:
            return None
        context = {
            "patch_id": patch_id,
            "review_report": self._review_report_context(report),
            "capabilities": sorted(CapabilityMatrix.REVIEWER),
        }
        output = self._ensure_crew(trace).run(
            "reviewer", context, ReviewAgentOutput, task_name="review_existing_patch"
        )
        expected = "PASS" if report.valid else "REVISE"
        if output.decision != expected:
            raise SkillRuntimeError(
                "CONTRACT_VALIDATION_FAILED",
                "Reviewer Agent 与 stored ReviewReport 不一致。",
            )
        return output.model_copy(
            update={
                "decision": expected,
                "issues": [_jsonable(issue) for issue in report.issues],
                "review_report": report,
            }
        )

    # Tool methods below are deliberately narrow.  Flow calls the same methods
    # directly so deterministic lifecycle decisions never depend on an LLM
    # choosing a tool, while the tools remain available to the specialist.
    def tool_research(self, **kwargs: Any) -> Mapping[str, Any]:
        request = self._request
        if request is None:
            return {
                "status": "FAILED",
                "error_code": "REQUEST_CONTEXT_REQUIRED",
                "message": "ResearchCapabilityTool 必须由 Flow 绑定 OrchestrationRequest。",
            }
        if not self._consume_tool_call("research_capability"):
            return {
                "status": "FAILED",
                "error_code": "TOOL_BUDGET_EXCEEDED",
                "message": "本次 orchestration 的 capability tool budget 已耗尽。",
            }
        requested_query = str(kwargs.get("query") or request.instruction)
        if (
            self._active_research_domain_result is not None
            and self._active_route != "SUPPORT_CLAIM"
            and requested_query == request.instruction
        ):
            return {
                "status": "COMPLETED",
                "result_type": "EvidencePack",
                "result": _jsonable(self._active_research_domain_result),
                "cached": True,
            }
        try:
            query = requested_query
            research_request = ResearchRequest(
                request_id=f"{self._run_id or request.request_id}:tool",
                query=query,
                workspace_id=self.research_service.workspace_id,
                scope_version=self.research_service.scope_version,
                purpose=str(kwargs.get("purpose") or "background"),  # type: ignore[arg-type]
                target_claims=(str(kwargs.get("target_claim") or query),),
                freshness_mode=str(kwargs.get("freshness_mode") or "LOCAL_ONLY"),  # type: ignore[arg-type]
                requested_from=self._date_value(kwargs.get("requested_from")),
                requested_to=self._date_value(kwargs.get("requested_to")),
                explicit_latest=bool(kwargs.get("explicit_latest", False)),
                budget=ResearchBudget(
                    max_web_queries=1 if request.metadata.get("allow_web") else 0
                ),
            )
            pack = self.research_service.research(
                research_request,
                allow_web=bool(request.metadata.get("allow_web", False)),
            )
            return {
                "status": "COMPLETED",
                "result_type": "EvidencePack",
                "result": _jsonable(pack),
            }
        except Exception as error:
            return {
                "status": "FAILED",
                "error_code": "RESEARCH_FAILED",
                "message": str(error)[:500],
            }

    def tool_write(self, **kwargs: Any) -> Mapping[str, Any]:
        request = self._request
        if request is None:
            return {
                "status": "FAILED",
                "error_code": "REQUEST_CONTEXT_REQUIRED",
                "message": "WriterCapabilityTool 必须由 Flow 绑定 OrchestrationRequest。",
            }
        if not self._consume_tool_call("writer_capability"):
            return {
                "status": "FAILED",
                "error_code": "TOOL_BUDGET_EXCEEDED",
                "message": "本次 orchestration 的 capability tool budget 已耗尽。",
            }
        route = self._active_route or request.task_type
        if route not in {
            "WRITE_INTRODUCTION",
            "WRITE_CONCLUSION",
            "WRITE_ABSTRACT",
        }:
            return {
                "status": "FAILED",
                "error_code": "CAPABILITY_DENIED",
                "message": f"Writer Agent 不允许 route={route!r}。",
            }
        try:
            domain_result = self._active_writer_domain_result
            if domain_result is None:
                capability_result = self._execute_write_capability(
                    request, route, self._active_research
                )
                if capability_result is None:
                    return {
                        "status": "FAILED",
                        "error_code": "RESEARCH_REQUIRED",
                        "message": "Introduction Writer 缺少 Research Agent 交付的 verified evidence。",
                    }
                _, domain_result = capability_result
                self._active_writer_domain_result = domain_result
            if domain_result is None:
                return {
                    "status": "FAILED",
                    "error_code": "WRITER_NOT_INJECTED",
                    "message": "Writer capability 未注入。",
                }
            patch = getattr(domain_result, "patch", None)
            return {
                "status": str(getattr(domain_result, "status", "FAILED")),
                "result_type": "WritingResult",
                "draft_patch": _jsonable(patch),
                "result": _jsonable(domain_result),
            }
        except SkillRuntimeError as error:
            return {
                "status": "FAILED",
                "error_code": error.code,
                "message": str(error)[:500],
            }
        except Exception as error:
            return {
                "status": "FAILED",
                "error_code": "WRITING_FAILED",
                "message": str(error)[:500],
            }

    def tool_review(self, **kwargs: Any) -> Mapping[str, Any]:
        request = self._request
        if request is None:
            return {
                "status": "FAILED",
                "error_code": "REQUEST_CONTEXT_REQUIRED",
                "message": "ReviewerCapabilityTool 必须由 Flow 绑定 OrchestrationRequest。",
            }
        if not self._consume_tool_call("review_capability"):
            return {
                "status": "FAILED",
                "error_code": "TOOL_BUDGET_EXCEEDED",
                "message": "本次 orchestration 的 capability tool budget 已耗尽。",
            }
        patch_id = kwargs.get("patch_id")
        if not isinstance(patch_id, str) or not patch_id.strip():
            return {
                "status": "FAILED",
                "error_code": "PATCH_ID_REQUIRED",
                "message": "Reviewer capability 需要 patch_id。",
            }
        try:
            stored = self.project_store.get_patch(patch_id)
        except KeyError:
            return {
                "status": "FAILED",
                "error_code": "PATCH_NOT_FOUND",
                "message": f"Patch 不存在：{patch_id}",
            }
        patch = getattr(stored, "patch", None)
        if getattr(patch, "project_id", request.project_id) != request.project_id:
            return {
                "status": "FAILED",
                "error_code": "PROJECT_CONFLICT",
                "message": "Patch 不属于当前 project。",
            }
        report = getattr(stored, "review_report", None)
        if not isinstance(report, ReviewReport):
            return {
                "status": "FAILED",
                "error_code": "REVIEW_REPORT_REQUIRED",
                "message": "Patch 没有可供 Reviewer 消费的 deterministic ReviewReport。",
            }
        return {
            "status": "PASS" if report.valid else "REVISE",
            "result_type": "ReviewReport",
            "patch_id": patch_id,
            "review_report": _jsonable(report),
        }

    def persist_result(self, result: OrchestrationResult) -> None:
        self._flow_result = result
        if self.session_manager is None or self._session_run is None:
            return
        try:
            runtime = self.session_manager.open(result.session_id)
            session_status = (
                "WAITING_USER"
                if result.status == "WAITING_HUMAN_APPROVAL"
                else "FAILED"
                if result.status == "FAILED"
                else "INTERRUPTED"
                if result.status == "INTERRUPTED"
                else "COMPLETED"
            )
            value = result.final_answer or (
                json.dumps(_jsonable(result.value), ensure_ascii=False, default=str)
                if result.value is not None
                else ""
            )
            persisted_evidence = self._evidence_from_result(result.value)
            persisted_citations = self._citations_from_evidence(persisted_evidence)
            metadata = {
                "backend": "crewai",
                "task_type": result.selected_route,
                "selected_skill": result.selected_route,
                "result_type": (
                    type(result.value).__name__ if result.value is not None else None
                ),
                "orchestration_status": result.status,
                "approval_required": result.approval_required,
                "pending_action": _jsonable(result.pending_action),
                "approval_patch_id": (
                    result.pending_action.get("patch_id")
                    if isinstance(result.pending_action, Mapping)
                    else None
                ),
                "error_codes": result.error_codes,
                "flow_state": result.diagnostics.get("flow_state"),
                "trace": result.diagnostics.get("trace", {}),
                "visible_capabilities": CapabilityMatrix.as_dict(),
                "structured_result": _jsonable(result.value),
            }
            runtime.complete_run(
                result.run_id,
                status=session_status,
                answer=value[:50_000],
                citations=persisted_citations,
                evidence=persisted_evidence,
                metadata=metadata,
            )
            runtime.clear_checkpoint(result.run_id)
            if self.event_store is not None:
                event_type = {
                    "WAITING_HUMAN_APPROVAL": "WAITING_USER",
                    "COMPLETED": "RUN_COMPLETED",
                    "INTERRUPTED": "RUN_INTERRUPTED",
                }.get(result.status, "RUN_FAILED")
                event_status = {
                    "WAITING_USER": "WAITING_USER",
                    "RUN_COMPLETED": "COMPLETED",
                    "RUN_INTERRUPTED": "INTERRUPTED",
                    "RUN_FAILED": "FAILED",
                }[event_type]
                self.event_store.append(
                    run_id=result.run_id,
                    project_id=result.project_id,
                    session_id=result.session_id,
                    event_type=event_type,
                    node="Human Approval" if result.approval_required else "Scholar Run",
                    status=event_status,
                    summary=(
                        "Run waiting for human approval"
                        if result.approval_required
                        else "Run completed"
                        if result.status == "COMPLETED"
                        else "Run failed"
                    ),
                    metadata={
                        "backend": "crewai",
                        "trace_id": result.trace_id,
                        "error_codes": result.error_codes,
                        "termination_reason": result.diagnostics.get("termination_reason"),
                    },
                    event_id=f"{result.run_id}:terminal:{event_type}",
                )
            self.session_manager.set_active_run(result.session_id, None)
        except (KeyError, OSError, ValueError, TypeError):
            # Domain result remains authoritative; a failed projection is
            # recorded in diagnostics by the caller rather than changing it.
            return

    def persist_checkpoint(self, state: Any) -> None:
        """Persist the bounded Manager/Flow projection in Session Runtime."""

        if self.session_manager is None or self._session_run is None:
            return
        try:
            runtime = self.session_manager.open(self._session_run.session_id)
            payload = state.model_dump(mode="json") if hasattr(state, "model_dump") else _jsonable(state)
            # A checkpoint is a replay boundary, not a second domain store.
            payload = {
                key: value
                for key, value in dict(payload).items()
                if key not in {"request"} or key == "request"
            }
            runtime.save_checkpoint(self._session_run.run_id, payload)
        except (KeyError, OSError, ValueError, TypeError):
            # A telemetry/checkpoint projection failure must not mutate the
            # authoritative domain result or turn a safe run into a write.
            return

    def load_checkpoint(self, run_id: str) -> Mapping[str, Any] | None:
        if self.session_manager is None or self._session_run is None:
            return None
        try:
            runtime = self.session_manager.open(self._session_run.session_id)
            return runtime.load_checkpoint(run_id)
        except (KeyError, OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _route_from_metadata(metadata: Mapping[str, Any]) -> Route | None:
        value = metadata.get("task_type") or metadata.get("selected_skill")
        allowed = {
            "RESEARCH",
            "SUPPORT_CLAIM",
            "WRITE_INTRODUCTION",
            "WRITE_CONCLUSION",
            "WRITE_ABSTRACT",
            "REVIEW",
        }
        return value if value in allowed else None  # type: ignore[return-value]

    def _find_session_run(
        self, thread_id: str, session_id: str | None, project_id: str
    ) -> tuple[Any, Any]:
        if self.session_manager is None:
            raise SkillRuntimeError(
                "RESUME_UNAVAILABLE",
                "CrewAI Approval Resume 需要 Session Runtime。",
            )
        sessions = (
            [self.session_manager.get(session_id)]
            if session_id
            else self.session_manager.list(include_deleted=False)
        )
        for session in sessions:
            if session.project_id and session.project_id != project_id:
                continue
            runtime = self.session_manager.open(session.session_id)
            for run in runtime.list_runs():
                if (
                    run.thread_id == thread_id
                    and (not run.project_id or run.project_id == project_id)
                ):
                    return runtime, run
        raise SkillRuntimeError(
            "RESUME_UNAVAILABLE",
            f"没有找到可恢复的 CrewAI Run：{thread_id}",
        )

    def resume(self, thread_id: str, resume_value: Any, project_id: str, *, instruction: str, session_id: str | None = None, task_type: str | None = None) -> OrchestrationResult:
        # CrewAI Flow is not a second durable store. The human decision is
        # made by PatchApprovalService; this method only reconciles its
        # persisted Patch status with the original Session Run.
        del resume_value, instruction
        if not thread_id.strip() or not project_id.strip():
            raise ValueError("thread_id/project_id 不能为空。")
        runtime, run = self._find_session_run(thread_id, session_id, project_id)
        if run.status == "INTERRUPTED" and runtime.load_checkpoint(run.run_id):
            # Interrupted CrewAI Runs resume through the same deterministic
            # Flow and Run identity. ``resume_value`` is never treated as an
            # approval signal; it is intentionally ignored here.
            return self.run(
                OrchestrationRequest(
                    request_id=f"resume:{run.run_id}",
                    project_id=project_id,
                    instruction=run.query,
                    session_id=run.session_id,
                    thread_id=run.thread_id,
                    task_type=task_type,
                    metadata={"_run_id": run.run_id, "_worker_id": "resume"},
                )
            )
        result_rows = {
            str(value["run_id"]): value
            for value in runtime.list_results()
            if isinstance(value, Mapping) and value.get("run_id")
        }
        stored_result = result_rows.get(run.run_id, {})
        metadata = (
            dict(stored_result.get("metadata", {}))
            if isinstance(stored_result.get("metadata"), Mapping)
            else {}
        )
        allowed_routes = {
            "RESEARCH",
            "SUPPORT_CLAIM",
            "WRITE_INTRODUCTION",
            "WRITE_CONCLUSION",
            "WRITE_ABSTRACT",
            "REVIEW",
        }
        route = (
            task_type
            if task_type in allowed_routes
            else self._route_from_metadata(metadata)
        )  # type: ignore[assignment]
        pending = metadata.get("pending_action")
        patch_id = (
            pending.get("patch_id")
            if isinstance(pending, Mapping)
            else metadata.get("approval_patch_id")
        )
        stored_patch = None
        if isinstance(patch_id, str) and patch_id:
            try:
                stored_patch = self.project_store.get_patch(patch_id)
            except KeyError:
                stored_patch = None
        if stored_patch is None:
            for candidate in self.project_store.list_patches():
                if getattr(candidate, "source_run_id", None) == run.run_id:
                    stored_patch = candidate
                    break
        if stored_patch is None:
            raise SkillRuntimeError(
                "RESUME_UNAVAILABLE",
                "等待中的 CrewAI Run 没有对应的 DraftPatch。",
            )

        patch = getattr(stored_patch, "patch", None)
        resolved_patch_id = str(getattr(patch, "patch_id", patch_id or ""))
        expected_base_hash = str(getattr(patch, "base_hash", ""))
        patch_status = str(getattr(stored_patch, "status", "")).upper()
        next_status = "WAITING_HUMAN_APPROVAL"
        session_status = "WAITING_USER"
        error_codes: tuple[str, ...] = ()
        if patch_status == "APPLIED":
            next_status = "COMPLETED"
            session_status = "COMPLETED"
            answer = str(stored_result.get("answer") or "")
            message = "DraftPatch 已由 Human Approval 应用。"
        elif patch_status == "REJECTED":
            next_status = "COMPLETED"
            session_status = "COMPLETED"
            error_codes = ("HUMAN_REJECTED",)
            answer = "DraftPatch 已被人工拒绝，Manuscript 未修改。"
            message = answer
        elif patch_status in {"CONFLICT", "FAILED", "BUILD_FAILED"}:
            next_status = "FAILED"
            session_status = "FAILED"
            error_codes = (f"PATCH_{patch_status}",)
            answer = "DraftPatch 未应用，Manuscript 未被覆盖。"
            message = answer
        else:
            answer = str(stored_result.get("answer") or "")
            message = "DraftPatch 仍等待人工审批。"

        trace = metadata.get("trace")
        if isinstance(trace, Mapping):
            trace = dict(trace)
            events = list(trace.get("events") or trace.get("trace") or ())
            events.append(
                {
                    "name": "human_approval_reconciled",
                    "kind": "approval",
                    "status": (
                        "COMPLETED"
                        if session_status == "COMPLETED"
                        else session_status
                    ),
                    "metadata": {
                        "patch_id": resolved_patch_id,
                        "patch_status": patch_status,
                        "project_id": project_id,
                        "session_id": run.session_id,
                        "run_id": run.run_id,
                        "trace_id": run.trace_id,
                    },
                }
            )
            trace["events"] = events
            trace["trace"] = events
            metadata["trace"] = trace
        metadata.update(
            {
                "approval_reconciled": True,
                "approval_patch_status": patch_status,
                "approval_patch_id": resolved_patch_id,
                "approval_required": session_status == "WAITING_USER",
                "pending_action": (
                    {
                        "type": "HUMAN_APPROVAL",
                        "patch_id": resolved_patch_id,
                        "expected_base_hash": expected_base_hash,
                        "safe_apply": "PatchApprovalService only",
                    }
                    if session_status == "WAITING_USER"
                    else None
                ),
                "orchestration_status": next_status,
                "error_codes": list(error_codes),
            }
        )
        runtime.complete_run(
            run.run_id,
            status=session_status,  # type: ignore[arg-type]
            answer=answer[:50_000] or message,
            citations=[],
            evidence=[],
            metadata=metadata,
        )
        if self.event_store is not None:
            event_type = "WAITING_USER" if session_status == "WAITING_USER" else "RUN_COMPLETED" if session_status == "COMPLETED" else "RUN_FAILED"
            event_status = "WAITING_USER" if session_status == "WAITING_USER" else "COMPLETED" if session_status == "COMPLETED" else "FAILED"
            self.event_store.append(
                run_id=run.run_id,
                project_id=project_id,
                session_id=run.session_id,
                event_type=event_type,
                node="Human Approval" if session_status == "WAITING_USER" else "Scholar Run",
                status=event_status,
                summary=message,
                metadata={"approval_patch_status": patch_status, "trace_id": run.trace_id},
                event_id=f"{run.run_id}:terminal:{event_type}",
            )
        if session_status != "WAITING_USER":
            self.session_manager.set_active_run(run.session_id, None)
        pending_action = metadata["pending_action"]
        return OrchestrationResult(
            status=next_status,  # type: ignore[arg-type]
            request_id=run.run_id,
            project_id=project_id,
            session_id=run.session_id,
            run_id=run.run_id,
            thread_id=thread_id,
            trace_id=run.trace_id,
            selected_route=route,
            final_answer=answer or message,
            pending_action=pending_action,
            specialist_results={"approval": _jsonable(stored_patch)},
            approval_required=pending_action is not None,
            error_codes=list(error_codes),
            diagnostics={
                "approval_reconciled": True,
                "patch_status": patch_status,
                "message": message,
            },
            value=stored_patch,
            backend="crewai",
        )

    def _new_run_backend(self) -> "CrewAIBackend":
        """Create a run-scoped adapter with no shared mutable Run state.

        Domain services and stores are shared immutable dependencies, but the
        Crew, trace, request, cancellation callback and specialist handoffs
        belong exclusively to one Run. This is the boundary that makes a
        single Runtime safe for concurrent users.
        """

        return CrewAIBackend(
            self.project_root,
            model=self.model,
            skill_runtime=self.skill_runtime,
            research=self.research_service,
            writers=self.writers,
            reviewer=self.reviewer,
            project_store=self.project_store,
            session_manager=self.session_manager,
            event_store=self.event_store,
            max_review_rounds=self.max_review_rounds,
            max_research_attempts=self.max_research_attempts,
            max_manager_steps=self.max_manager_steps,
            max_tool_calls=self.max_tool_calls,
            max_token_budget=self.max_token_budget,
        )

    def run(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute one Run on a private backend context.

        The public backend is a dependency holder only. Never execute the
        mutable Flow state on the shared application-level instance.
        """

        return self._new_run_backend()._run_impl(request)

    def _run_impl(self, request: OrchestrationRequest) -> OrchestrationResult:
        raw_cancellation_checker = request.metadata.get("_cancellation_checker")
        self._cancellation_checker = (
            raw_cancellation_checker if callable(raw_cancellation_checker) else None
        )
        if self._cancellation_checker is not None:
            request = request.model_copy(
                update={
                    "metadata": {
                        key: value
                        for key, value in request.metadata.items()
                        if key != "_cancellation_checker"
                    }
                }
            )
        self._request = request
        trace_id = f"TRACE_{secrets.token_hex(8)}"
        self._tool_calls_used = 0
        self._active_route = None
        self._active_research = None
        self._active_research_domain_result = None
        self._active_writer_domain_result = None
        self._active_review_round = 0
        self._run_goal = None
        session_id, run_id, thread_id = self._prepare_run(
            request, trace_id=trace_id
        )
        self._run_id = run_id
        if self._session_run is not None and getattr(self._session_run, "trace_id", None):
            trace_id = str(self._session_run.trace_id)
        with _crewai_import_guard():
            from app.orchestration.crewai.flow import CrewAIOrchestrationFlow, FlowState
            from app.orchestration.crewai.tracing import CrewAITraceAdapter

        trace = CrewAITraceAdapter(
            project_id=request.project_id,
            session_id=session_id,
            run_id=run_id,
            trace_id=trace_id,
            event_sink=(
                lambda event: self.event_store.append_trace(
                    event,
                    run_id=run_id,
                    project_id=request.project_id,
                    session_id=session_id,
                )
                if self.event_store is not None
                else None
            ),
        )
        self._trace = trace
        request_with_run = request.model_copy(
            update={"session_id": session_id, "thread_id": thread_id}
        )
        persisted_goal = self._run_goal or default_run_goal(
            request.instruction, self._route_hint(request)
        )
        state_payload: dict[str, Any] = {
            "project_id": request.project_id,
            "session_id": session_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "trace_id": trace_id,
            "request": request_with_run,
            "selected_route": self._route_hint(request),
            "goal": persisted_goal,
            "budget": RunBudget(
                max_manager_steps=self.max_manager_steps,
                max_research_rounds=self.max_research_attempts,
                max_review_rounds=max(1, self.max_review_rounds + 1),
                max_tool_calls=self.max_tool_calls,
                max_tokens=self.max_token_budget,
            ),
            "user_preferences": [
                str(value)
                for value in request.metadata.get("local_preferences", ())
                if isinstance(value, str) and value.strip()
            ][-8:],
        }
        checkpoint = self.load_checkpoint(run_id)
        if checkpoint:
            # Restore only the bounded Flow projection. Domain objects are
            # deliberately replayed from the deterministic capability boundary
            # after an interruption instead of being copied into checkpoint
            # state.
            checkpoint_state = checkpoint.get("run_state", checkpoint)
            if not isinstance(checkpoint_state, Mapping):
                checkpoint_state = {}
            for key in (
                "stage", "status", "current_agent", "manager_steps", "research_round",
                "review_calls", "research_status", "writer_ready", "review_decision",
                "last_agent", "last_specialist_status", "completed_steps", "pending_gaps",
                "success_criteria_status", "evidence_references", "research_result_reference",
                "domain_result_refs", "draft_patch_id", "draft_patch_reference",
                "latest_specialist_result", "last_specialist_result", "review_round",
                "pending_approval", "manager_decisions", "error_codes", "tool_calls",
                "last_progress_fingerprint", "no_progress_rounds", "user_preferences",
            ):
                if key in checkpoint_state:
                    state_payload[key] = checkpoint_state[key]
            state_payload["request"] = request_with_run
            state_payload["selected_route"] = checkpoint_state.get("selected_route") or self._route_hint(request)
            state_payload["user_preferences"] = list(
                dict.fromkeys(
                    list(checkpoint_state.get("user_preferences") or [])
                    + list(state_payload.get("user_preferences") or [])
                )
            )[-8:]
            if isinstance(checkpoint.get("run_goal"), Mapping):
                stored_goal = RunGoal.model_validate(checkpoint["run_goal"])
                # The persisted row is authoritative. The checkpoint goal is
                # only a consistency assertion, never an overwrite source.
                if stored_goal != persisted_goal:
                    trace.record(
                        "goal_checkpoint_mismatch",
                        "FAILED",
                        {"run_id": run_id},
                        kind="checkpoint",
                    )
            trace.record(
                "manager_checkpoint_loaded",
                "COMPLETED",
                {"checkpoint_ref": getattr(self._session_run, "checkpoint_ref", None)},
                kind="checkpoint",
            )
        state = FlowState.model_validate(state_payload)
        flow = CrewAIOrchestrationFlow(
            backend=self,
            state=state,
            trace=trace,
            max_research_attempts=self.max_research_attempts,
            max_review_rounds=self.max_review_rounds,
            max_manager_steps=self.max_manager_steps,
        )
        try:
            result = flow.kickoff()
            if not isinstance(result, OrchestrationResult):
                result = self._flow_result or OrchestrationResult(
                    status="FAILED",
                    request_id=request.request_id,
                    project_id=request.project_id,
                    session_id=session_id,
                    run_id=run_id,
                    thread_id=thread_id,
                    trace_id=trace_id,
                    error_codes=["FLOW_RESULT_INVALID"],
                    backend="crewai",
                )
            return result
        except JobCancelled:
            trace.record(
                "run_cancelled",
                "CANCELLED",
                {"termination_reason": "CANCELLED_BY_USER"},
                kind="flow",
            )
            raise
        except Exception as error:
            trace.record(
                "flow_failed",
                "FAILED",
                {"error_type": type(error).__name__, "error": str(error)[:500]},
                kind="flow",
            )
            result = OrchestrationResult(
                status="FAILED",
                request_id=request.request_id,
                project_id=request.project_id,
                session_id=session_id,
                run_id=run_id,
                thread_id=thread_id,
                trace_id=trace_id,
                selected_route=self._route_hint(request),
                error_codes=["ORCHESTRATION_FAILED"],
                diagnostics={"error": str(error)[:500], "trace": trace.diagnostics()},
                backend="crewai",
            )
            self.persist_result(result)
            return result

    def _prepare_run(
        self, request: OrchestrationRequest, *, trace_id: str
    ) -> tuple[str, str, str]:
        if self.session_manager is None:
            run_id = f"RUN_{secrets.token_hex(8)}"
            return (
                request.session_id or f"SESSION_{secrets.token_hex(8)}",
                run_id,
                request.thread_id or f"crew_{run_id}",
            )
        session = self.session_manager.resolve(
            request.session_id,
            title=request.instruction[:120],
            project_id=request.project_id,
            tenant_id=request.tenant_id,
            principal_id=request.principal_id,
        )
        runtime = self.session_manager.open(session.session_id)
        requested_run_id = request.metadata.get("_run_id")
        run_id = str(requested_run_id) if requested_run_id else f"RUN_{secrets.token_hex(8)}"
        existing = None
        if requested_run_id:
            try:
                existing = runtime.get_run(run_id)
            except KeyError:
                existing = None
            if existing is not None:
                thread_id = existing.thread_id
                if request.thread_id and request.thread_id != thread_id:
                    raise ValueError("RUN_CORRELATION_INVALID: thread_id 与持久化 Run 不一致。")
                if existing.project_id and existing.project_id != request.project_id:
                    raise ValueError("RUN_CORRELATION_INVALID: Run 不属于当前 Project。")
                if existing.tenant_id != request.tenant_id or existing.principal_id != request.principal_id:
                    error = ValueError("RUN_ACCESS_DENIED: Run 不属于当前租户或主体。")
                    setattr(error, "code", "RUN_ACCESS_DENIED")
                    raise error
                if existing.status in {"COMPLETED", "FAILED", "CANCELLED", "WAITING_USER", "WAITING_HUMAN_APPROVAL"}:
                    raise ValueError(f"Run 已经是终态，不能重复执行：{run_id}")
                if existing.trace_id:
                    trace_id = existing.trace_id
                self._run_goal = runtime.get_run_goal(run_id)
                # The asynchronous Scholar Worker moves the durable Run to
                # RUNNING before invoking the backend.  A direct/embedded
                # caller may still hand us PENDING/QUEUED/INTERRUPTED.  Keep
                # this transition idempotent across both entry points; a
                # second start_run() would reject an already RUNNING Run and
                # turn an otherwise valid production queue execution into a
                # retry/failure loop.
                if existing.status in {"PENDING", "QUEUED", "INTERRUPTED"}:
                    runtime.start_run(
                        run_id,
                        worker_id=str(
                            request.metadata.get("_worker_id")
                            or getattr(self.session_manager, "worker_id", "crewai")
                        ),
                    )
                self._session_run = runtime.get_run(run_id)
                self.session_manager.set_active_run(session.session_id, run_id)
                return session.session_id, run_id, thread_id
        thread_id = request.thread_id or f"crew_{run_id}"
        if any(run.thread_id == thread_id for run in runtime.list_runs()):
            raise ValueError(f"RUN_CORRELATION_INVALID: thread_id 已绑定：{thread_id}")
        goal = default_run_goal(
            request.instruction,
            self._route_hint(request),
            success_criteria=request.metadata.get("success_criteria")
            if isinstance(request.metadata.get("success_criteria"), (list, tuple))
            else None,
            hard_constraints=request.metadata.get("hard_constraints")
            if isinstance(request.metadata.get("hard_constraints"), (list, tuple))
            else None,
        )
        self._run_goal = goal
        runtime.create_run(
            request.instruction,
            run_id=run_id,
            thread_id=thread_id,
            trace_id=trace_id,
            project_id=request.project_id,
            task_type=self._route_hint(request),
            goal=goal,
            tenant_id=request.tenant_id,
            principal_id=request.principal_id,
        )
        runtime.start_run(
            run_id, worker_id=getattr(self.session_manager, "worker_id", "crewai")
        )
        self.session_manager.set_active_run(session.session_id, run_id)
        self._session_run = runtime.get_run(run_id)
        return session.session_id, run_id, thread_id


class ScholarOrchestrationService:
    """The single Scholar orchestration entry point backed by CrewAI Manager."""

    def __init__(
        self,
        *,
        backend: str | None = "crewai",
        crewai: CrewAIBackend | None = None,
    ) -> None:
        selected = (backend or "crewai").strip().lower()
        if selected != "crewai":
            raise ValueError(
                "ScholarOrchestrationService 只支持 CrewAI Manager；旧 backend 已移除。"
            )
        if crewai is None:
            raise ValueError("crewai backend 需要注入 CrewAIBackend。")
        self.backend_name = selected
        self._backend = crewai

    def run(
        self,
        request: OrchestrationRequest | str,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> OrchestrationResult:
        if isinstance(request, str):
            if not project_id:
                raise ValueError("project_id 不能为空。")
            request = OrchestrationRequest(
                request_id=str(kwargs.pop("request_id", f"REQ_{secrets.token_hex(8)}")),
                project_id=project_id,
                instruction=request,
                **kwargs,
            )
        if not isinstance(request, OrchestrationRequest):
            raise TypeError("run() 需要 OrchestrationRequest 或 instruction 字符串。")
        assert self._backend is not None
        return self._backend.run(request)

    def resume(
        self,
        thread_id: str,
        resume_value: Any,
        project_id: str,
        *,
        instruction: str,
        session_id: str | None = None,
        task_type: str | None = None,
    ) -> OrchestrationResult:
        # The existing PatchApprovalService owns the decision and Safe Apply.
        # CrewAI only reconciles its persisted Session Run after that decision;
        # resume_value is never interpreted as approval.
        return self._backend.resume(
            thread_id,
            resume_value,
            project_id,
            instruction=instruction,
            session_id=session_id,
            task_type=task_type,
        )
