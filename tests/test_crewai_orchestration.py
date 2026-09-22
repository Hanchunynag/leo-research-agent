from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.jobs.worker import JobCancelled
from app.orchestration.contracts import OrchestrationRequest, ResearchAgentOutput
from app.orchestration.evaluation import (
    OrchestrationEvaluationCase,
    evaluate_backend,
)
from app.orchestration.service import CrewAIBackend, ScholarOrchestrationService
from app.scholar.approval import PatchApprovalRequest, PatchApprovalService
from app.scholar.events import RunEventStore
from app.scholar.models import DraftPatch, EvidencePack, ReviewReport
from app.scholar.project import ScholarProjectStore
from app.scholar.writing.models import WritingResult
from app.scholar.writing.runtime import SkillResult
from app.session import SessionManager


class FakeResearch:
    workspace_id = "workspace"
    scope_version = 1

    def __init__(self) -> None:
        self.calls: list[str] = []

    def research(self, request: object, *, allow_web: bool = False) -> EvidencePack:
        query = str(getattr(request, "query"))
        self.calls.append(query)
        return EvidencePack(
            request_id=str(getattr(request, "request_id")),
            query=query,
            claims=(),
            evidence=(),
            unresolved=(),
            metadata={"freshness_decision": {"mode": "LOCAL_ONLY"}},
        )


@dataclass
class FakeWriter:
    calls: int = 0


class FakeSkillRuntime:
    def __init__(self, store: ScholarProjectStore, *, valid: bool = True) -> None:
        self.project_store = store
        self.valid = valid
        self.calls: list[str] = []

    def execute(self, request: object, *, writer: object | None = None) -> SkillResult:
        task_type = str(getattr(request, "task_type"))
        self.calls.append(task_type)
        patch = DraftPatch(
            patch_id=f"patch-{task_type.lower()}",
            target_section=str(getattr(request, "target_section")),
            base_hash="base-hash",
            proposed_content="new manuscript content",
            project_id=str(getattr(request, "project_id")),
        )
        report = ReviewReport(valid=self.valid, issues=())
        value = WritingResult(
            status="READY" if self.valid else "NEEDS_USER_REVIEW",
            request=request,  # type: ignore[arg-type]
            patch=patch,
            review_report=report,
        )
        self.project_store.save_patch(patch, review_report=report)
        return SkillResult(task_type, task_type.lower(), value.status, value)


def make_backend(
    tmp_path: Path, *, valid: bool = True
) -> tuple[CrewAIBackend, FakeResearch, FakeSkillRuntime, ScholarProjectStore]:
    store = ScholarProjectStore(tmp_path)
    research = FakeResearch()
    runtime = FakeSkillRuntime(store, valid=valid)
    crew_backend = CrewAIBackend(
        tmp_path,
        model=None,
        skill_runtime=runtime,
        research=research,
        writers={
            "WRITE_INTRODUCTION": FakeWriter(),
            "WRITE_CONCLUSION": FakeWriter(),
            "WRITE_ABSTRACT": FakeWriter(),
        },
        project_store=store,
        session_manager=SessionManager(tmp_path),
    )
    return crew_backend, research, runtime, store


def test_crewai_crew_has_exactly_four_roles_and_isolated_tools(tmp_path: Path) -> None:
    crew_backend, _, _, store = make_backend(tmp_path)
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="research-1",
            project_id=store.project_id,
            instruction="Which measurements aid state prediction?",
            task_type="RESEARCH",
        )
    )

    assert result.status == "COMPLETED"
    assert set(crew_backend.agents) == {"manager", "research", "writer", "reviewer"}
    assert crew_backend.agents["manager"].tools == []
    assert [tool.name for tool in crew_backend.agents["research"].tools] == [
        "research_capability"
    ]
    assert [tool.name for tool in crew_backend.agents["writer"].tools] == [
        "writer_capability"
    ]
    assert [tool.name for tool in crew_backend.agents["reviewer"].tools] == [
        "review_capability"
    ]


def test_introduction_enters_human_approval_without_apply(tmp_path: Path) -> None:
    crew_backend, research, _, store = make_backend(tmp_path)
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="intro-1",
            project_id=store.project_id,
            instruction="Write the introduction",
            task_type="WRITE_INTRODUCTION",
        )
    )

    assert result.status == "WAITING_HUMAN_APPROVAL"
    assert result.approval_required is True
    assert result.pending_action is not None
    # IntroductionSkill creates three bounded ResearchNeeds; CrewAI must
    # hand all of them to the shared Writing Runtime before it can write.
    assert len(research.calls) == 3
    assert result.pending_action["safe_apply"] == "PatchApprovalService only"


def test_introduction_source_coverage_error_reaches_orchestration_result(
    tmp_path: Path,
) -> None:
    store = ScholarProjectStore(tmp_path)

    class InsufficientSourcesRuntime(FakeSkillRuntime):
        def execute(self, request: object, *, writer: object | None = None) -> SkillResult:
            self.calls.append(str(getattr(request, "task_type")))
            value = WritingResult(
                status="FAILED",
                request=request,  # type: ignore[arg-type]
                error_codes=("INSUFFICIENT_INTRODUCTION_SOURCES",),
                warnings=("当前只有 3 篇可绑定引用的不同论文，还缺 12 篇。",),
                metadata={
                    "citation_coverage": {
                        "available_paper_count": 3,
                        "cited_paper_count": 3,
                        "minimum_unique_papers": 15,
                        "missing_paper_count": 12,
                    }
                },
            )
            return SkillResult(
                "WRITE_INTRODUCTION",
                "write-introduction",
                "FAILED",
                value,
                value.error_codes,
                value.warnings,
            )

    runtime = InsufficientSourcesRuntime(store)
    backend = CrewAIBackend(
        tmp_path,
        model=None,
        skill_runtime=runtime,
        research=FakeResearch(),
        writers={"WRITE_INTRODUCTION": FakeWriter()},
        project_store=store,
        session_manager=SessionManager(tmp_path),
    )

    result = ScholarOrchestrationService(backend="crewai", crewai=backend).run(
        OrchestrationRequest(
            request_id="intro-insufficient-sources",
            project_id=store.project_id,
            instruction="Write the introduction",
            task_type="WRITE_INTRODUCTION",
        )
    )

    assert result.status == "FAILED"
    assert result.error_codes == ["INSUFFICIENT_INTRODUCTION_SOURCES"]
    assert isinstance(result.value, WritingResult)
    assert result.value.metadata["citation_coverage"]["available_paper_count"] == 3
    assert result.value.patch is None
    assert runtime.calls == ["WRITE_INTRODUCTION"]


def test_conclusion_never_triggers_research(tmp_path: Path) -> None:
    crew_backend, research, runtime, store = make_backend(tmp_path)
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="conclusion-1",
            project_id=store.project_id,
            instruction="Write the conclusion",
            task_type="WRITE_CONCLUSION",
        )
    )

    assert result.status == "WAITING_HUMAN_APPROVAL"
    assert research.calls == []
    assert runtime.calls == ["WRITE_CONCLUSION"]


def test_missing_manuscript_is_reported_as_insufficient_state(tmp_path: Path) -> None:
    store = ScholarProjectStore(tmp_path)

    class MissingManuscriptRuntime:
        project_store = store

        @staticmethod
        def execute(request: object, *, writer: object | None = None) -> SkillResult:
            raise FileNotFoundError("LaTeX Project 中没有找到 root .tex 文件。")

    backend = CrewAIBackend(
        tmp_path,
        model=None,
        skill_runtime=MissingManuscriptRuntime(),
        research=FakeResearch(),
        writers={"WRITE_CONCLUSION": object()},
        project_store=store,
    )
    result = backend.write(
        OrchestrationRequest(
            request_id="missing-manuscript-1",
            project_id=store.project_id,
            instruction="Write the conclusion",
            task_type="WRITE_CONCLUSION",
        ),
        "WRITE_CONCLUSION",
        None,
        None,
    )

    assert result.status == "INSUFFICIENT_MANUSCRIPT_STATE"
    assert result.missing_context == ["INSUFFICIENT_MANUSCRIPT_STATE"]


def test_result_evidence_projection_preserves_verified_pack_and_citation_fields() -> None:
    content = "Verified canonical evidence"
    pack = EvidencePack(
        request_id="evidence-1",
        query="evidence",
        evidence=(
            {
                "evidence_id": "E1",
                "paper_id": "P1",
                "content": content,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "validation_status": "canonical_locator_verified",
            },
        ),
    )

    evidence = CrewAIBackend._evidence_from_result(pack)

    assert evidence == [pack.evidence[0]]
    assert CrewAIBackend._citations_from_evidence(evidence) == [
        {"evidence_id": "E1", "paper_id": "P1"}
    ]


def test_introduction_fails_closed_when_research_has_unresolved_claims(
    tmp_path: Path,
) -> None:
    class InsufficientResearch(FakeResearch):
        def research(self, request: object, *, allow_web: bool = False) -> EvidencePack:
            query = str(getattr(request, "query"))
            self.calls.append(query)
            return EvidencePack(
                request_id=str(getattr(request, "request_id")),
                query=query,
                unresolved=("missing claim",),
                metadata={"freshness_decision": {"mode": "LOCAL_ONLY"}},
            )

    store = ScholarProjectStore(tmp_path)
    research = InsufficientResearch()
    runtime = FakeSkillRuntime(store)
    crew_backend = CrewAIBackend(
        tmp_path,
        model=None,
        skill_runtime=runtime,
        research=research,
        writers={"WRITE_INTRODUCTION": FakeWriter()},
        project_store=store,
        session_manager=SessionManager(tmp_path),
    )
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="intro-insufficient",
            project_id=store.project_id,
            instruction="Write the introduction",
            task_type="WRITE_INTRODUCTION",
        )
    )

    assert result.status == "FAILED"
    assert result.error_codes == ["INSUFFICIENT_EVIDENCE"]
    assert runtime.calls == []


def test_crewai_run_reuses_one_trace_id_in_session_runtime(tmp_path: Path) -> None:
    crew_backend, _, _, store = make_backend(tmp_path)
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="trace-1",
            project_id=store.project_id,
            instruction="Which measurements aid state prediction?",
            task_type="RESEARCH",
        )
    )

    assert result.trace_id
    assert result.session_id
    assert result.run_id
    stored = crew_backend.session_manager.open(result.session_id).get_run(result.run_id)
    assert stored.trace_id == result.trace_id


def test_crewai_backend_accepts_worker_started_run_without_double_start(
    tmp_path: Path,
) -> None:
    crew_backend, _, _, store = make_backend(tmp_path)
    session = crew_backend.session_manager.resolve(
        "worker-started-session",
        title="worker-started",
        project_id=store.project_id,
    )
    session_runtime = crew_backend.session_manager.open(session.session_id)
    session_runtime.create_run(
        "research",
        run_id="RUN_WORKER_STARTED",
        thread_id="crew_RUN_WORKER_STARTED",
        trace_id="TRACE_WORKER_STARTED",
        project_id=store.project_id,
    )
    session_runtime.start_run("RUN_WORKER_STARTED", worker_id="worker")

    session_id, run_id, thread_id = crew_backend._prepare_run(  # noqa: SLF001
        OrchestrationRequest(
            request_id="worker-started-request",
            project_id=store.project_id,
            instruction="research",
            session_id=session.session_id,
            thread_id="crew_RUN_WORKER_STARTED",
            task_type="RESEARCH",
            metadata={"_run_id": "RUN_WORKER_STARTED", "_worker_id": "worker"},
        ),
        trace_id="TRACE_WORKER_STARTED",
    )

    assert (session_id, run_id, thread_id) == (
        session.session_id,
        "RUN_WORKER_STARTED",
        "crew_RUN_WORKER_STARTED",
    )


def test_crewai_resume_reuses_original_thread_and_persisted_manager_checkpoint(
    tmp_path: Path,
) -> None:
    crew_backend, _, _, store = make_backend(tmp_path)
    service = ScholarOrchestrationService(backend="crewai", crewai=crew_backend)

    def cancel_at_first_capability() -> None:
        raise JobCancelled("test interruption")

    request = OrchestrationRequest(
        request_id="resume-checkpoint-1",
        project_id=store.project_id,
        instruction="Find verified evidence",
        task_type="RESEARCH",
        session_id="resume-session",
        thread_id="resume-thread",
        tenant_id="tenant-resume",
        principal_id="principal-resume",
        metadata={"_cancellation_checker": cancel_at_first_capability},
    )
    with pytest.raises(JobCancelled):
        service.run(request)

    runtime = crew_backend.session_manager.open("resume-session")
    interrupted = runtime.list_runs()[0]
    checkpoint = runtime.load_checkpoint(interrupted.run_id)
    assert checkpoint is not None
    assert interrupted.thread_id == "resume-thread"
    assert interrupted.tenant_id == "tenant-resume"
    assert interrupted.principal_id == "principal-resume"
    assert checkpoint["run_state"]["recovery_action"] == "CALL_RESEARCH"

    # The worker/recovery layer owns the RUNNING -> INTERRUPTED transition;
    # simulate that durable recovery before invoking the public resume API.
    runtime.complete_run(
        interrupted.run_id,
        status="INTERRUPTED",
        answer="",
        citations=[],
        evidence=[],
        metadata={"orchestration_status": "INTERRUPTED"},
    )
    resumed = service.resume(
        "resume-thread",
        {"ignored": True},
        store.project_id,
        instruction="ignored by checkpoint resume",
        session_id="resume-session",
        task_type="RESEARCH",
    )

    assert resumed.status == "COMPLETED"
    assert resumed.run_id == interrupted.run_id
    assert resumed.thread_id == "resume-thread"
    assert runtime.load_checkpoint(interrupted.run_id) is None


def test_crewai_reviewer_checkpoint_resumes_reviewer_without_replaying_writer(
    tmp_path: Path,
) -> None:
    crew_backend, research, runtime, store = make_backend(tmp_path)
    service = ScholarOrchestrationService(backend="crewai", crewai=crew_backend)
    capability_checks = 0

    def cancel_when_reviewer_starts() -> None:
        nonlocal capability_checks
        capability_checks += 1
        if capability_checks == 2:
            raise JobCancelled("interrupt before reviewer capability")

    request = OrchestrationRequest(
        request_id="resume-reviewer-1",
        project_id=store.project_id,
        instruction="Write the conclusion",
        task_type="WRITE_CONCLUSION",
        session_id="resume-reviewer-session",
        thread_id="resume-reviewer-thread",
        tenant_id="tenant-reviewer",
        principal_id="principal-reviewer",
        metadata={"_cancellation_checker": cancel_when_reviewer_starts},
    )
    with pytest.raises(JobCancelled):
        service.run(request)

    session_runtime = crew_backend.session_manager.open("resume-reviewer-session")
    interrupted = session_runtime.list_runs()[0]
    checkpoint = session_runtime.load_checkpoint(interrupted.run_id)
    assert checkpoint is not None
    assert checkpoint["run_state"]["recovery_action"] == "CALL_REVIEWER"
    assert checkpoint["run_state"]["draft_patch_id"]
    session_runtime.complete_run(
        interrupted.run_id,
        status="INTERRUPTED",
        answer="",
        citations=[],
        evidence=[],
        metadata={"orchestration_status": "INTERRUPTED"},
    )

    resumed = service.resume(
        "resume-reviewer-thread",
        None,
        store.project_id,
        instruction="ignored by checkpoint resume",
        session_id="resume-reviewer-session",
        task_type="WRITE_CONCLUSION",
    )

    assert resumed.status == "WAITING_HUMAN_APPROVAL"
    assert runtime.calls == ["WRITE_CONCLUSION"]
    assert len(research.calls) == 0
    assert any(
        decision["next_action"] == "CALL_REVIEWER"
        for decision in resumed.diagnostics["manager_decisions"]
    )


def test_checkpoint_persistence_failure_fails_closed_before_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.session.runtime import SessionRuntime

    crew_backend, research, runtime, store = make_backend(tmp_path)

    def fail_checkpoint(*_: object, **__: object) -> str:
        raise OSError("checkpoint store unavailable")

    monkeypatch.setattr(SessionRuntime, "save_checkpoint", fail_checkpoint)
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="checkpoint-failure-1",
            project_id=store.project_id,
            instruction="Find verified evidence",
            task_type="RESEARCH",
            session_id="checkpoint-failure-session",
        )
    )

    assert result.status == "FAILED"
    assert result.error_codes == ["CHECKPOINT_PERSIST_FAILED"]
    assert research.calls == []
    assert runtime.calls == []
    persisted = crew_backend.session_manager.open("checkpoint-failure-session").list_runs()[0]
    assert persisted.status == "FAILED"


def test_crewai_trace_maps_provider_usage_and_flow_capability_calls(
    tmp_path: Path,
) -> None:
    class UsageProvider:
        model_name = "fixture/usage"
        provider_name = "fixture"

        def chat_completion(self, messages: object, **_: object) -> dict[str, object]:
            content = "\n".join(
                str(item.get("content", ""))
                for item in messages  # type: ignore[union-attr]
                if isinstance(item, dict)
            ).casefold()
            if "scholar manager agent" in content:
                payload = {"status": "COMPLETED", "selected_route": "RESEARCH"}
            elif "research agent" in content:
                payload = {"status": "COMPLETED", "research_summary": "ok"}
            else:
                payload = {"status": "COMPLETED", "final_answer": "ok"}
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }

    crew_backend, _, _, store = make_backend(tmp_path)
    crew_backend.model = UsageProvider()
    result = ScholarOrchestrationService(backend="crewai", crewai=crew_backend).run(
        OrchestrationRequest(
            request_id="trace-usage-1",
            project_id=store.project_id,
            instruction="Which measurements aid state prediction?",
            task_type="RESEARCH",
        )
    )

    trace = result.diagnostics["trace"]
    assert trace["agent_call_count"] == 3
    assert trace["tool_call_count"] == 1
    assert trace["llm_call_count"] == 3
    assert trace["prompt_tokens"] == 9
    assert trace["completion_tokens"] == 6
    assert trace["total_tokens"] == 15
    assert trace["usage"]["total_tokens"] == 15
    assert trace["cost"] is None


def test_crewai_provider_adapter_exposes_pinned_converter_capability() -> None:
    from app.orchestration.crewai.models import CrewAIProviderAdapter

    class Provider:
        model_name = "fixture/compatibility"

        def chat_completion(self, messages: object, **_: object) -> dict[str, object]:
            return {"choices": [{"message": {"content": "{}"}}]}

    adapter = CrewAIProviderAdapter.from_model(Provider())
    assert adapter.supports_function_calling() is True


def test_crewai_trace_persists_duration_for_completed_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.orchestration.crewai.tracing as tracing

    samples = iter((10.0, 10.2505))
    monkeypatch.setattr(tracing, "perf_counter", lambda: next(samples))
    store = RunEventStore(tmp_path)
    adapter = tracing.CrewAITraceAdapter(
        project_id="PROJECT",
        session_id="SESSION",
        run_id="RUN",
        trace_id="TRACE",
        event_sink=lambda event: store.append_trace(
            event,
            run_id="RUN",
            project_id="PROJECT",
            session_id="SESSION",
        ),
    )
    metadata = {
        "task_id": "TASK",
        "agent_id": "AGENT",
        "agent_role": "Research Agent",
    }

    adapter.record("LLMCallStartedEvent", "RUNNING", metadata, kind="llm")
    adapter.record("LLMCallCompletedEvent", "COMPLETED", metadata, kind="llm")

    rows = store.list("RUN")
    assert rows[0]["duration_ms"] is None
    assert rows[1]["duration_ms"] == pytest.approx(250.5)


def test_crewai_structured_call_disables_hidden_reasoning_when_supported() -> None:
    from app.orchestration.crewai.models import CrewAIProviderAdapter

    class Provider:
        model_name = "fixture/reasoning"

        def __init__(self) -> None:
            self.reasoning_efforts: list[str | None] = []

        def chat_completion(
            self,
            messages: object,
            *,
            max_tokens: int,
            reasoning_effort: str | None = None,
        ) -> dict[str, object]:
            self.reasoning_efforts.append(reasoning_effort)
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"status":"COMPLETED","research_summary":"ok"}'
                        }
                    }
                ]
            }

    provider = Provider()
    adapter = CrewAIProviderAdapter.from_model(provider)
    result = adapter.call(
        [{"role": "user", "content": "Return the contract."}],
        response_model=ResearchAgentOutput,
    )

    assert result.status == "COMPLETED"
    assert provider.reasoning_efforts == ["none"]


def test_capability_tools_are_bound_to_domain_runtime_and_budgeted(
    tmp_path: Path,
) -> None:
    crew_backend, research, runtime, store = make_backend(tmp_path)
    request = OrchestrationRequest(
        request_id="tool-1",
        project_id=store.project_id,
        instruction="Write the conclusion",
        task_type="WRITE_CONCLUSION",
    )
    crew_backend._request = request
    crew_backend._run_id = "run-tool-1"
    crew_backend._active_route = "WRITE_CONCLUSION"
    crew_backend.max_tool_calls = 2

    writer_result = crew_backend.tool_write(instruction=request.instruction)
    assert writer_result["status"] == "READY"
    assert writer_result["draft_patch"]["patch_id"] == "patch-write_conclusion"
    assert runtime.calls == ["WRITE_CONCLUSION"]

    review_result = crew_backend.tool_review(patch_id="patch-write_conclusion")
    assert review_result["status"] == "PASS"
    assert research.calls == []

    over_budget = crew_backend.tool_review(patch_id="patch-write_conclusion")
    assert over_budget["error_code"] == "TOOL_BUDGET_EXCEEDED"


def test_invalid_task_type_fails_at_contract_boundary() -> None:
    with pytest.raises(ValueError, match="task_type"):
        OrchestrationRequest(
            request_id="invalid-task",
            project_id="project",
            instruction="do something",
            task_type="CITATION_AGENT",
        )


def test_rejected_patch_reconciles_waiting_run_without_using_resume_value(
    tmp_path: Path,
) -> None:
    crew_backend, _, _, store = make_backend(tmp_path)
    service = ScholarOrchestrationService(backend="crewai", crewai=crew_backend)
    waiting = service.run(
        OrchestrationRequest(
            request_id="approval-reject-1",
            project_id=store.project_id,
            instruction="Write the conclusion",
            task_type="WRITE_CONCLUSION",
            session_id="approval-session",
            thread_id="approval-thread",
        )
    )
    assert waiting.status == "WAITING_HUMAN_APPROVAL"
    assert waiting.pending_action is not None

    approval = PatchApprovalService(tmp_path, project_store=store)
    rejected = approval.reject(
        PatchApprovalRequest(
            patch_id=str(waiting.pending_action["patch_id"]),
            project_id=store.project_id,
            decision="REJECT",
            expected_base_hash=str(waiting.pending_action["expected_base_hash"]),
            actor="human:test",
        )
    )
    assert rejected.status == "REJECTED"

    resumed = service.resume(
        "approval-thread",
        {"approve": True},
        store.project_id,
        instruction="ignored by approval boundary",
        session_id="approval-session",
        task_type="WRITE_CONCLUSION",
    )
    assert resumed.status == "COMPLETED"
    assert resumed.error_codes == ["HUMAN_REJECTED"]
    assert resumed.approval_required is False
    assert resumed.pending_action is None
    assert (
        crew_backend.session_manager.open("approval-session")
        .get_run(str(waiting.run_id))
        .status
        == "COMPLETED"
    )

    # Reconciliation is idempotent: a repeated resume cannot reapply or
    # reinterpret the caller-provided resume payload as authorization.
    repeated = service.resume(
        "approval-thread",
        {"approve": True},
        store.project_id,
        instruction="ignored",
        session_id="approval-session",
    )
    assert repeated.status == "COMPLETED"
    assert repeated.error_codes == ["HUMAN_REJECTED"]


def test_flow_state_keeps_only_reference_projection_for_domain_handoffs() -> None:
    from app.orchestration.crewai.flow import FlowState

    fields = set(FlowState.model_fields)
    assert "research_result" not in fields
    assert "writer_result" not in fields
    assert "review_result" not in fields
    assert {
        "research_result_reference",
        "draft_patch_id",
        "review_decision",
        "pending_approval",
    } <= fields


def test_backend_neutral_evaluation_reads_crewai_trace_contract() -> None:
    from app.orchestration.contracts import OrchestrationResult

    case = OrchestrationEvaluationCase(
        "research",
        "research",
        "RESEARCH",
        "COMPLETED",
        True,
        False,
        False,
    )
    result = OrchestrationResult(
        status="COMPLETED",
        request_id="r",
        project_id="p",
        selected_route="RESEARCH",
        specialist_results={"research": {"status": "COMPLETED"}},
        diagnostics={
            "trace": {
                "agent_call_count": 2,
                "tool_call_count": 1,
                "total_tokens": 12,
                "events": [
                    {
                        "name": "flow_transition",
                        "kind": "flow",
                        "metadata": {"state": "ROUTING"},
                    },
                    {
                        "name": "capability_tool_call",
                        "kind": "tool",
                        "metadata": {"tool_name": "research_capability"},
                    },
                    {
                        "name": "flow_transition",
                        "kind": "flow",
                        "metadata": {"state": "RESEARCHING"},
                    },
                ],
            }
        },
        backend="crewai",
    )
    report = evaluate_backend("crewai", (case,), lambda _: result)
    assert report.passed is True
    assert report.metrics["Routing Accuracy"] == 1.0
    assert report.metrics["Average Agent Calls"] == 2.0
    assert report.metrics["Token Usage"] == 12.0
