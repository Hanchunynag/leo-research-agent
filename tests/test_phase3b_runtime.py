from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agentic.config import AgenticRAGConfig
from app.scholar.composition import ScholarRuntimeFactory
from app.langchain_agent.checkpoint_factory import CheckpointUnavailable
from app.scholar.evaluation import (
    HarnessEvaluationCase,
    ScholarHarnessEvaluationSuite,
)
from app.scholar.harness import ScholarHarnessService, _token_estimate
from app.scholar.errors import CorrelationConflict, ResumeUnavailable
from app.scholar.context import ScholarContextBudget
from app.scholar.writing.harness import ScholarSkillRuntime
from app.scholar.writing.runtime import SkillResult
from app.session import SessionManager


def _minimal_runtime(tmp_path: Path):
    from tests.test_deepagents_harness import _runtime

    return _runtime(tmp_path)


def test_production_factory_requires_persistent_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointUnavailable):
        ScholarRuntimeFactory(
            tmp_path,
            config=AgenticRAGConfig(runtime_mode="production"),
            harness=object(),
        ).build()


def test_default_web_app_uses_factory_and_fails_fast_without_checkpoint(tmp_path: Path) -> None:
    from app.web.api import create_app

    app = create_app(tmp_path)
    assert isinstance(app.state.scholar_runtime_factory, ScholarRuntimeFactory)
    with pytest.raises(CheckpointUnavailable):
        with TestClient(app):
            pass


def test_cli_scholar_commands_are_thin_factory_clients() -> None:
    from main import build_parser

    parser = build_parser()
    request = parser.parse_args(
        [
            "scholar",
            "request",
            "write conclusion",
            "--project-id",
            "PROJECT",
            "--task-type",
            "WRITE_CONCLUSION",
        ]
    )
    status = parser.parse_args(
        ["scholar", "status", "--project-id", "PROJECT", "--thread-id", "THREAD"]
    )
    assert request.scholar_command == "request"
    assert request.task_type == "WRITE_CONCLUSION"
    assert status.scholar_command == "status"


def test_test_factory_uses_explicit_memory_and_closes(tmp_path: Path) -> None:
    factory = ScholarRuntimeFactory(
        tmp_path,
        config=AgenticRAGConfig(runtime_mode="test"),
        harness=object(),
    )
    bundle = factory.build()
    assert isinstance(bundle.checkpointer, InMemorySaver)
    assert bundle.mode == "test"
    factory.close()
    assert bundle._closed is True


def test_production_factory_rejects_inmemory_override(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="production"):
        ScholarRuntimeFactory(
            tmp_path,
            config=AgenticRAGConfig(runtime_mode="production"),
            checkpointer=InMemorySaver(),
            harness=object(),
        ).build()


def test_harness_evaluation_detects_routing_and_forbidden_tools() -> None:
    result = SimpleNamespace(
        result_type="ClaimSupportResult",
        metadata={
            "selected_skill": "support-claim",
            "visible_tools": {"main": ["get_project_context"]},
            "trace": {
                "usage": {"context_tokens": 10},
                "trace": [
                    {"kind": "tool", "name": "research_evidence", "status": "succeeded"},
                    {"kind": "tool", "name": "execute", "status": "succeeded"},
                ],
            },
        },
    )
    case = HarnessEvaluationCase(
        "claim",
        "support claim",
        "PROJECT",
        "support-claim",
        True,
        True,
        "ClaimSupportResult",
        False,
    )
    report = ScholarHarnessEvaluationSuite().run((case,), lambda _: result)
    assert report.passed is False
    assert report.metrics["Forbidden Tool Call Count"] == 1
    assert report.metrics["Capability Violation Count"] >= 1


def test_harness_evaluation_rejects_failed_domain_result_with_matching_type() -> None:
    result = SimpleNamespace(
        status="FAILED",
        result_type="WritingResult",
        metadata={"selected_skill": "write-introduction"},
    )
    case = HarnessEvaluationCase(
        "failed-writing",
        "write introduction",
        "PROJECT",
        "write-introduction",
        None,
        True,
        "WritingResult",
    )

    record = ScholarHarnessEvaluationSuite().evaluate_result(case, result)

    assert record.passed is False
    assert "INVALID_DOMAIN_RESULT" in record.failures


def test_harness_evaluation_accepts_crewai_route_and_tool_metadata() -> None:
    result = SimpleNamespace(
        status="COMPLETED",
        result_type="ClaimSupportResult",
        metadata={
            "selected_skill": "SUPPORT_CLAIM",
            "trace": {
                "trace": [
                    {
                        "kind": "tool",
                        "name": "capability_tool_call",
                        "status": "COMPLETED",
                        "metadata": {"tool_name": "research_capability"},
                    }
                ]
            },
        },
    )
    case = HarnessEvaluationCase(
        "crewai-claim",
        "support claim",
        "PROJECT",
        "support-claim",
        True,
        True,
        "ClaimSupportResult",
        False,
    )

    record = ScholarHarnessEvaluationSuite().evaluate_result(case, result)

    assert record.passed is True
    assert record.research_invoked is True
    assert record.reviewer_invoked is False


def test_session_runtime_tracks_scholar_run_lifecycle(tmp_path: Path) -> None:
    from tests.test_deepagents_harness import _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    manager = SessionManager(tmp_path)
    model = _ScriptedModel(
        responses=(
            {"tool": "get_project_context"},
            {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "conclude"}},
        )
    )
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=model,
        checkpointer=InMemorySaver(),
        session_manager=manager,
        writers={"WRITE_CONCLUSION": object()},
    )
    result = service.scholar_request(
        "conclude",
        runtime.project_store.project_id,
        task_type="WRITE_CONCLUSION",
        session_id="SESSION_LIFECYCLE",
        thread_id="THREAD_LIFECYCLE",
    )
    assert result.status == "COMPLETED"
    runs = manager.open("SESSION_LIFECYCLE").list_runs()
    assert len(runs) == 1
    assert runs[0].thread_id == "THREAD_LIFECYCLE"
    assert runs[0].status == "COMPLETED"


def test_production_factory_uses_real_sqlite_and_resumes_after_rebuild(tmp_path: Path) -> None:
    from app.scholar.project import ScholarProjectStore
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    project_store = ScholarProjectStore(tmp_path)
    checkpoint = tmp_path / "runtime" / "scholar-checkpoint.db"
    config = AgenticRAGConfig(
        runtime_mode="production",
        scholar_checkpoint_path=checkpoint,
    )
    first_calls: list[str] = []
    def execute(request, writer=None) -> SkillResult:
        first_calls.append(request.task_type)
        return SkillResult(
            request.task_type,
            "write-conclusion",
            "READY",
            {"draft": "after restart"},
        )

    runtime.execute = execute  # type: ignore[method-assign]
    first_model = _ScriptedModel(responses=(
        {"tool": "get_project_context"},
        {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "conclude"}},
    ))
    first_factory = ScholarRuntimeFactory(
        tmp_path,
        config=config,
        project_store=project_store,
        skill_runtime=runtime,
        model=first_model,
        research=_Research(),
        writers={"WRITE_CONCLUSION": object()},
        interrupt_on={"execute_scholar_skill": True},
    )
    first = first_factory.build()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        assert isinstance(first.checkpointer, SqliteSaver)
        interrupted = first.harness.scholar_request(
            "conclude",
            project_store.project_id,
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_PROCESS_RESTART",
            thread_id="THREAD_PROCESS_RESTART",
        )
        assert interrupted.status == "INTERRUPTED"
    finally:
        first_factory.close()

    rebuilt_runtime = ScholarSkillRuntime(tmp_path, project_store=project_store)
    rebuilt_runtime.execute = runtime.execute  # type: ignore[method-assign]
    rebuilt_model = _ScriptedModel(responses=({"content": "completed after restart"},))
    second_factory = ScholarRuntimeFactory(
        tmp_path,
        config=config,
        project_store=project_store,
        skill_runtime=rebuilt_runtime,
        model=rebuilt_model,
        research=_Research(),
        writers={"WRITE_CONCLUSION": object()},
        interrupt_on={"execute_scholar_skill": True},
    )
    second = second_factory.build()
    try:
        resumed = second.harness.resume(
            "THREAD_PROCESS_RESTART",
            {"decisions": [{"type": "approve"}]},
            project_store.project_id,
            instruction="conclude",
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_PROCESS_RESTART",
        )
        assert resumed.status == "COMPLETED"
        assert resumed.run_id == interrupted.run_id
        assert resumed.metadata["resumed"] is True
        assert first_calls == ["WRITE_CONCLUSION"]
        runs = second.session_manager.open("SESSION_PROCESS_RESTART").list_runs()
        assert len(runs) == 1
        assert runs[0].status == "COMPLETED"
        assert runs[0].checkpoint_ref == "THREAD_PROCESS_RESTART"
    finally:
        second_factory.close()


def test_factory_startup_marks_orphan_without_checkpoint_as_not_resumable(tmp_path: Path) -> None:
    from app.scholar.project import ScholarProjectStore
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    project_store = ScholarProjectStore(tmp_path)
    manager = SessionManager(tmp_path)
    manager.resolve("SESSION_ORPHAN", title="orphan", project_id=project_store.project_id)
    session_runtime = manager.open("SESSION_ORPHAN")
    session_runtime.create_run(
        "orphan",
        run_id="RUN_ORPHAN",
        thread_id="THREAD_ORPHAN",
        project_id=project_store.project_id,
    )
    session_runtime.start_run("RUN_ORPHAN", worker_id="old-worker")

    factory = ScholarRuntimeFactory(
        tmp_path,
        config=AgenticRAGConfig(runtime_mode="test"),
        project_store=project_store,
        session_manager=SessionManager(tmp_path),
        skill_runtime=runtime,
        model=_ScriptedModel(responses=({"content": "unused"},)),
        research=_Research(),
    )
    bundle = factory.build()
    try:
        assert bundle.session_manager.open("SESSION_ORPHAN").get_run("RUN_ORPHAN").status == "INTERRUPTED"
        with pytest.raises(ResumeUnavailable):
            bundle.harness.resume(
                "THREAD_ORPHAN",
                {"decisions": [{"type": "approve"}]},
                project_store.project_id,
                instruction="orphan",
                session_id="SESSION_ORPHAN",
            )
    finally:
        factory.close()


def test_new_request_rejects_reused_thread_correlation(tmp_path: Path) -> None:
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=_ScriptedModel(responses=(
            {"tool": "get_project_context"},
            {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "conclude"}},
        )),
        research=_Research(),
        writers={"WRITE_CONCLUSION": object()},
        checkpointer=InMemorySaver(),
        session_manager=SessionManager(tmp_path),
    )
    project_id = runtime.project_store.project_id
    service.scholar_request(
        "conclude",
        project_id,
        task_type="WRITE_CONCLUSION",
        session_id="SESSION_DUPLICATE_THREAD",
        thread_id="THREAD_DUPLICATE_THREAD",
    )
    with pytest.raises(CorrelationConflict):
        service.scholar_request(
            "conclude again",
            project_id,
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_DUPLICATE_THREAD",
            thread_id="THREAD_DUPLICATE_THREAD",
        )


def test_thread_correlation_cannot_cross_sessions(tmp_path: Path) -> None:
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=_ScriptedModel(responses=(
            {"tool": "get_project_context"},
            {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "conclude"}},
        )),
        research=_Research(),
        writers={"WRITE_CONCLUSION": object()},
        checkpointer=InMemorySaver(),
        session_manager=SessionManager(tmp_path),
    )
    project_id = runtime.project_store.project_id
    service.scholar_request(
        "conclude",
        project_id,
        task_type="WRITE_CONCLUSION",
        session_id="SESSION_THREAD_OWNER",
        thread_id="THREAD_GLOBAL_CORRELATION",
    )
    with pytest.raises(CorrelationConflict):
        service.scholar_request(
            "conclude elsewhere",
            project_id,
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_THREAD_OTHER",
            thread_id="THREAD_GLOBAL_CORRELATION",
        )


def test_context_budget_failure_is_a_failed_run_not_a_stale_running_run(tmp_path: Path) -> None:
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)
    manager = SessionManager(tmp_path)
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=_ScriptedModel(responses=({"content": "unused"},)),
        research=_Research(),
        checkpointer=InMemorySaver(),
        session_manager=manager,
        context_budget=ScholarContextBudget(
            supervisor=1,
            research=1,
            reviewer=1,
            total=3,
        ),
    )
    result = service.scholar_request(
        "conclude",
        runtime.project_store.project_id,
        task_type="WRITE_CONCLUSION",
        session_id="SESSION_CONTEXT_BUDGET",
        thread_id="THREAD_CONTEXT_BUDGET",
    )
    assert result.status == "FAILED"
    assert result.metadata["trace"]["termination_reason"] == "BUDGET_EXHAUSTED"
    assert manager.open("SESSION_CONTEXT_BUDGET").get_run(result.run_id).status == "FAILED"


def test_reviewer_context_projection_deduplicates_and_bounds_evidence() -> None:
    raw = [
        {
            "evidence_id": f"EVIDENCE_{index % 4}",
            "source_type": "x" * 1_000,
            "canonical_id": "x" * 1_000,
            "source_locator": "x" * 1_000,
            "locator_type": "x" * 1_000,
            "publication_date": "x" * 1_000,
            "provider": "x" * 1_000,
            "paper_id": "x" * 1_000,
            "work_id": "x" * 1_000,
            "document_id": "x" * 1_000,
            "section_id": "x" * 1_000,
            "chunk_id": "x" * 1_000,
            "evidence_grade": "x" * 1_000,
            "directness": "x" * 1_000,
            "content": "x" * 10_000,
            "metadata": {
                "title": "x" * 1_000,
                "authors": ["x" * 1_000 for _ in range(20)],
                "venue": "x" * 1_000,
                "doi": "x" * 1_000,
                "arxiv_id": "x" * 1_000,
                "canonical_id": "x" * 1_000,
            },
        }
        for index in range(12)
    ]

    projected = ScholarHarnessService._bounded_reviewer_evidence(raw)

    assert [item["evidence_id"] for item in projected] == [
        "EVIDENCE_0",
        "EVIDENCE_1",
        "EVIDENCE_2",
        "EVIDENCE_3",
    ]
    assert all(len(str(item["content"])) == 120 for item in projected)
    assert all(len(item.get("metadata", {}).get("authors", ())) == 4 for item in projected)
    # The projection itself stays comfortably inside the fixed 8k reviewer
    # audience budget even when upstream metadata is unexpectedly verbose.
    assert _token_estimate({"evidence": projected}) < 8_000


def test_existing_session_schema_migrates_checkpoint_reference(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project_id = _minimal_runtime(tmp_path).project_store.project_id
    manager.resolve("SESSION_SCHEMA_MIGRATION", title="migration", project_id=project_id)
    session_db = tmp_path / "data" / "runtime" / "sessions" / "SESSION_SCHEMA_MIGRATION" / "session.db"
    with sqlite3.connect(session_db) as connection:
        connection.execute("ALTER TABLE agent_runs DROP COLUMN checkpoint_ref")

    runtime = manager.open("SESSION_SCHEMA_MIGRATION")
    runtime.create_run(
        "migration",
        run_id="RUN_SCHEMA_MIGRATION",
        thread_id="THREAD_SCHEMA_MIGRATION",
        project_id=project_id,
    )
    runtime.set_checkpoint_ref("RUN_SCHEMA_MIGRATION", "THREAD_SCHEMA_MIGRATION")
    assert runtime.get_run("RUN_SCHEMA_MIGRATION").checkpoint_ref == "THREAD_SCHEMA_MIGRATION"


def test_provider_failure_metadata_keeps_provider_termination_reason(tmp_path: Path) -> None:
    from app.scholar.writing.support import ClaimSupportResult
    from tests.test_deepagents_harness import _Research, _ScriptedModel

    runtime = _minimal_runtime(tmp_path)

    def execute(request, writer=None) -> SkillResult:
        return SkillResult(
            "SUPPORT_CLAIM",
            "support-claim",
            "READY",
            ClaimSupportResult(
                "claim",
                "claim",
                (),
                "INSUFFICIENT_EVIDENCE",
                metadata={"error_code": "FRESHNESS_UNAVAILABLE"},
            ),
        )

    runtime.execute = execute  # type: ignore[method-assign]
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=_ScriptedModel(responses=(
            {"tool": "get_project_context"},
            {"tool": "execute_scholar_skill", "args": {"task_type": "SUPPORT_CLAIM", "instruction": "claim"}},
        )),
        research=_Research(),
        checkpointer=InMemorySaver(),
        session_manager=SessionManager(tmp_path),
    )
    result = service.scholar_request(
        "claim",
        runtime.project_store.project_id,
        task_type="SUPPORT_CLAIM",
    )
    assert result.error_codes == ("FRESHNESS_UNAVAILABLE",)
    assert result.metadata["trace"]["termination_reason"] == "PROVIDER_FAILED"
