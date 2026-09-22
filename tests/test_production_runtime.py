from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.orchestration.contracts import OrchestrationRequest, OrchestrationResult
from app.scholar.events import RunEventStore
from app.scholar.project import ScholarProjectStore
from app.scholar.runs import ScholarRunManager, ScholarRunWorker
from app.scholar.models import DraftPatch, ReviewReport
from app.jobs.registry import WorkerRegistry
from app.observability.metrics import build_metrics_snapshot
from app.knowledge.service import knowledge_runtime_status
from app.web.api import create_app
from tests.test_web_api import FakeWebRuntime


def test_public_retrieval_status_names_hierarchical_architecture(tmp_path: Path) -> None:
    status = knowledge_runtime_status(tmp_path)

    assert status["retrieval_backend"] == "hierarchical"
    assert status["retrieval_components"][:3] == [
        "paper_level_bm25",
        "paper_level_bge_m3_dense",
        "paper_level_rrf",
    ]
    assert "per_paper_content_bge_m3_dense" in status["retrieval_components"]


class FixtureOrchestration:
    backend_name = "crewai"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, request: OrchestrationRequest) -> OrchestrationResult:
        run_id = str(request.metadata["_run_id"])
        self.calls.append(run_id)
        return OrchestrationResult(
            status="COMPLETED",
            request_id=request.request_id,
            project_id=request.project_id,
            session_id=request.session_id,
            run_id=run_id,
            thread_id=request.thread_id,
            trace_id=request.metadata.get("_trace_id"),
            selected_route="RESEARCH",
            final_answer="fixture result",
            backend="crewai",
        )


@dataclass
class ApprovalFixtureOrchestration:
    store: ScholarProjectStore
    calls: list[str]

    backend_name = "crewai"

    def run(self, request: OrchestrationRequest) -> OrchestrationResult:
        run_id = str(request.metadata["_run_id"])
        patch_id = f"patch-{run_id}"
        patch = DraftPatch(
            patch_id=patch_id,
            target_section="conclusion",
            base_hash="fixture-base",
            proposed_content="fixture content",
            project_id=request.project_id,
        )
        self.store.save_patch(
            patch,
            review_report=ReviewReport(valid=True),
            source_session_id=request.session_id,
            source_run_id=run_id,
        )
        self.calls.append(f"run:{run_id}")
        return OrchestrationResult(
            status="WAITING_HUMAN_APPROVAL",
            request_id=request.request_id,
            project_id=request.project_id,
            session_id=request.session_id,
            run_id=run_id,
            thread_id=request.thread_id,
            trace_id=str(request.metadata.get("_trace_id") or "fixture-trace"),
            selected_route="WRITE_CONCLUSION",
            pending_action={
                "type": "HUMAN_APPROVAL",
                "patch_id": patch_id,
                "expected_base_hash": patch.base_hash,
            },
            backend="crewai",
        )

    def resume(self, thread_id: str, *_: object, **__: object) -> OrchestrationResult:
        self.calls.append(f"resume:{thread_id}")
        return OrchestrationResult(
            status="COMPLETED",
            request_id="resume",
            project_id=self.store.project_id,
            session_id=None,
            run_id=None,
            thread_id=thread_id,
            trace_id="fixture-trace",
            selected_route="WRITE_CONCLUSION",
            final_answer="reconciled",
            backend="crewai",
        )


@dataclass
class CancellingFixtureOrchestration:
    repository: object

    backend_name = "crewai"

    def run(self, request: OrchestrationRequest) -> OrchestrationResult:
        job_id = str(request.metadata["_job_id"])
        self.repository.request_cancel(job_id)  # type: ignore[attr-defined]
        run_id = str(request.metadata["_run_id"])
        return OrchestrationResult(
            status="COMPLETED",
            request_id=request.request_id,
            project_id=request.project_id,
            session_id=request.session_id,
            run_id=run_id,
            thread_id=request.thread_id,
            trace_id="fixture-trace",
            selected_route="RESEARCH",
            final_answer="should not be committed",
            backend="crewai",
        )


def _manager(tmp_path: Path) -> tuple[ScholarRunManager, FixtureOrchestration]:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    orchestration = FixtureOrchestration()
    manager = ScholarRunManager(
        tmp_path,
        repository=PersistentJobRepository(tmp_path),
        session_manager=SessionManager(tmp_path),
        event_store=RunEventStore(tmp_path),
        orchestration=orchestration,
    )
    return manager, orchestration


def test_async_run_is_queued_until_worker_claims_it(tmp_path: Path) -> None:
    manager, orchestration = _manager(tmp_path)
    project_id = ScholarProjectStore(tmp_path).project_id
    request = OrchestrationRequest(
        request_id="REQ_ASYNC",
        project_id=project_id,
        instruction="research",
        task_type="RESEARCH",
    )

    queued = manager.create(request, idempotency_key="async-key")
    assert queued["status"] == "QUEUED"
    assert orchestration.calls == []
    assert manager.snapshot(queued["run_id"])["task_type"] == "RESEARCH"

    repeated = manager.create(request, idempotency_key="async-key")
    assert repeated["run_id"] == queued["run_id"]
    worker = ScholarRunWorker(manager, worker_id="fixture-worker")
    result = worker.run_once()
    assert result is not None and result.status == "SUCCEEDED"
    assert orchestration.calls == [queued["run_id"]]
    assert manager.snapshot(queued["run_id"])["status"] == "COMPLETED"


def test_run_events_are_persisted_and_cursor_replay_is_stable(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    project_id = ScholarProjectStore(tmp_path).project_id
    request = OrchestrationRequest(
        request_id="REQ_EVENTS",
        project_id=project_id,
        instruction="research",
        task_type="RESEARCH",
    )
    created = manager.create(request, idempotency_key="events-key")
    worker = ScholarRunWorker(manager, worker_id="fixture-worker")
    worker.run_once()

    events = manager.event_store.list(created["run_id"])
    assert [item["cursor"] for item in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "RUN_COMPLETED"
    assert manager.event_store.list(created["run_id"], after=1)[0]["cursor"] == 2


def test_approval_api_queues_run_reconciliation_after_reject(tmp_path: Path) -> None:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    store = ScholarProjectStore(tmp_path)
    orchestration = ApprovalFixtureOrchestration(store, [])
    manager = ScholarRunManager(
        tmp_path,
        repository=PersistentJobRepository(tmp_path),
        session_manager=SessionManager(tmp_path),
        event_store=RunEventStore(tmp_path),
        orchestration=orchestration,
    )
    app = create_app(
        tmp_path,
        runtime=FakeWebRuntime(tmp_path),
        scholar_run_manager=manager,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/scholar/runs",
            json={
                "instruction": "write conclusion",
                "project_id": store.project_id,
                "task_type": "WRITE_CONCLUSION",
                "session_id": "approval-api-session",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        worker = ScholarRunWorker(manager, worker_id="approval-api-worker")
        try:
            assert worker.run_once() is not None
            waiting = client.get(f"/api/scholar/runs/{run_id}").json()
            patch_id = f"patch-{run_id}"
            rejected = client.post(
                f"/api/scholar/approvals/{patch_id}/reject",
                json={
                    "project_id": store.project_id,
                    "session_id": "approval-api-session",
                    "expected_base_hash": "fixture-base",
                    "actor": "human:test",
                },
            )
            assert rejected.status_code == 200
            assert rejected.json()["status"] == "REJECTED"
            assert rejected.json()["run_reconciliation"]["queued"] is True
            assert waiting["async_run"]["status"] == "WAITING_HUMAN_APPROVAL"

            assert worker.run_once() is not None
            assert manager.snapshot(run_id)["status"] == "COMPLETED"
            assert orchestration.calls[-1].startswith("resume:")
        finally:
            worker.close()


def test_worker_registry_and_run_metrics_are_durable_projections(tmp_path: Path) -> None:
    registry = WorkerRegistry(tmp_path)
    registered = registry.register("worker-metrics", pid=123, hostname="fixture")
    assert registered["status"] == "RUNNING"
    assert registry.list()[0]["worker_id"] == "worker-metrics"
    stopped = registry.stop("worker-metrics")
    assert stopped is not None and stopped["status"] == "STOPPED"

    manager, _ = _manager(tmp_path)
    project_id = ScholarProjectStore(tmp_path).project_id
    created = manager.create(
        OrchestrationRequest(
            request_id="REQ_METRICS",
            project_id=project_id,
            instruction="research",
            task_type="RESEARCH",
        ),
        idempotency_key="metrics-key",
    )
    manager.event_store.append(
        run_id=created["run_id"],
        project_id=project_id,
        session_id=created["session_id"],
        event_type="SUBAGENT_STARTED",
        node="Research Agent",
        status="RUNNING",
        summary="research started",
        metadata={"kind": "agent"},
    )
    manager.event_store.append(
        run_id=created["run_id"],
        project_id=project_id,
        session_id=created["session_id"],
        event_type="DOMAIN_RESULT",
        node="Provider",
        status="COMPLETED",
        summary="provider usage",
        metadata={
            "kind": "llm",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "cost": 0.01},
        },
    )
    metrics = build_metrics_snapshot(tmp_path, run_manager=manager)["metrics"]
    assert metrics["agent_calls"] == 1
    assert metrics["token_usage"] == 6
    assert metrics["cost"] == 0.01


def test_running_run_cancellation_is_projected_without_completing_cognition(tmp_path: Path) -> None:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    repository = PersistentJobRepository(tmp_path)
    store = ScholarProjectStore(tmp_path)
    manager = ScholarRunManager(
        tmp_path,
        repository=repository,
        session_manager=SessionManager(tmp_path),
        event_store=RunEventStore(tmp_path),
        orchestration=CancellingFixtureOrchestration(repository),
    )
    created = manager.create(
        OrchestrationRequest(
            request_id="REQ_CANCEL_RUNNING",
            project_id=store.project_id,
            instruction="research",
            task_type="RESEARCH",
        ),
        idempotency_key="cancel-running-key",
    )
    worker = ScholarRunWorker(manager, worker_id="cancel-worker")
    try:
        result = worker.run_once()
        assert result is not None and result.status == "CANCELLED"
        assert manager.snapshot(created["run_id"])["status"] == "CANCELLED"
        assert manager.event_store.list(created["run_id"])[-1]["type"] == "RUN_CANCELLED"
    finally:
        worker.close()


def test_recovery_reconciles_failed_job_running_run_and_clears_active_pointer(
    tmp_path: Path,
) -> None:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    repository = PersistentJobRepository(tmp_path)
    sessions = SessionManager(tmp_path)
    manager = ScholarRunManager(
        tmp_path,
        repository=repository,
        session_manager=sessions,
        event_store=RunEventStore(tmp_path),
    )
    store = ScholarProjectStore(tmp_path)
    created = manager.create(
        OrchestrationRequest(
            request_id="REQ_RECONCILE_FAILED",
            project_id=store.project_id,
            instruction="research",
            task_type="RESEARCH",
        ),
        idempotency_key="reconcile-failed-key",
    )
    runtime = sessions.open(created["session_id"])
    runtime.start_run(created["run_id"], worker_id="dead-worker")
    sessions.set_active_run(created["session_id"], created["run_id"])
    repository.claim(created["job_id"], "dead-worker")
    repository.set_status(
        created["job_id"],
        "FAILED",
        error_type="ProcessRestart",
        error_summary="Worker heartbeat lost after process restart.",
    )

    reconciled = manager.reconcile_durable_state()

    assert any(item["run_id"] == created["run_id"] and item["transitioned"] for item in reconciled)
    assert manager.snapshot(created["run_id"])["status"] == "FAILED"
    assert sessions.get(created["session_id"]).active_run_id is None
    assert manager.event_store.list(created["run_id"])[-1]["type"] == "RUN_FAILED"


def test_recovery_reconciles_cancelled_job_running_run_and_is_idempotent(
    tmp_path: Path,
) -> None:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    repository = PersistentJobRepository(tmp_path)
    sessions = SessionManager(tmp_path)
    manager = ScholarRunManager(
        tmp_path,
        repository=repository,
        session_manager=sessions,
        event_store=RunEventStore(tmp_path),
    )
    store = ScholarProjectStore(tmp_path)
    created = manager.create(
        OrchestrationRequest(
            request_id="REQ_RECONCILE_CANCELLED",
            project_id=store.project_id,
            instruction="research",
            task_type="RESEARCH",
        ),
        idempotency_key="reconcile-cancelled-key",
    )
    runtime = sessions.open(created["session_id"])
    runtime.start_run(created["run_id"], worker_id="dead-worker")
    sessions.set_active_run(created["session_id"], created["run_id"])
    repository.claim(created["job_id"], "dead-worker")
    repository.set_status(
        created["job_id"],
        "CANCELLED",
        error_type="CancelledAfterWorkerLost",
        error_summary="Cancellation finalized after Worker heartbeat expired.",
    )

    first = manager.reconcile_durable_state()
    event_count = manager.event_store.count(created["run_id"])
    second = manager.reconcile_durable_state()

    assert any(item["run_id"] == created["run_id"] and item["transitioned"] for item in first)
    assert manager.snapshot(created["run_id"])["status"] == "CANCELLED"
    assert sessions.get(created["session_id"]).active_run_id is None
    assert manager.event_store.count(created["run_id"]) == event_count
    assert all(not item["transitioned"] for item in second if item["run_id"] == created["run_id"])


def test_recovery_uses_legacy_run_job_binding_when_payload_has_no_run_id(
    tmp_path: Path,
) -> None:
    from app.jobs.repository import PersistentJobRepository
    from app.session import SessionManager

    repository = PersistentJobRepository(tmp_path)
    sessions = SessionManager(tmp_path)
    manager = ScholarRunManager(
        tmp_path,
        repository=repository,
        session_manager=sessions,
        event_store=RunEventStore(tmp_path),
    )
    session = sessions.create("legacy recovery", project_id="PROJECT")
    runtime = sessions.open(session.session_id)
    job, _ = repository.submit(
        "web.answer",
        workspace_id="PROJECT",
        scope_version=1,
        payload={"project_id": None, "session_id": None},
        idempotency_key="legacy-job-binding",
    )
    run_id = "RUN_LEGACY_JOB_BINDING"
    runtime.create_run(
        "research",
        run_id=run_id,
        thread_id="THREAD_LEGACY_JOB_BINDING",
        job_id=job.job_id,
        project_id="PROJECT",
    )
    runtime.start_run(run_id, worker_id="dead-worker")
    sessions.set_active_run(session.session_id, run_id)
    repository.set_status(
        job.job_id,
        "FAILED",
        error_type="ProcessRestart",
        error_summary="Worker heartbeat lost after process restart.",
    )

    manager.reconcile_durable_state()

    assert manager.snapshot(run_id)["status"] == "FAILED"
    assert sessions.get(session.session_id).active_run_id is None
