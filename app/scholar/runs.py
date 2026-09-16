"""Asynchronous Scholar Run lifecycle and Worker boundary.

The API creates a durable Run and a durable Job.  Only the Worker calls the
orchestration backend.  The request text remains in the Run Store (the
authoritative domain record); queue payloads contain references and bounded
routing metadata only.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Mapping

from app.generation.security import redact_sensitive_text
from app.jobs.repository import JobRecord, PersistentJobRepository
from app.jobs.registry import WorkerRegistry
from app.jobs.worker import JobCancelled, JobExecutionContext, PersistentJobWorker
from app.orchestration.contracts import OrchestrationRequest
from app.scholar.events import RunEventStore
from app.session import SessionManager


_ROUTES = {
    "RESEARCH",
    "SUPPORT_CLAIM",
    "WRITE_INTRODUCTION",
    "WRITE_CONCLUSION",
    "WRITE_ABSTRACT",
    "REVIEW",
}
_TERMINAL_RUNS = {"COMPLETED", "FAILED", "CANCELLED", "WAITING_USER"}
_SENSITIVE = ("prompt", "secret", "token", "password", "api_key", "raw_output", "pdf")


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_metadata(child)
            for key, child in value.items()
            if not any(marker in str(key).casefold() for marker in _SENSITIVE)
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_safe_metadata(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:2000]
    return str(value)[:500]


class ScholarRunManager:
    """Create/query/cancel Scholar Runs without executing agent cognition."""

    def __init__(
        self,
        project_root: Path,
        *,
        repository: PersistentJobRepository | None = None,
        session_manager: SessionManager | None = None,
        event_store: RunEventStore | None = None,
        worker_registry: WorkerRegistry | None = None,
        orchestration: Any | None = None,
        runtime_factory: Any | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.repository = repository or PersistentJobRepository(self.project_root)
        self.session_manager = session_manager or SessionManager(self.project_root)
        self.event_store = event_store or RunEventStore(self.project_root)
        self.worker_registry = worker_registry or WorkerRegistry(self.project_root)
        self.orchestration = orchestration
        self.runtime_factory = runtime_factory
        self._bundle: Any | None = None

    def _runtime_for(self, session_id: str) -> Any:
        return self.session_manager.open(session_id)

    def _existing_job(self, idempotency_key: str) -> JobRecord | None:
        getter = getattr(self.repository, "get_by_idempotency", None)
        if callable(getter):
            return getter(idempotency_key)
        for job in self.repository.list():
            if job.idempotency_key == idempotency_key:
                return job
        return None

    def _mark_duplicate_run_cancelled(
        self,
        runtime: Any,
        run_id: str,
        *,
        project_id: str,
        session_id: str,
        winner_run_id: str,
    ) -> None:
        """Close a losing Run created during an idempotency race.

        The repository's unique idempotency key is the authority. The losing
        Run is kept inspectable and explicitly cancelled instead of leaving an
        orphaned PENDING record that would look like work still waiting.
        """

        try:
            runtime.complete_run(
                run_id,
                status="CANCELLED",
                answer="",
                citations=[],
                evidence=[],
                metadata={
                    "orchestration_status": "CANCELLED",
                    "termination_reason": "IDEMPOTENCY_RACE",
                    "winner_run_id": winner_run_id,
                },
            )
            self.event_store.append(
                run_id=run_id,
                project_id=project_id,
                session_id=session_id,
                event_type="RUN_CANCELLED",
                node="Scholar Run",
                status="CANCELLED",
                summary="Duplicate idempotency request cancelled",
                metadata={
                    "termination_reason": "IDEMPOTENCY_RACE",
                    "winner_run_id": winner_run_id,
                },
                event_id=f"{run_id}:cancelled:idempotency",
            )
        except (KeyError, OSError, ValueError):
            # The winning Job remains authoritative. A projection cleanup
            # failure must not make a successful idempotent retry fail.
            return

    def create(
        self,
        request: OrchestrationRequest,
        *,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        key = (idempotency_key or f"scholar.run:{request.request_id}").strip()
        if not key:
            raise ValueError("idempotency_key 不能为空。")
        existing_job = self._existing_job(key)
        if existing_job is not None:
            run_id = str(existing_job.payload.get("run_id") or "")
            if run_id:
                return self.snapshot(run_id)

        session = self.session_manager.resolve(
            request.session_id,
            title=request.instruction[:120],
            project_id=request.project_id,
        )
        runtime = self._runtime_for(session.session_id)
        run_id = f"RUN_{secrets.token_hex(8)}"
        thread_id = request.thread_id or f"crew_{run_id}"
        trace_id = f"TRACE_{secrets.token_hex(8)}"
        runtime.create_run(
            request.instruction,
            run_id=run_id,
            thread_id=thread_id,
            trace_id=trace_id,
            project_id=request.project_id,
        )
        payload = {
            "request_id": request.request_id,
            "project_id": request.project_id,
            "session_id": session.session_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "trace_id": trace_id,
            "task_type": request.task_type,
            "metadata": _safe_metadata(request.metadata),
        }
        try:
            job, created = self.repository.submit(
                "scholar.run",
                workspace_id=request.project_id,
                scope_version=1,
                payload=payload,
                idempotency_key=key,
                max_attempts=max_attempts,
            )
            if not created:
                # The race-safe repository winner owns the Run.  Remove no
                # data here; close the losing local Run explicitly and return
                # the winner's durable identity instead.
                winner_id = str(job.payload.get("run_id") or "")
                if winner_id:
                    if winner_id != run_id:
                        self._mark_duplicate_run_cancelled(
                            runtime,
                            run_id,
                            project_id=request.project_id,
                            session_id=session.session_id,
                            winner_run_id=winner_id,
                        )
                    return self.snapshot(winner_id)
            runtime.bind_job(run_id, job.job_id)
            runtime.queue_run(run_id, job_id=job.job_id)
            self.event_store.append(
                run_id=run_id,
                project_id=request.project_id,
                session_id=session.session_id,
                event_type="RUN_STARTED",
                node="Scholar Run",
                status="PENDING",
                summary="Scholar Run queued",
                metadata={"lifecycle": "QUEUED", "job_id": job.job_id, "backend": "crewai"},
                event_id=f"{run_id}:created",
            )
        except Exception:
            # The Run is intentionally left inspectable if queue persistence
            # fails; callers can retry with the same request key safely.
            raise
        return self.snapshot(run_id)

    def _locate(self, run_id: str) -> tuple[Any, Any]:
        requested = run_id.strip()
        if not requested:
            raise KeyError("Run ID 不能为空。")
        for session in self.session_manager.list(include_deleted=False):
            runtime = self.session_manager.open(session.session_id)
            try:
                return runtime, runtime.get_run(requested)
            except KeyError:
                continue
        raise KeyError(f"Run 不存在：{requested}")

    def _locate_job_run(self, job: JobRecord) -> tuple[Any, Any] | None:
        """Locate a Run from new payloads or legacy ``agent_runs.job_id``."""

        payload_run_id = str(job.payload.get("run_id") or "").strip()
        if payload_run_id:
            try:
                return self._locate(payload_run_id)
            except KeyError:
                pass
        # Older Jobs only retained project/session routing fields.  The
        # Session DB still has the durable reverse binding, so use it before
        # declaring the Job orphaned.
        for session in self.session_manager.list(include_deleted=False):
            runtime = self.session_manager.open(session.session_id)
            for run in runtime.list_runs():
                if run.job_id == job.job_id:
                    return runtime, run
        return None

    def snapshot(self, run_id: str) -> dict[str, Any]:
        runtime, run = self._locate(run_id)
        result = next(
            (
                item
                for item in runtime.list_results()
                if isinstance(item, Mapping) and item.get("run_id") == run.run_id
            ),
            {},
        )
        metadata = result.get("metadata") if isinstance(result, Mapping) else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        job = None
        if run.job_id:
            try:
                job = self.repository.get(run.job_id)
            except KeyError:
                job = None
        job_payload = job.payload if job is not None else {}
        task_type = metadata.get("task_type") or job_payload.get("task_type")
        return {
            "run_id": run.run_id,
            "session_id": run.session_id,
            "project_id": run.project_id,
            "thread_id": run.thread_id,
            "trace_id": run.trace_id,
            "status": "WAITING_HUMAN_APPROVAL" if run.status == "WAITING_USER" else run.status,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "worker_id": run.worker_id,
            "job_id": run.job_id,
            "job_status": job.status if job is not None else None,
            "attempt": job.attempt if job is not None else 0,
            "error": run.failure_message or (job.error_summary if job is not None else None),
            "event_cursor": self.event_store.count(run.run_id),
            # These fields let the Web Console render a useful history row
            # without reopening the Session DB just to discover its route.
            "title": run.query[:120],
            "task_type": task_type,
            "selected_skill": metadata.get("selected_skill"),
            "detail_available": True,
        }

    def list_runs(self, *, project_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        sessions = self.session_manager.list(include_deleted=False)
        for session in sessions:
            if session_id and session.session_id != session_id:
                continue
            if project_id and session.project_id and session.project_id != project_id:
                continue
            runtime = self.session_manager.open(session.session_id)
            for run in runtime.list_runs():
                if project_id and run.project_id and run.project_id != project_id:
                    continue
                values.append(self.snapshot(run.run_id))
        return sorted(values, key=lambda item: (str(item.get("created_at")), str(item.get("run_id"))))

    def cancel(self, run_id: str) -> dict[str, Any]:
        _, run = self._locate(run_id)
        if run.job_id:
            self.repository.request_cancel(run.job_id)
        runtime, current = self._locate(run_id)
        if current.status in {"PENDING", "QUEUED", "WAITING_USER"}:
            runtime.complete_run(
                run_id,
                status="CANCELLED",
                answer="",
                citations=[],
                evidence=[],
                metadata={"orchestration_status": "CANCELLED", "termination_reason": "CANCELLED_BY_USER"},
            )
            self.event_store.append(
                run_id=run_id,
                project_id=current.project_id or "",
                session_id=current.session_id,
                event_type="RUN_CANCELLED",
                node="Scholar Run",
                status="CANCELLED",
                summary="Scholar Run cancelled",
                metadata={"termination_reason": "CANCELLED_BY_USER"},
                event_id=f"{run_id}:cancelled",
            )
            self.session_manager.clear_active_run_if(current.session_id, run_id)
        return self.snapshot(run_id)

    def reconcile_job(self, job: JobRecord) -> dict[str, Any] | None:
        """Reconcile a durable Scholar Job with its Session Run.

        Job and Run are stored separately, so a Worker can disappear between
        either write.  This method is intentionally idempotent and only
        advances a Run when the Job is authoritative for that transition.
        """

        located = self._locate_job_run(job)
        if located is None:
            return None
        runtime, run = located
        run_id = run.run_id

        target: str | None = None
        if job.status == "CANCELLED" and run.status not in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }:
            target = "CANCELLED"
        elif job.status == "FAILED" and run.status not in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }:
            # A terminal Job failure is the durable decision once retries are
            # exhausted, including ProcessRestart failures.
            target = "FAILED"
        elif job.status in {"INTERRUPTED", "RETRY_PENDING"} and run.status == "RUNNING":
            target = "INTERRUPTED"

        if target is not None:
            runtime.complete_run(
                run_id,
                status=target,  # type: ignore[arg-type]
                answer="",
                citations=[],
                evidence=[],
                metadata={
                    "backend": "crewai",
                    "orchestration_status": target,
                    "reconciled_from_job": job.status,
                    "job_id": job.job_id,
                    "job_error_type": job.error_type,
                    "job_error_summary": job.error_summary,
                },
            )
            event_type, summary = {
                "CANCELLED": ("RUN_CANCELLED", "Run cancelled after Job reconciliation"),
                "FAILED": ("RUN_FAILED", "Run failed after Job reconciliation"),
                "INTERRUPTED": ("RUN_INTERRUPTED", "Run interrupted after Worker recovery"),
            }[target]
            self.event_store.append(
                run_id=run_id,
                project_id=str(run.project_id or job.workspace_id),
                session_id=run.session_id,
                event_type=event_type,
                node="Scholar Run",
                status=target,
                summary=summary,
                metadata={
                    "job_id": job.job_id,
                    "job_status": job.status,
                    "job_error_type": job.error_type,
                    "termination_reason": (
                        "CANCELLED_BY_USER"
                        if target == "CANCELLED"
                        else job.error_type or "JOB_RECONCILIATION"
                    ),
                },
                event_id=f"{run_id}:reconciled:{event_type}",
            )
            run = runtime.get_run(run_id)

        if run.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            self.session_manager.clear_active_run_if(run.session_id, run.run_id)
        return {
            "run_id": run.run_id,
            "job_id": job.job_id,
            "job_status": job.status,
            "run_status": run.status,
            "transitioned": target is not None,
        }

    def reconcile_durable_state(self) -> tuple[dict[str, Any], ...]:
        """Reconcile all persisted Scholar Jobs and stale active pointers."""

        values = tuple(
            result
            for job in self.repository.list()
            for result in (self.reconcile_job(job),)
            if result is not None
        )
        self.session_manager.reconcile_terminal_active_runs()
        return values

    def enqueue_resume(
        self,
        run_id: str,
        *,
        resume_value: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        runtime, run = self._locate(run_id)
        if run.status in _TERMINAL_RUNS - {"WAITING_USER"}:
            return self.snapshot(run_id)
        key = (idempotency_key or f"scholar.resume:{run_id}:{run.completed_at or 'pending'}").strip()
        existing = self._existing_job(key)
        if existing is not None:
            return self.snapshot(run_id)
        job, created = self.repository.submit(
            "scholar.resume",
            workspace_id=str(run.project_id or ""),
            scope_version=1,
            payload={
                "run_id": run_id,
                "project_id": run.project_id,
                "session_id": run.session_id,
                "thread_id": run.thread_id,
                "task_type": None,
                "resume_value": _safe_metadata(resume_value),
            },
            idempotency_key=key,
            max_attempts=3,
        )
        runtime.queue_run(run_id, job_id=job.job_id)
        self.event_store.append(
            run_id=run_id,
            project_id=str(run.project_id or ""),
            session_id=run.session_id,
            event_type="RUN_RESUMED",
            node="Scholar Run",
            status="PENDING",
            summary="Scholar Run resume queued",
            metadata={"job_id": job.job_id},
            event_id=f"{run_id}:resume:{job.job_id}",
        )
        return self.snapshot(run_id)

    def _runner(self) -> Any:
        if self.orchestration is not None:
            return self.orchestration
        if self._bundle is None:
            if self.runtime_factory is not None:
                factory = self.runtime_factory() if callable(self.runtime_factory) else self.runtime_factory
            else:
                from app.scholar.composition import ScholarRuntimeFactory

                factory = ScholarRuntimeFactory(
                    self.project_root,
                    orchestration_backend=os.getenv("ORCHESTRATION_BACKEND") or "crewai",
                )
            self._bundle = factory.build() if callable(getattr(factory, "build", None)) else factory
        return getattr(self._bundle, "scholar_orchestration", self._bundle)

    def execute(self, job: JobRecord, context: JobExecutionContext) -> str:
        payload = dict(job.payload)
        run_id = str(payload.get("run_id") or "")
        runtime, run = self._locate(run_id)
        if run.status in _TERMINAL_RUNS:
            return run_id
        worker_id = context.worker_id
        if run.status in {"PENDING", "QUEUED", "INTERRUPTED"}:
            try:
                runtime.start_run(run_id, worker_id=worker_id)
            except ValueError:
                run = runtime.get_run(run_id)
        self.event_store.append(
            run_id=run_id,
            project_id=str(payload.get("project_id") or run.project_id or ""),
            session_id=str(payload.get("session_id") or run.session_id),
            event_type="DOMAIN_RESULT",
            node="CrewAI Worker",
            status="RUNNING",
            summary="CrewAI Worker started Scholar Run",
            metadata={"job_id": job.job_id, "worker_id": worker_id, "attempt": job.attempt},
            event_id=f"{run_id}:worker:{job.attempt}",
        )
        request = OrchestrationRequest(
            request_id=str(payload.get("request_id") or run_id),
            project_id=str(payload.get("project_id") or run.project_id or ""),
            instruction=run.query,
            session_id=run.session_id,
            task_type=payload.get("task_type") if payload.get("task_type") in _ROUTES else None,
            thread_id=run.thread_id,
            metadata={
                **(_safe_metadata(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}),
                "_run_id": run_id,
                "_job_id": job.job_id,
                "_worker_id": worker_id,
                # Keep cancellation live-only. It is intentionally attached
                # after the durable payload is read and is stripped by the
                # orchestration backend before any result is persisted.
                "_cancellation_checker": context.raise_if_cancelled,
            },
        )
        try:
            result = self._runner().run(request)
            context.checkpoint({"stage": "orchestration_result", "run_id": run_id})
            # CrewAIBackend persists its own canonical result.  For injected
            # test/fallback runners, provide the same projection here.
            current = runtime.get_run(run_id)
            if current.status not in {"COMPLETED", "FAILED", "CANCELLED", "WAITING_USER", "INTERRUPTED"}:
                status = "WAITING_USER" if getattr(result, "status", "") == "WAITING_HUMAN_APPROVAL" else "FAILED" if getattr(result, "status", "") == "FAILED" else "COMPLETED"
                value = getattr(result, "final_answer", None) or ""
                runtime.complete_run(
                    run_id,
                    status=status,
                    answer=str(value)[:50_000],
                    citations=[],
                    evidence=[],
                    metadata={"backend": "crewai", "orchestration_status": getattr(result, "status", status)},
                )
                current = runtime.get_run(run_id)
            terminal_type = {
                "WAITING_USER": ("WAITING_USER", "WAITING_USER", "Human approval required"),
                "COMPLETED": ("RUN_COMPLETED", "COMPLETED", "Run completed"),
                "INTERRUPTED": ("RUN_INTERRUPTED", "INTERRUPTED", "Run interrupted"),
                "FAILED": ("RUN_FAILED", "FAILED", "Run failed"),
                "CANCELLED": ("RUN_CANCELLED", "CANCELLED", "Run cancelled"),
            }.get(current.status)
            if terminal_type is not None:
                event_type, event_status, summary = terminal_type
                self.event_store.append(
                    run_id=run_id,
                    project_id=str(payload.get("project_id") or run.project_id or ""),
                    session_id=str(payload.get("session_id") or run.session_id),
                    event_type=event_type,
                    node="Human Approval" if current.status == "WAITING_USER" else "Scholar Run",
                    status=event_status,
                    summary=summary,
                    metadata={"job_id": job.job_id, "worker_id": worker_id},
                    event_id=f"{run_id}:terminal:{event_type}",
                )
            return run_id
        except JobCancelled:
            try:
                runtime.complete_run(
                    run_id,
                    status="CANCELLED",
                    answer="",
                    citations=[],
                    evidence=[],
                    metadata={
                        "backend": "crewai",
                        "orchestration_status": "CANCELLED",
                        "termination_reason": "CANCELLED_BY_USER",
                        "job_id": job.job_id,
                    },
                )
            except (KeyError, OSError, ValueError):
                pass
            self.event_store.append(
                run_id=run_id,
                project_id=str(payload.get("project_id") or run.project_id or ""),
                session_id=str(payload.get("session_id") or run.session_id),
                event_type="RUN_CANCELLED",
                node="Scholar Run",
                status="CANCELLED",
                summary="Scholar Run cancelled",
                metadata={"termination_reason": "CANCELLED_BY_USER", "job_id": job.job_id},
                event_id=f"{run_id}:cancelled:worker:{job.attempt}",
            )
            raise
        except Exception as error:
            try:
                runtime.complete_run(
                    run_id,
                    status="INTERRUPTED",
                    answer="",
                    citations=[],
                    evidence=[],
                    metadata={
                        "backend": "crewai",
                        "orchestration_status": "INTERRUPTED",
                        "error": redact_sensitive_text(str(error))[:1000],
                        "job_id": job.job_id,
                    },
                )
            except Exception:
                pass
            self.event_store.append(
                run_id=run_id,
                project_id=str(payload.get("project_id") or run.project_id or ""),
                session_id=str(payload.get("session_id") or run.session_id),
                event_type="RUN_INTERRUPTED",
                node="CrewAI Worker",
                status="INTERRUPTED",
                summary="Worker failed before Run completion",
                metadata={
                    "job_id": job.job_id,
                    "worker_id": worker_id,
                    "error_type": type(error).__name__,
                    "error": redact_sensitive_text(str(error))[:1000],
                },
                event_id=f"{run_id}:interrupted:{job.attempt}",
            )
            raise

    def close(self) -> None:
        if self._bundle is not None:
            close = getattr(self._bundle, "close", None)
            if callable(close):
                close()
            self._bundle = None


class ScholarRunWorker:
    """Independent process entrypoint backed by the existing durable queue."""

    def __init__(self, manager: ScholarRunManager, *, worker_id: str | None = None) -> None:
        self.manager = manager
        self.worker_id = worker_id or f"scholar-worker:{os.getpid()}:{secrets.token_hex(4)}"
        self.worker = PersistentJobWorker(
            manager.repository,
            {
                "scholar.run": self._run,
                "scholar.resume": self._resume,
            },
            worker_id=self.worker_id,
        )
        self.manager.worker_registry.register(
            self.worker_id,
            worker_type="scholar",
            backend="crewai",
        )

    def _run(self, job: JobRecord, context: JobExecutionContext) -> str | None:
        return self.manager.execute(job, context)

    def _resume(self, job: JobRecord, context: JobExecutionContext) -> str | None:
        payload = dict(job.payload)
        run_id = str(payload.get("run_id") or "")
        runtime, run = self.manager._locate(run_id)
        runner = self.manager._runner()
        if run.status in {"PENDING", "QUEUED", "INTERRUPTED"}:
            try:
                runtime.start_run(run_id, worker_id=context.worker_id)
            except ValueError:
                run = runtime.get_run(run_id)
        try:
            result = runner.resume(
                run.thread_id,
                payload.get("resume_value"),
                str(payload.get("project_id") or run.project_id or ""),
                instruction=run.query,
                session_id=run.session_id,
                task_type=payload.get("task_type"),
            )
            context.checkpoint({"stage": "resume_result", "run_id": run_id})
            current = runtime.get_run(run_id)
            # Real CrewAI reconciliation persists through its backend.  Keep
            # the Worker boundary equally correct for injected/fallback
            # runners, otherwise a successful resume could leave the Run in
            # RUNNING forever after the Job itself succeeds.
            if current.status not in {"COMPLETED", "FAILED", "CANCELLED", "WAITING_USER", "INTERRUPTED"}:
                result_status = str(getattr(result, "status", "COMPLETED"))
                session_status = (
                    "WAITING_USER"
                    if result_status == "WAITING_HUMAN_APPROVAL"
                    else "FAILED"
                    if result_status == "FAILED"
                    else "INTERRUPTED"
                    if result_status == "INTERRUPTED"
                    else "COMPLETED"
                )
                runtime.complete_run(
                    run_id,
                    status=session_status,
                    answer=str(getattr(result, "final_answer", None) or "")[:50_000],
                    citations=[],
                    evidence=[],
                    metadata={
                        "backend": "crewai",
                        "orchestration_status": result_status,
                        "resume_job_id": job.job_id,
                    },
                )
                event_type, event_status, summary = {
                    "WAITING_USER": ("WAITING_USER", "WAITING_USER", "Human approval required"),
                    "COMPLETED": ("RUN_COMPLETED", "COMPLETED", "Run completed"),
                    "INTERRUPTED": ("RUN_INTERRUPTED", "INTERRUPTED", "Run interrupted"),
                    "FAILED": ("RUN_FAILED", "FAILED", "Run failed"),
                }[session_status]
                self.manager.event_store.append(
                    run_id=run_id,
                    project_id=str(payload.get("project_id") or run.project_id or ""),
                    session_id=str(payload.get("session_id") or run.session_id),
                    event_type=event_type,
                    node="Human Approval" if session_status == "WAITING_USER" else "Scholar Run",
                    status=event_status,
                    summary=summary,
                    metadata={"job_id": job.job_id, "worker_id": context.worker_id},
                    event_id=f"{run_id}:terminal:{event_type}",
                )
                if session_status != "WAITING_USER":
                    self.manager.session_manager.set_active_run(run.session_id, None)
            return run_id if result is not None else None
        except Exception as error:
            try:
                runtime.complete_run(
                    run_id,
                    status="INTERRUPTED",
                    answer="",
                    citations=[],
                    evidence=[],
                    metadata={"backend": "crewai", "orchestration_status": "INTERRUPTED", "error": redact_sensitive_text(str(error))[:1000]},
                )
            except Exception:
                pass
            raise

    def recover_after_restart(self, *, stale_after_seconds: float = 30.0) -> tuple[JobRecord, ...]:
        recovered = self.worker.recover_after_restart(stale_after_seconds=stale_after_seconds)
        self.manager.reconcile_durable_state()
        return recovered

    def run_once(self) -> JobRecord | None:
        self.manager.worker_registry.heartbeat(self.worker_id)
        result = self.worker.run_once()
        if result is not None:
            self.manager.reconcile_job(result)
        return result

    def heartbeat(self) -> dict[str, Any]:
        """Publish an idle heartbeat for API worker-status projections."""

        return self.manager.worker_registry.heartbeat(self.worker_id)

    def run_until_idle(self, *, max_jobs: int = 100) -> tuple[JobRecord, ...]:
        results: list[JobRecord] = []
        for _ in range(max_jobs):
            result = self.run_once()
            if result is None:
                break
            results.append(result)
        return tuple(results)

    def status(self) -> dict[str, Any]:
        jobs = self.manager.repository.list()
        active = [job for job in jobs if job.worker_id == self.worker_id and job.status == "RUNNING"]
        return {
            "worker_id": self.worker_id,
            "status": "RUNNING",
            "active": len(active),
            "active_job_ids": [job.job_id for job in active],
            "queue_depth": sum(job.status in {"QUEUED", "RETRY_PENDING"} and job.job_type.startswith("scholar.") for job in jobs),
            "interrupted": sum(job.status == "INTERRUPTED" for job in jobs),
        }

    def close(self) -> None:
        self.manager.worker_registry.stop(self.worker_id)
        self.manager.close()
