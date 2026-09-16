"""LEO Research Agent 的本地 FastAPI 入口。"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import tempfile
import socket
from urllib.parse import urlparse
from urllib.request import urlopen
from dataclasses import asdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import os
from typing import Any, Literal, Protocol

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.ingestion.ingest import sanitize_pdf_filename
from app.scholar.approval import (
    LatexBridgeService,
    PatchApprovalRequest,
    PatchApprovalError,
    PatchApprovalService,
)
from app.scholar.approval.build import normalize_diagnostic
from app.scholar.project import ScholarProjectStore
from app.scholar.events import RunEventStore
from app.scholar.runs import ScholarRunManager
from app.scholar.errors import ScholarRuntimeError
from app.orchestration.contracts import OrchestrationRequest
from app.web.jobs import JobManager
from app.web.models import (
    AnswerRequest,
    BuildReportRequest,
    JobCreated,
    JobSnapshot,
    ParseOptions,
    PatchDecisionRequest,
    ResumeRequest,
    ScholarResumeRequest,
    ScholarRunCreateRequest,
    ScholarRunResumeRequest,
    ScholarTaskRequest,
)
from app.web.runtime import EmitProgress, LocalRAGWebRuntime
from app.scholar.console import ScholarConsoleProjection, demo_console_payload
from app.scholar.manuscript import ManuscriptSynchronizer


MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class WebRuntime(Protocol):
    """API 可注入的业务契约，测试时不需加载大模型。"""

    project_root: Path

    def answer(
        self, request: AnswerRequest, emit: EmitProgress
    ) -> dict[str, Any]: ...

    def parse_pdf(
        self, pdf_path: Path, options: ParseOptions, emit: EmitProgress
    ) -> dict[str, Any]: ...

    def list_papers(self) -> dict[str, Any]: ...

    def list_sessions(self) -> dict[str, Any]: ...

    def session_details(self, session_id: str) -> dict[str, Any]: ...

    def session_evidence(self, session_id: str) -> dict[str, Any]: ...

    def session_transcript(self, session_id: str) -> dict[str, Any]: ...

    def compact_session(self, session_id: str) -> dict[str, Any]: ...

    def public_status(self) -> dict[str, Any]: ...


def _http_error(error: Exception) -> HTTPException:
    # Patch approval errors also expose ``code``.  Handle them before the
    # generic runtime-code mapping so PATCH_NOT_FOUND keeps its established
    # 404 contract instead of being flattened into a generic 400.
    if isinstance(error, PatchApprovalError):
        status = 404 if error.code == "PATCH_NOT_FOUND" else 409
        return HTTPException(status_code=status, detail={"code": error.code, "message": str(error)})
    code = getattr(error, "code", None)
    if isinstance(code, str):
        status_by_code = {
            "CAPABILITY_DENIED": 403,
            "CAPABILITY_VIOLATION": 403,
            "CHECKPOINT_UNAVAILABLE": 503,
            "CONFIGURATION_ERROR": 503,
            "PROVIDER_UNAVAILABLE": 503,
            "RESUME_UNAVAILABLE": 409,
            "DOMAIN_CONFLICT": 409,
            "RUN_CORRELATION_INVALID": 409,
            "SESSION_CONFLICT": 409,
        }
        return HTTPException(
            status_code=status_by_code.get(code, 400),
            detail={"code": code, "message": str(error)},
        )
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error).strip("'"))
    if isinstance(error, (OSError, ValueError)):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=type(error).__name__)


def _probe_tcp(url: str, *, default_port: int) -> str:
    """Return a bounded local readiness state without exposing credentials."""

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or default_port
    if not host:
        return "invalid"
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return "ready"
    except (OSError, ValueError):
        return "unavailable"


def _probe_http(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.hostname:
        return "invalid"
    health_url = url.rstrip("/") + "/healthz"
    try:
        with urlopen(health_url, timeout=0.8) as response:  # noqa: S310 - URL is deployment configuration.
            return "ready" if 200 <= int(response.status) < 400 else "unavailable"
    except (OSError, ValueError):
        return "unavailable"


def create_app(
    project_root: Path | None = None,
    *,
    runtime: WebRuntime | None = None,
    jobs: JobManager | None = None,
    approval_service: PatchApprovalService | None = None,
    latex_bridge: LatexBridgeService | None = None,
    scholar_harness: Any | None = None,
    scholar_run_manager: ScholarRunManager | None = None,
) -> FastAPI:
    """创建可测试的 API；默认使用仓库根目录和本地长驻 Runtime。"""

    configured_root = os.getenv("LEO_PROJECT_ROOT")
    root = (
        project_root
        or (Path(configured_root) if configured_root else Path(__file__).resolve().parents[2])
    ).resolve()
    supplied_runtime = runtime is not None
    web_runtime: WebRuntime = runtime or LocalRAGWebRuntime(root)
    production_factory: Any | None = None
    if not supplied_runtime and scholar_harness is None:
        # The default deployment path owns one complete Scholar Runtime. Test
        # callers can still inject a lightweight WebRuntime/Harness pair.
        from app.scholar.composition import ScholarRuntimeFactory

        production_factory = ScholarRuntimeFactory(root)
    job_manager = jobs or JobManager(max_workers=2, project_root=root)
    project_store = ScholarProjectStore(root)
    patch_approval = approval_service or PatchApprovalService(
        root,
        project_store=project_store,
    )
    build_bridge = latex_bridge or LatexBridgeService(
        root,
        project_store=project_store,
    )
    scholar_runs = scholar_run_manager or ScholarRunManager(
        root,
        repository=job_manager.repository,
        event_store=RunEventStore(root),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if production_factory is not None:
                bundle = production_factory.build()
                app.state.scholar_runtime = bundle
                app.state.scholar_harness = bundle.harness
                app.state.scholar_orchestration = bundle.scholar_orchestration
                app.state.scholar_console = ScholarConsoleProjection(
                    root,
                    project_store=bundle.project_store,
                    session_manager=bundle.session_manager,
                )
                scholar_runs.session_manager = bundle.session_manager
                scholar_runs.orchestration = bundle.scholar_orchestration
            yield
        finally:
            if production_factory is not None:
                production_factory.close()
                app.state.scholar_runtime = None
            job_manager.close()
            close_runtime = getattr(web_runtime, "close", None)
            if callable(close_runtime):
                close_runtime()

    app = FastAPI(
        title="LEO Research Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = web_runtime
    app.state.jobs = job_manager
    app.state.patch_approval = patch_approval
    app.state.latex_bridge = build_bridge
    app.state.scholar_harness = scholar_harness or getattr(web_runtime, "scholar_harness", None)
    app.state.scholar_orchestration = None
    app.state.scholar_runtime = None
    app.state.scholar_runtime_factory = production_factory
    app.state.scholar_console = ScholarConsoleProjection(root, project_store=project_store)
    app.state.scholar_runs = scholar_runs

    def _reconcile_approval_run(patch_id: str, decision_status: str) -> dict[str, Any] | None:
        """Queue the lifecycle reconciliation after the side effect is durable.

        PatchApprovalService remains the only component allowed to apply a
        manuscript change.  This hook only projects the completed human
        decision into the originating asynchronous Run so the normal Worker
        can close the Run after an API restart or a disconnected browser.
        """

        if decision_status not in {"APPLIED", "REJECTED", "CONFLICT", "FAILED", "BUILD_FAILED"}:
            return None
        try:
            stored = project_store.get_patch(patch_id)
        except KeyError:
            return None
        source_run_id = getattr(stored, "source_run_id", None)
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            return None
        try:
            run_status = app.state.scholar_runs.snapshot(source_run_id).get("status")
            if run_status not in {"WAITING_HUMAN_APPROVAL", "WAITING_USER", "QUEUED"}:
                return {
                    "run_id": source_run_id,
                    "status": run_status,
                    "queued": False,
                }
            event_status = "COMPLETED" if decision_status in {"APPLIED", "REJECTED"} else "FAILED"
            app.state.scholar_runs.event_store.append(
                run_id=source_run_id,
                project_id=project_store.project_id,
                session_id=getattr(stored, "source_session_id", None),
                event_type="DOMAIN_RESULT",
                node="Human Approval",
                status=event_status,
                summary=f"Human approval decision: {decision_status}",
                metadata={"patch_id": patch_id, "decision_status": decision_status},
                event_id=f"{source_run_id}:approval:{patch_id}:{decision_status}",
            )
            queued = app.state.scholar_runs.enqueue_resume(
                source_run_id,
                resume_value={"approval_status": decision_status},
                idempotency_key=f"scholar.approval:{patch_id}:{decision_status}",
            )
            return {
                "run_id": source_run_id,
                "status": queued.get("status"),
                "job_id": queued.get("job_id"),
                "queued": True,
            }
        except (KeyError, OSError, ValueError) as error:
            # The patch decision is already durable.  Expose the reconciliation
            # failure without rolling back or masking the approval result.
            return {
                "run_id": source_run_id,
                "queued": False,
                "error": type(error).__name__,
            }
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    def process_health() -> dict[str, Any]:
        """Liveness only; an unavailable third-party provider is not a crash."""

        return {"status": "ok", "service": "leo-research-agent"}

    @app.get("/ready")
    def readiness() -> Response:
        try:
            app.state.scholar_runs.repository.list()
            from app.knowledge.corpus import knowledge_index_readiness

            backend = (
                os.getenv("ORCHESTRATION_BACKEND")
                or os.getenv("LEO_AGENTIC_ORCHESTRATION_BACKEND")
                or "crewai"
            ).strip().lower()
            if backend not in {"crewai", "legacy"}:
                raise ValueError("invalid orchestration backend")
            workers = list(app.state.scholar_runs.worker_registry.list())
            live_workers = [item for item in workers if item.get("status") == "RUNNING"]
            redis_url = os.getenv("REDIS_URL")
            qdrant_url = os.getenv("QDRANT_URL")
            redis_status = _probe_tcp(redis_url, default_port=6379) if redis_url else "not_configured"
            qdrant_status = _probe_http(qdrant_url) if qdrant_url else "not_configured"
            external_provider = "configured" if os.getenv("LEO_LLM_BASE_URL") else "deferred"
            knowledge_index = knowledge_index_readiness(root)
            components = {
                "run_store": "ready",
                "queue_store": "ready",
                "redis": redis_status,
                "qdrant": qdrant_status,
                "worker": "ready" if live_workers else "waiting",
                "crewai_backend": "ready" if backend == "crewai" else "fallback",
                "provider": external_provider,
                "knowledge_index": knowledge_index["status"],
            }
            required_components = {"run_store", "queue_store", "redis", "qdrant"}
            unavailable = {
                name
                for name in required_components
                if components[name] in {"invalid", "unavailable"}
            }
            if unavailable:
                return JSONResponse(
                    {
                        "status": "not_ready",
                        "components": components,
                        "knowledge_index": knowledge_index,
                        "unavailable": sorted(unavailable),
                        "research_readiness": "not_ready",
                        "application_runtime": "ready",
                    },
                    status_code=503,
                )
            research_ready = (
                knowledge_index["status"] == "ready"
                and external_provider == "configured"
            )
            degraded = knowledge_index["status"] != "ready" or external_provider != "configured"
            return JSONResponse(
                {
                    "status": "degraded" if degraded else "ready",
                    "orchestration_backend": backend,
                    "queue": "available",
                    "components": components,
                    "knowledge_index": knowledge_index,
                    "worker_count": len(live_workers),
                    "external_provider": external_provider,
                    "application_runtime": "ready",
                    "research_readiness": "ready" if research_ready else "degraded",
                },
                status_code=200,
            )

        except Exception as error:
            return JSONResponse(
                {"status": "not_ready", "error": type(error).__name__},
                status_code=503,
            )

    @app.get("/api/metrics")
    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        from app.observability.metrics import build_metrics_snapshot

        return build_metrics_snapshot(root, run_manager=app.state.scholar_runs)
    @app.get("/api/scholar/patches/{patch_id}")
    def patch_preview(patch_id: str) -> dict[str, Any]:
        try:
            preview = patch_approval.get_preview(patch_id)
            return {"preview": asdict(preview)}
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/requests")
    def scholar_request(request: ScholarTaskRequest) -> dict[str, Any]:
        orchestrator = app.state.scholar_orchestration
        harness = app.state.scholar_harness
        if orchestrator is None and harness is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "SCHOLAR_HARNESS_NOT_CONFIGURED", "message": "Scholar Harness 未配置。"},
            )
        try:
            runner = orchestrator or harness
            result = runner.scholar_request(
                request.instruction,
                request.project_id,
                session_id=request.session_id,
                task_type=request.task_type,
                thread_id=request.thread_id,
            )
            return result.to_dict() if callable(getattr(result, "to_dict", None)) else dict(result)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/requests/resume")
    def resume_scholar_request(request: ScholarResumeRequest) -> dict[str, Any]:
        orchestrator = app.state.scholar_orchestration
        harness = app.state.scholar_harness
        if orchestrator is None and harness is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "SCHOLAR_HARNESS_NOT_CONFIGURED", "message": "Scholar Harness 未配置。"},
            )
        try:
            runner = orchestrator or harness
            result = runner.resume(
                request.thread_id,
                request.resume_value,
                request.project_id,
                instruction=request.instruction,
                session_id=request.session_id,
                task_type=request.task_type,
            )
            return result.to_dict() if callable(getattr(result, "to_dict", None)) else dict(result)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/runs", status_code=202)
    def create_scholar_run(
        request: ScholarRunCreateRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        """Persist and enqueue a Run; the API process never runs an Agent."""

        try:
            key = request.idempotency_key or idempotency_key or f"scholar.run:{secrets.token_hex(12)}"
            contract = OrchestrationRequest(
                request_id=key,
                project_id=request.project_id,
                instruction=request.instruction,
                session_id=request.session_id,
                task_type=request.task_type,
                thread_id=request.thread_id,
                metadata=request.metadata,
            )
            return app.state.scholar_runs.create(contract, idempotency_key=key)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/runs")
    def list_scholar_runs(
        project_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            runs = app.state.scholar_runs.list_runs(project_id=project_id, session_id=session_id)
            # The durable Job/Run list can outlive an older Session projection
            # after a manual migration or an interrupted storage restore. Do
            # not advertise such a row as normally openable; preserve it as an
            # explicitly marked historical record for auditability.
            for run in runs:
                try:
                    app.state.scholar_console.run_snapshot(str(run["run_id"]))
                except Exception as error:
                    run["detail_available"] = False
                    run["availability_message"] = "历史运行数据不可用，无法打开详情。"
                    run["availability_error_type"] = type(error).__name__
            return {"runs": runs}
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/runs/{run_id}/cancel", status_code=202)
    def cancel_scholar_run(run_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_runs.cancel(run_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/runs/{run_id}/stop", status_code=202)
    def stop_scholar_run(run_id: str) -> dict[str, Any]:
        """Explicit user-facing stop entrypoint for the Scholar console.

        ``cancel`` remains available for backwards compatibility.  ``stop``
        makes the intent unambiguous in the UI while sharing the same durable,
        cooperative cancellation lifecycle in the Run Manager.
        """

        try:
            return app.state.scholar_runs.cancel(run_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/runs/{run_id}/resume", status_code=202)
    def resume_scholar_run(
        run_id: str,
        request: ScholarRunResumeRequest | None = None,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        try:
            body = request or ScholarRunResumeRequest()
            return app.state.scholar_runs.enqueue_resume(
                run_id,
                resume_value=body.resume_value,
                idempotency_key=body.idempotency_key or idempotency_key,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/runs/{run_id}")
    def scholar_run_snapshot(run_id: str) -> dict[str, Any]:
        try:
            projection = app.state.scholar_console.run_snapshot(run_id)
            projection["async_run"] = app.state.scholar_runs.snapshot(run_id)
            # The durable Job carries the selected route before the Worker
            # writes its final result metadata. Keep the live detail view and
            # its evaluation contract routed while the Run is still running.
            async_run = projection["async_run"]
            routing = projection.setdefault("routing", {})
            if isinstance(routing, dict):
                routing["task_type"] = routing.get("task_type") or async_run.get("task_type")
            return projection
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/runs/{run_id}/events")
    async def scholar_run_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            app.state.scholar_console.run_snapshot(run_id)
        except Exception as error:
            raise _http_error(error) from error

        async def stream() -> AsyncIterator[str]:
            last_cursor = after
            sent: set[str] = set()
            while True:
                persisted = list(app.state.scholar_runs.event_store.list(run_id, after=last_cursor))
                events = persisted or [
                    event
                    for event in app.state.scholar_console.run_events(run_id)
                    if int(event.get("cursor", 0)) > last_cursor
                ]
                for event in events:
                    event_id = str(event.get("event_id") or f"{run_id}:{event.get('cursor')}")
                    cursor = int(event.get("cursor") or 0)
                    if event_id in sent or cursor <= last_cursor:
                        continue
                    sent.add(event_id)
                    last_cursor = max(last_cursor, cursor)
                    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    yield f"id: {cursor}\nevent: scholar_run\ndata: {payload}\n\n"
                snapshot = app.state.scholar_console.run_snapshot(run_id)
                raw_status = str(snapshot.get("run", {}).get("status") or "")
                terminal = raw_status in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED", "WAITING_USER", "WAITING_HUMAN_APPROVAL"}
                if terminal:
                    yield f"event: end\ndata: {json.dumps({'status': raw_status}, ensure_ascii=False)}\n\n"
                    break
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/scholar/projects/{project_id}/state")
    def scholar_project_state(project_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_console.project_state(project_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/projects/{project_id}/manuscript")
    def scholar_manuscript(project_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_console.manuscript_view(project_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/projects/{project_id}/manuscript/initialize")
    def initialize_scholar_manuscript(project_id: str) -> dict[str, Any]:
        """Create only the empty LaTeX project skeleton, if it is missing."""

        try:
            if project_id != project_store.project_id:
                raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
            # This operation is additive and idempotent. It never creates a
            # DraftPatch and never writes generated prose.
            ManuscriptSynchronizer(root).ensure_initialized()
            return app.state.scholar_console.manuscript_view(project_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/projects/{project_id}/manuscript/pdf")
    def scholar_manuscript_pdf(project_id: str) -> FileResponse:
        try:
            if project_id != project_store.project_id:
                raise ValueError("PROJECT_CONFLICT: Project 不属于当前 Runtime。")
            synchronizer = ManuscriptSynchronizer(root)
            state = synchronizer.scan(previous=project_store.load_manuscript_state())
            pdf_path = synchronizer._safe_path(state.root_tex).with_suffix(".pdf")
            if not pdf_path.is_file():
                raise KeyError("当前项目尚未生成 PDF，请先在 VS Code 中执行 LaTeX Workshop Build。")
            return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/projects")
    def scholar_projects() -> dict[str, Any]:
        return {
            "projects": [
                {"project_id": project_store.project_id, "root_path": str(root)}
            ]
        }

    @app.get("/api/scholar/projects/{project_id}")
    def scholar_project(project_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_console.project_state(project_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/projects/{project_id}/evidence")
    def scholar_project_evidence(project_id: str, run_id: str | None = None) -> dict[str, Any]:
        try:
            return app.state.scholar_console.evidence_view(project_id, run_id=run_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/runs/{run_id}/evaluation")
    def scholar_run_evaluation(run_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_console.evaluation(run_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/approvals")
    def scholar_approvals(project_id: str | None = Query(default=None)) -> dict[str, Any]:
        if project_id and project_id != project_store.project_id:
            raise HTTPException(status_code=409, detail="Project 不属于当前 Runtime。")
        approvals: list[dict[str, Any]] = []
        for stored in project_store.list_patches():
            patch = getattr(stored, "patch", None)
            if patch is None:
                continue
            report = getattr(stored, "review_report", None)
            approvals.append(
                {
                    "patch_id": patch.patch_id,
                    "project_id": patch.project_id,
                    "status": str(getattr(stored, "status", "UNKNOWN")),
                    "patch": asdict(patch),
                    "review_report": asdict(report) if report is not None else None,
                }
            )
        return {"approvals": approvals}

    @app.post("/api/scholar/approvals/{patch_id}/approve")
    def approve_scholar_approval(patch_id: str, request: PatchDecisionRequest) -> Response:
        return accept_patch(patch_id, request)

    @app.post("/api/scholar/approvals/{patch_id}/reject")
    def reject_scholar_approval(patch_id: str, request: PatchDecisionRequest) -> dict[str, Any]:
        return reject_patch(patch_id, request)

    @app.get("/api/scholar/demo")
    def scholar_demo() -> dict[str, Any]:
        return demo_console_payload()

    @app.post("/api/scholar/patches/{patch_id}/accept")
    def accept_patch(patch_id: str, request: PatchDecisionRequest) -> Response:
        try:
            stored = project_store.get_patch(patch_id)
            source_session_id = getattr(stored, "source_session_id", None)
            if request.session_id and source_session_id != request.session_id:
                raise ScholarRuntimeError(
                    "Patch 不属于当前 Session。", code="SESSION_CONFLICT"
                )
            result = patch_approval.approve(
                PatchApprovalRequest(
                    patch_id=patch_id,
                    project_id=request.project_id,
                    decision="ACCEPT",
                    expected_base_hash=request.expected_base_hash,
                    actor=request.actor,
                )
            )
        except Exception as error:
            raise _http_error(error) from error
        status_code = 409 if result.error_code else 200
        payload = asdict(result)
        reconciliation = _reconcile_approval_run(patch_id, result.status)
        if reconciliation is not None:
            payload["run_reconciliation"] = reconciliation
        return JSONResponse(payload, status_code=status_code)

    @app.post("/api/scholar/patches/{patch_id}/reject")
    def reject_patch(patch_id: str, request: PatchDecisionRequest) -> dict[str, Any]:
        try:
            stored = project_store.get_patch(patch_id)
            source_session_id = getattr(stored, "source_session_id", None)
            if request.session_id and source_session_id != request.session_id:
                raise ScholarRuntimeError(
                    "Patch 不属于当前 Session。", code="SESSION_CONFLICT"
                )
            result = patch_approval.reject(
                PatchApprovalRequest(
                    patch_id=patch_id,
                    project_id=request.project_id,
                    decision="REJECT",
                    expected_base_hash=request.expected_base_hash,
                    actor=request.actor,
                )
            )
            payload = asdict(result)
            reconciliation = _reconcile_approval_run(patch_id, result.status)
            if reconciliation is not None:
                payload["run_reconciliation"] = reconciliation
            return payload
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/projects/{project_id}/build")
    def request_build(project_id: str, patch_id: str | None = None) -> dict[str, Any]:
        try:
            return asdict(build_bridge.request_build(project_id, patch_id=patch_id))
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/scholar/projects/{project_id}/diagnostics")
    def project_diagnostics(project_id: str) -> dict[str, Any]:
        try:
            latest = build_bridge.latest(project_id)
            return {
                "project_id": project_id,
                "build": asdict(latest) if latest is not None else None,
                "diagnostics": [asdict(value) for value in build_bridge.diagnostics(project_id)],
            }
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/projects/{project_id}/build/report")
    def report_build(project_id: str, request: BuildReportRequest) -> dict[str, Any]:
        try:
            diagnostics = tuple(
                normalize_diagnostic(value)
                for value in request.diagnostics
            )
            result = build_bridge.report_build(
                project_id,
                build_id=request.build_id,
                status=request.status,
                diagnostics=diagnostics,
                completed_at=request.completed_at,
                message=request.message,
            )
            return asdict(result)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/system/status")
    def system_status() -> dict[str, Any]:
        return web_runtime.public_status()

    @app.get("/api/scholar/runtime/status")
    def scholar_runtime_status() -> dict[str, Any]:
        bundle = app.state.scholar_runtime
        if bundle is None:
            return {
                "status": "NOT_STARTED",
                "mode": None,
                "checkpoint": None,
                "persistent": False,
                "orchestration_backend": "crewai",
            }
        return {
            "status": "CLOSED" if bundle._closed else "READY",
            "mode": bundle.mode,
            "checkpoint": type(bundle.checkpointer).__name__,
            "persistent": type(bundle.checkpointer).__name__ != "InMemorySaver",
            "project_id": bundle.project_store.project_id,
            "orchestration_backend": getattr(bundle.scholar_orchestration, "backend_name", None),
        }

    @app.get("/api/scholar/workers/status")
    def scholar_workers_status() -> dict[str, Any]:
        jobs = app.state.scholar_runs.repository.list()
        workers = list(app.state.scholar_runs.worker_registry.list())
        return {
            "workers": workers,
            "worker_process": "external",
            "queue_depth": sum(
                job.status in {"QUEUED", "RETRY_PENDING"}
                and job.job_type.startswith("scholar.")
                for job in jobs
            ),
            "active_jobs": [
                {
                    "job_id": job.job_id,
                    "run_id": job.payload.get("run_id"),
                    "worker_id": job.worker_id,
                    "status": job.status,
                }
                for job in jobs
                if job.status == "RUNNING" and job.job_type.startswith("scholar.")
            ],
        }

    @app.get("/api/papers")
    def papers() -> dict[str, Any]:
        try:
            return web_runtime.list_papers()
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/papers/upload", response_model=JobCreated, status_code=202)
    async def upload_paper(
        file: UploadFile = File(...),
        method: Literal["auto", "txt", "ocr"] = Form("auto"),
        backend: Literal[
            "pipeline",
            "vlm-engine",
            "hybrid-engine",
            "vlm-http-client",
            "hybrid-http-client",
        ] = Form("pipeline"),
        formula_enabled: bool = Form(True),
        table_enabled: bool = Form(True),
        force_mineru: bool = Form(False),
    ) -> JobCreated:
        filename = file.filename or "paper.pdf"
        if Path(filename).suffix.casefold() != ".pdf":
            raise HTTPException(status_code=400, detail="只允许上传 PDF。")
        try:
            options = ParseOptions(
                method=method,
                backend=backend,
                formula_enabled=formula_enabled,
                table_enabled=table_enabled,
                force_mineru=force_mineru,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        upload_root = root / "data" / "runtime" / "web_uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(prefix="upload_", dir=upload_root)
        )
        temporary_pdf = temporary_directory / sanitize_pdf_filename(filename)
        size = 0
        signature = b""
        try:
            with temporary_pdf.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    if not signature:
                        signature = chunk[:5]
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF 不能超过 200 MB。",
                        )
                    target.write(chunk)
            if signature != b"%PDF-":
                raise HTTPException(status_code=400, detail="文件不是有效 PDF。")
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        finally:
            await file.close()

        def task(emit: EmitProgress) -> dict[str, Any]:
            try:
                return web_runtime.parse_pdf(temporary_pdf, options, emit)
            finally:
                shutil.rmtree(temporary_directory, ignore_errors=True)

        return job_manager.submit("parse", task)

    @app.post("/api/answers", response_model=JobCreated, status_code=202)
    def answer(request: AnswerRequest) -> JobCreated:
        return job_manager.submit(
            "answer",
            lambda emit: web_runtime.answer(request, emit),
            payload={
                "session_id": request.session_id,
                "project_id": request.project_id,
            },
        )

    @app.post("/api/answers/resume", response_model=JobCreated, status_code=202)
    def resume_answer(request: ResumeRequest) -> JobCreated:
        resume = getattr(web_runtime, "resume", None)
        if not callable(resume):
            raise HTTPException(status_code=501, detail="当前 Runtime 不支持 LangGraph Resume。")
        return job_manager.submit(
            "answer",
            lambda emit: resume(request.thread_id, request.user_input, emit),
        )

    @app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
    def job(job_id: str) -> JobSnapshot:
        try:
            return job_manager.snapshot(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error).strip("'")) from error

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobSnapshot)
    def cancel_job(job_id: str) -> JobSnapshot:
        try:
            return job_manager.cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error).strip("'")) from error

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        try:
            job_manager.snapshot(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error).strip("'")) from error

        async def stream() -> AsyncIterator[str]:
            last_sequence = 0
            while True:
                snapshot = job_manager.snapshot(job_id)
                for event in snapshot.events:
                    if event.sequence <= last_sequence:
                        continue
                    payload = json.dumps(
                        event.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: progress\n"
                        f"data: {payload}\n\n"
                    )
                    last_sequence = event.sequence
                if snapshot.status in {"succeeded", "failed", "cancelled"}:
                    done = json.dumps(
                        {"status": snapshot.status},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    yield f"event: done\ndata: {done}\n\n"
                    break
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/sessions")
    def sessions() -> dict[str, Any]:
        return web_runtime.list_sessions()

    @app.get("/api/sessions/{session_id}")
    def session_details(session_id: str) -> dict[str, Any]:
        try:
            return web_runtime.session_details(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/sessions/{session_id}/evidence")
    def session_evidence(session_id: str) -> dict[str, Any]:
        try:
            return web_runtime.session_evidence(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/sessions/{session_id}/transcript")
    def session_transcript(session_id: str) -> dict[str, Any]:
        try:
            return web_runtime.session_transcript(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/sessions/{session_id}/compact")
    def compact_session(session_id: str) -> dict[str, Any]:
        try:
            return web_runtime.compact_session(session_id)
        except Exception as error:
            raise _http_error(error) from error

    frontend = root / "web" / "dist"
    assets = frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False, response_model=None)
    def index() -> Response:
        index_file = frontend / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse(
            {
                "service": "LEO Research Agent API",
                "docs": "/docs",
                "frontend": "请先在 web/ 运行 npm run build。",
            }
        )

    @app.get("/{route:path}", include_in_schema=False)
    def spa_fallback(route: str) -> FileResponse:
        if route.startswith("api/"):
            raise HTTPException(status_code=404, detail="API 不存在。")
        index_file = frontend / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="前端尚未构建。")
        return FileResponse(index_file)

    return app
