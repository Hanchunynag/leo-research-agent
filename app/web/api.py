"""LEO Research Agent 的本地 FastAPI 入口。"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from dataclasses import asdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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
    ScholarTaskRequest,
)
from app.web.runtime import EmitProgress, LocalRAGWebRuntime
from app.scholar.console import ScholarConsoleProjection, demo_console_payload


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


def create_app(
    project_root: Path | None = None,
    *,
    runtime: WebRuntime | None = None,
    jobs: JobManager | None = None,
    approval_service: PatchApprovalService | None = None,
    latex_bridge: LatexBridgeService | None = None,
    scholar_harness: Any | None = None,
) -> FastAPI:
    """创建可测试的 API；默认使用仓库根目录和本地长驻 Runtime。"""

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if production_factory is not None:
                bundle = production_factory.build()
                app.state.scholar_runtime = bundle
                app.state.scholar_harness = bundle.harness
                app.state.scholar_console = ScholarConsoleProjection(
                    root,
                    project_store=bundle.project_store,
                    session_manager=bundle.session_manager,
                )
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
    app.state.scholar_runtime = None
    app.state.scholar_runtime_factory = production_factory
    app.state.scholar_console = ScholarConsoleProjection(root, project_store=project_store)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/scholar/patches/{patch_id}")
    def patch_preview(patch_id: str) -> dict[str, Any]:
        try:
            preview = patch_approval.get_preview(patch_id)
            return {"preview": asdict(preview)}
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/scholar/requests")
    def scholar_request(request: ScholarTaskRequest) -> dict[str, Any]:
        harness = app.state.scholar_harness
        if harness is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "SCHOLAR_HARNESS_NOT_CONFIGURED", "message": "Scholar Harness 未配置。"},
            )
        try:
            result = harness.scholar_request(
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
        harness = app.state.scholar_harness
        if harness is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "SCHOLAR_HARNESS_NOT_CONFIGURED", "message": "Scholar Harness 未配置。"},
            )
        try:
            result = harness.resume(
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

    @app.get("/api/scholar/runs/{run_id}")
    def scholar_run_snapshot(run_id: str) -> dict[str, Any]:
        try:
            return app.state.scholar_console.run_snapshot(run_id)
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
            events = app.state.scholar_console.run_events(run_id)
            for event in events:
                cursor = int(event["cursor"])
                if cursor <= after:
                    continue
                payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                yield f"id: {cursor}\nevent: scholar_run\ndata: {payload}\n\n"
            yield "event: end\ndata: {}\n\n"

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

    @app.get("/api/scholar/demo")
    def scholar_demo() -> dict[str, Any]:
        return demo_console_payload()

    @app.post("/api/scholar/patches/{patch_id}/accept")
    def accept_patch(patch_id: str, request: PatchDecisionRequest) -> Response:
        try:
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
        return JSONResponse(asdict(result), status_code=status_code)

    @app.post("/api/scholar/patches/{patch_id}/reject")
    def reject_patch(patch_id: str, request: PatchDecisionRequest) -> dict[str, Any]:
        try:
            result = patch_approval.reject(
                PatchApprovalRequest(
                    patch_id=patch_id,
                    project_id=request.project_id,
                    decision="REJECT",
                    expected_base_hash=request.expected_base_hash,
                    actor=request.actor,
                )
            )
            return asdict(result)
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
            }
        return {
            "status": "CLOSED" if bundle._closed else "READY",
            "mode": bundle.mode,
            "checkpoint": type(bundle.checkpointer).__name__,
            "persistent": type(bundle.checkpointer).__name__ != "InMemorySaver",
            "project_id": bundle.project_store.project_id,
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
