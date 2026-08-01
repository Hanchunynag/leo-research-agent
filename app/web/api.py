"""LEO Research Agent 的本地 FastAPI 入口。"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.ingestion.ingest import sanitize_pdf_filename
from app.web.jobs import JobManager
from app.web.models import AnswerRequest, JobCreated, JobSnapshot, ParseOptions
from app.web.runtime import EmitProgress, LocalRAGWebRuntime


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
) -> FastAPI:
    """创建可测试的 API；默认使用仓库根目录和本地长驻 Runtime。"""

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    web_runtime: WebRuntime = runtime or LocalRAGWebRuntime(root)
    job_manager = jobs or JobManager(max_workers=2)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        job_manager.close()

    app = FastAPI(
        title="LEO Research Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = web_runtime
    app.state.jobs = job_manager
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

    @app.get("/api/system/status")
    def system_status() -> dict[str, Any]:
        return web_runtime.public_status()

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
        )

    @app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
    def job(job_id: str) -> JobSnapshot:
        try:
            return job_manager.snapshot(job_id)
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
                if snapshot.status in {"succeeded", "failed"}:
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
