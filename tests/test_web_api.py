from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.web.api import create_app
from app.web.jobs import JobManager
from app.web.models import AnswerRequest, ParseOptions
from app.web.runtime import EmitProgress, WebRuntimeConfig
from app.web.runtime import LocalRAGWebRuntime


class FakeWebRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.parsed_path: Path | None = None

    def answer(
        self,
        request: AnswerRequest,
        emit: EmitProgress,
    ) -> dict[str, Any]:
        emit("retrieving", "正在检索", 0.4)
        return {
            "query": request.query,
            "answerable": True,
            "answer": "多普勒观测可约束钟漂。[S1]",
            "claims": [],
            "citations": [],
            "refusal_reason": None,
            "validation": {"valid": True},
            "diagnostics": {"retrieval_mode": "agentic"},
            "session": {
                "session_id": request.session_id or "web-demo",
                "topic_id": "T001",
                "relation": "same_topic",
                "standalone_query": request.query,
            },
            "coverage": {"overall_sufficient": True, "coverage": []},
            "retrieval_rounds": [],
        }

    def parse_pdf(
        self,
        pdf_path: Path,
        options: ParseOptions,
        emit: EmitProgress,
    ) -> dict[str, Any]:
        self.parsed_path = pdf_path
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        assert options.method == "auto"
        emit("normalizing", "正在标准化", 0.8)
        return {"paper": {"paper_id": "P_web", "title": "LEO Web Paper"}}

    def list_papers(self) -> dict[str, Any]:
        return {
            "records": [
                {
                    "paper_id": "P_web",
                    "document_id": "D_web",
                    "work_id": "W_web",
                    "title": "LEO Web Paper",
                    "authors": ["Ada"],
                    "year": 2026,
                    "page_count": 10,
                    "quality_issue_count": 0,
                }
            ],
            "issues": [],
            "status": {"catalog_consistent": True},
        }

    def list_sessions(self) -> dict[str, Any]:
        return {"sessions": [{"session_id": "web-demo", "title": "LEO"}]}

    def session_details(self, session_id: str) -> dict[str, Any]:
        if session_id == "missing":
            raise KeyError(session_id)
        return {"session": {"session_id": session_id}, "topics": []}

    def session_evidence(self, session_id: str) -> dict[str, Any]:
        return {"session_id": session_id, "evidence": []}

    def session_transcript(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": [{"role": "user", "text": "fixture question"}],
        }

    def compact_session(self, session_id: str) -> dict[str, Any]:
        return {"session_id": session_id, "after_tokens": 100}

    def public_status(self) -> dict[str, Any]:
        return {
            "service": "test-web",
            "llm_configured": True,
            "embedding_model": "fixture/embedding",
        }


def wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("后台任务未在测试时限内完成。")


def test_web_api_lists_library_and_runs_answer_job(tmp_path: Path) -> None:
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/system/status").json()["llm_configured"] is True
        papers = client.get("/api/papers").json()
        assert papers["records"][0]["title"] == "LEO Web Paper"

        created = client.post(
            "/api/answers",
            json={
                "query": "为什么多普勒能估计钟漂？",
                "session_id": "web-demo",
                "include_context": True,
            },
        )
        assert created.status_code == 202
        completed = wait_for_job(client, created.json()["job_id"])
        assert completed["status"] == "succeeded"
        assert completed["result"]["answerable"] is True
        assert [event["stage"] for event in completed["events"]] == [
            "queued",
            "running",
            "retrieving",
            "completed",
        ]

        stream = client.get(
            f"/api/jobs/{created.json()['job_id']}/events"
        )
        assert stream.status_code == 200
        assert "event: progress" in stream.text
        assert "event: done" in stream.text


def test_web_api_uploads_pdf_and_removes_temporary_copy(tmp_path: Path) -> None:
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        created = client.post(
            "/api/papers/upload",
            files={"file": ("leo paper.pdf", b"%PDF-1.7 fixture", "application/pdf")},
        )
        assert created.status_code == 202
        completed = wait_for_job(client, created.json()["job_id"])
        assert completed["status"] == "succeeded"
        assert completed["result"]["paper"]["title"] == "LEO Web Paper"
        assert runtime.parsed_path is not None
        assert runtime.parsed_path.exists() is False

        invalid = client.post(
            "/api/papers/upload",
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )
        assert invalid.status_code == 400


def test_web_api_sessions_and_spa_fallback(tmp_path: Path) -> None:
    frontend = tmp_path / "web" / "dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("<main>LEO UI</main>", encoding="utf-8")
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        assert client.get("/api/sessions").json()["sessions"][0][
            "session_id"
        ] == "web-demo"
        assert client.get("/api/sessions/web-demo").status_code == 200
        assert client.get("/api/sessions/missing").status_code == 404
        assert client.get("/api/sessions/web-demo/evidence").json()[
            "evidence"
        ] == []
        assert client.get("/api/sessions/web-demo/transcript").json()[
            "messages"
        ][0]["role"] == "user"
        assert client.post("/api/sessions/web-demo/compact").status_code == 200
        assert "LEO UI" in client.get("/").text
        assert "LEO UI" in client.get("/research/session").text


def test_web_runtime_config_resolves_relative_model_cache(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("LEO_WEB_MODEL_CACHE", "private/models")

    config = WebRuntimeConfig.from_environment(tmp_path)

    assert config.model_cache == tmp_path / "private" / "models"


def test_web_runtime_config_inherits_existing_dense_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest = tmp_path / "data" / "index" / "dense_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        (
            '{"model_name":"fixture/bge-m3",'
            '"model_revision":"fixed-revision"}'
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "LEO_WEB_RERANKER_REVISION=reranker-revision\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LEO_WEB_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LEO_WEB_EMBEDDING_REVISION", raising=False)

    config = WebRuntimeConfig.from_environment(tmp_path)

    assert config.embedding_model == "fixture/bge-m3"
    assert config.embedding_revision == "fixed-revision"
    assert config.reranker_revision == "reranker-revision"


def test_explicit_web_embedding_revision_overrides_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest = tmp_path / "data" / "index" / "dense_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"model_name":"fixture/bge-m3","model_revision":"old"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LEO_WEB_EMBEDDING_REVISION", "explicit")

    config = WebRuntimeConfig.from_environment(tmp_path)

    assert config.embedding_revision == "explicit"


def test_web_job_error_redacts_api_key() -> None:
    manager = JobManager(max_workers=1)
    secret = "sk-web-secret-value"

    def fail(_: EmitProgress) -> dict[str, Any]:
        raise RuntimeError(f"upstream api_key={secret}")

    try:
        created = manager.submit("answer", fail)
        for _ in range(100):
            snapshot = manager.snapshot(created.job_id)
            if snapshot.status == "failed":
                break
            time.sleep(0.01)
        serialized = snapshot.model_dump_json()
        assert snapshot.status == "failed"
        assert secret not in serialized
        assert "[REDACTED]" in serialized
    finally:
        manager.close()


def test_local_web_parse_builds_searchable_indexes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from app.parsing.pipeline import PaperParseResult
    from app.web import runtime as runtime_module

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 fixture")
    result = PaperParseResult(
        paper_id="P_web",
        sha256="abc",
        raw_pdf=pdf,
        paper_json=tmp_path / "paper.json",
        mineru_output_directory=tmp_path / "mineru",
        mineru_reused=False,
        page_count=2,
        block_count=4,
        formula_count=0,
        table_count=0,
        figure_count=0,
        precheck={},
    )

    def fake_parse(**kwargs: Any) -> PaperParseResult:
        kwargs["progress_callback"]("writing")
        return result

    monkeypatch.setattr(runtime_module, "parse_paper", fake_parse)
    monkeypatch.setattr(
        runtime_module,
        "rebuild_catalog",
        lambda _: SimpleNamespace(summary=lambda: {"record_count": 1}),
    )
    monkeypatch.setattr(
        "app.chunking.builder.build_knowledge_base",
        lambda _: SimpleNamespace(
            issues=[],
            to_dict=lambda: {"total_chunk_count": 3},
        ),
    )
    monkeypatch.setattr(
        "app.indexing.dense.build_dense_index",
        lambda project_root, provider: SimpleNamespace(
            to_dict=lambda: {"status": "built", "chunk_count": 3}
        ),
    )
    runtime = LocalRAGWebRuntime(
        tmp_path,
        WebRuntimeConfig(model_cache=tmp_path / "models"),
    )
    monkeypatch.setattr(
        runtime,
        "_retrieval_runtime",
        lambda: SimpleNamespace(embedding_provider=object()),
    )
    stages: list[str] = []

    payload = runtime.parse_pdf(
        pdf,
        ParseOptions(),
        lambda stage, message, progress, details=None: stages.append(stage),
    )

    assert payload["knowledge"]["total_chunk_count"] == 3
    assert payload["dense"]["status"] == "built"
    assert stages == ["writing", "building_knowledge", "building_dense"]


def test_web_public_status_never_exposes_api_key(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    secret = "sk-web-status-secret"
    monkeypatch.setenv("LEO_LLM_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("LEO_LLM_MODEL", "fixture-model")
    monkeypatch.setenv("LEO_LLM_API_KEY", secret)
    runtime = LocalRAGWebRuntime(
        tmp_path,
        WebRuntimeConfig(model_cache=tmp_path / "models"),
    )

    serialized = str(runtime.public_status())

    assert secret not in serialized
    assert "api_key" not in serialized.casefold()
