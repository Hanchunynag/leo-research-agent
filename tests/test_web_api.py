from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.web.api import create_app
from app.web.jobs import JobManager
from app.web.models import ParseOptions
from app.web.runtime import EmitProgress, WebRuntimeConfig
from app.web.runtime import LocalRAGWebRuntime
from app.tenancy import TenantPrincipal


class FakeWebRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.parsed_path: Path | None = None

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


def test_web_api_lists_library_and_keeps_agent_execution_out_of_web_jobs(tmp_path: Path) -> None:
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/system/status").json()["llm_configured"] is True
        papers = client.get("/api/papers").json()
        assert papers["records"][0]["title"] == "LEO Web Paper"

        assert client.get("/api/answers").status_code == 404
        assert client.get("/api/sessions").status_code == 404
        assert client.get("/api/scholar/runs").json() == {"runs": []}


def test_ready_exposes_knowledge_readiness_without_breaking_liveness(
    tmp_path: Path,
) -> None:
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["application_runtime"] == "ready"
        assert payload["research_readiness"] == "degraded"
        assert payload["knowledge_index"]["status"] == "not_initialized"
        assert client.get("/health").status_code == 200


def test_web_is_control_plane_and_does_not_build_agent_harness(
    tmp_path: Path,
) -> None:
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        assert app.state.harness is None
        status = client.get("/api/scholar/runtime/status").json()
        assert status["execution_plane"] == "external_worker"
        assert status["orchestration_backend"] == "crewai"
        # Client-supplied identity headers are ignored unless an authenticated
        # gateway/resolver is explicitly configured.
        assert client.get(
            "/api/scholar/projects",
            headers={"X-Tenant-Id": "attacker", "X-Principal-Id": "attacker"},
        ).status_code == 200


def test_web_jobs_are_scoped_to_the_authenticated_identity(tmp_path: Path) -> None:
    runtime = FakeWebRuntime(tmp_path)

    def resolve_identity(request: Any) -> TenantPrincipal:
        return TenantPrincipal(
            tenant_id=request.headers.get("X-Tenant-Id", "local"),
            principal_id=request.headers.get("X-Principal-Id", "local"),
        )

    app = create_app(
        tmp_path,
        runtime=runtime,
        jobs=JobManager(max_workers=1),
        identity_resolver=resolve_identity,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/papers/upload",
            headers={"X-Tenant-Id": "tenant-a", "X-Principal-Id": "user-a"},
            files={"file": ("paper.pdf", b"%PDF-1.7 fixture", "application/pdf")},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        assert client.get(
            f"/api/jobs/{job_id}",
            headers={"X-Tenant-Id": "tenant-b", "X-Principal-Id": "user-b"},
        ).status_code == 403
        assert client.get(
            f"/api/jobs/{job_id}",
            headers={"X-Tenant-Id": "tenant-a", "X-Principal-Id": "user-a"},
        ).status_code == 200


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


def test_web_api_spa_fallback_and_run_surface(tmp_path: Path) -> None:
    frontend = tmp_path / "web" / "dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("<main>LEO UI</main>", encoding="utf-8")
    runtime = FakeWebRuntime(tmp_path)
    app = create_app(tmp_path, runtime=runtime, jobs=JobManager(max_workers=1))

    with TestClient(app) as client:
        assert client.get("/api/sessions").status_code == 404
        assert client.get("/api/scholar/runs").status_code == 200
        assert "LEO UI" in client.get("/").text
        assert "LEO UI" in client.get("/research/session").text


def test_web_api_initializes_empty_manuscript_without_writing_agent_content(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from app.scholar.project import ScholarProjectStore
    from app.web.api import create_app

    class Runtime:
        project_root = tmp_path

        def public_status(self):
            return {"status": "ok"}

    store = ScholarProjectStore(tmp_path)
    app = create_app(tmp_path, runtime=Runtime(), scholar_orchestration=SimpleNamespace())
    with TestClient(app) as client:
        response = client.post(
            f"/api/scholar/projects/{store.project_id}/manuscript/initialize"
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["manuscript_available"] is True
        assert payload["root_tex"] == "main.tex"
        assert (tmp_path / "main.tex").is_file()
        assert (tmp_path / "sections" / "introduction.tex").read_text(encoding="utf-8") == "% Introduction\n"
        assert payload["patches"] == []

        existing = tmp_path / "main.tex"
        authored = "\\documentclass{book}\n% keep this\n"
        existing.write_text(authored, encoding="utf-8")
        second = client.post(
            f"/api/scholar/projects/{store.project_id}/manuscript/initialize"
        )
        assert second.status_code == 200
        assert existing.read_text(encoding="utf-8") == authored


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
        created = manager.submit("parse", fail)
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


def test_local_web_duplicate_pdf_reuses_canonical_without_changing_result_shape(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from app.web import runtime as runtime_module
    from tests.test_corpus_workspace import write_fixture

    pdf = tmp_path / "duplicate.pdf"
    pdf.write_bytes(b"%PDF-1.7 duplicate fixture")
    write_fixture(tmp_path, "D_001")
    canonical_path = tmp_path / "data" / "canonical" / "P_001" / "paper.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["source"]["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    canonical["pipeline"] = {"mineru_output_directory": "data/parsed/P_001/mineru"}
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")

    def unexpected_parse(**kwargs: Any) -> Any:
        raise AssertionError("duplicate content must not be parsed again")

    monkeypatch.setattr(runtime_module, "parse_paper", unexpected_parse)
    monkeypatch.setattr(
        runtime_module,
        "rebuild_catalog",
        lambda _: SimpleNamespace(
            records=[object()], summary=lambda: {"record_count": 1}
        ),
    )
    monkeypatch.setattr(
        "app.chunking.builder.build_knowledge_base",
        lambda _: SimpleNamespace(
            issues=[], to_dict=lambda: {"total_chunk_count": 1}
        ),
    )
    monkeypatch.setattr(
        "app.indexing.dense.build_dense_index",
        lambda project_root, provider: SimpleNamespace(
            to_dict=lambda: {"status": "reused", "chunk_count": 1}
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

    assert set(payload) == {"paper", "catalog", "knowledge", "dense"}
    assert set(payload["paper"]) == {
        "paper_id",
        "document_id",
        "sha256",
        "paper_json",
        "mineru_directory",
    }
    assert payload["paper"]["document_id"] == "D_001"
    assert payload["dense"]["status"] == "reused"
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
