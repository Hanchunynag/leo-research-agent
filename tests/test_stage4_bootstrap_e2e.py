from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.academic_mcp.models import (
    DiscoveryResult,
    FullTextLocation,
    FullTextResult,
    PaperRecord,
)
from app.parsing.pipeline import PaperParseResult
from app.research import ResearchRuntime, WorkflowName, WorkflowRequest, build_default_gateway
from app.research.providers import build_bootstrap_provider_composition


class FakeAcademicBackend:
    def __init__(
        self,
        tmp_path: Path,
        *,
        empty: bool = False,
        search_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.empty = empty
        self.search_error = search_error
        self.download_error = download_error

    async def search(self, query: str, limit: int = 20) -> DiscoveryResult:
        if self.search_error:
            raise self.search_error
        papers = [] if self.empty else [
            PaperRecord(
                title="LEO navigation review",
                authors=["Researcher"],
                abstract="Review of LEO navigation evidence.",
                doi="10.1000/leo-review",
                open_access=True,
                sources=["fixture-crossref"],
            )
        ]
        return DiscoveryResult(papers=papers[:limit], failures=[])

    async def find_fulltext(
        self,
        doi: str | None = None,
        openalex_id: str | None = None,
        arxiv_id: str | None = None,
    ) -> FullTextResult:
        return FullTextResult(
            locations=[
                FullTextLocation(
                    url="https://example.org/paper.pdf",
                    source="fixture",
                    download_token="download-1",
                )
            ],
            failures=[],
        )

    async def download_open_pdf(
        self, download_token: str, filename: str | None = None
    ) -> dict[str, str | int | bool]:
        if self.download_error:
            raise self.download_error
        path = self.tmp_path / "data" / "inbox" / (filename or "paper.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.7 fixture")
        return {
            "path": path.as_posix(),
            "sha256": "sha-download",
            "byte_count": path.stat().st_size,
            "source_url": "https://example.org/paper.pdf",
            "reused": False,
        }


class Generator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, context: Any) -> Mapping[str, Any]:
        self.calls += 1
        evidence = context.selected_evidence[0]
        return {
            "answerable": True,
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Canonical evidence supports the answer.",
                    "evidence_ids": [evidence["evidence_id"]],
                }
            ],
            "usage": {"total_tokens": 20},
        }


def _fake_parser(tmp_path: Path, *, fail: bool = False):  # type: ignore[no-untyped-def]
    def parse(source: Path, config: Any) -> PaperParseResult:
        if fail:
            raise RuntimeError("parse failed")
        paper_json = tmp_path / "data" / "canonical" / "P_new" / "paper.json"
        paper_json.parent.mkdir(parents=True, exist_ok=True)
        paper_json.write_text(
            json.dumps(
                {
                    "paper_id": "P_new",
                    "identity": {"document_id": "D_new", "work_id": "W_new"},
                    "source": {"sha256": "sha-download"},
                    "blocks": [],
                }
            ),
            encoding="utf-8",
        )
        return PaperParseResult(
            paper_id="P_new",
            sha256="sha-download",
            raw_pdf=source,
            paper_json=paper_json,
            mineru_output_directory=tmp_path / "data" / "parsed" / "P_new",
            mineru_reused=False,
            page_count=1,
            block_count=1,
            formula_count=0,
            table_count=0,
            figure_count=0,
            precheck={},
        )

    return parse


def _runtime(
    tmp_path: Path,
    backend: FakeAcademicBackend,
    *,
    parse_fail: bool = False,
    scope_conflict: bool = False,
):  # type: ignore[no-untyped-def]
    composition = build_bootstrap_provider_composition(
        tmp_path,
        backend,
        parser=_fake_parser(tmp_path, fail=parse_fail),
        worker_id="test-bootstrap",
    )
    scope_version = 1

    def read_scope(arguments, context):  # type: ignore[no-untyped-def]
        return {
            "workspace_id": "default",
            "scope_version": arguments["scope_version"],
            "constraints": ["workspace-only"],
        }

    def update_scope(arguments, context):  # type: ignore[no-untyped-def]
        nonlocal scope_version
        if scope_conflict:
            raise ValueError("scope version conflict")
        if arguments["scope_version"] != scope_version:
            raise ValueError("scope version conflict")
        scope_version += 1
        return {"workspace_id": "default", "scope_version": scope_version}

    def retrieve(arguments, context):  # type: ignore[no-untyped-def]
        if int(arguments["scope_version"]) < 2:
            return {"results": []}
        return {
            "results": [
                {
                    "evidence_id": "E_new",
                    "evidence_state": "selected",
                    "document_id": "D_new",
                    "chunk_id": "D_new_c1",
                    "page_start": 1,
                    "page_end": 1,
                    "block_ids": ["B_1"],
                    "content": "Canonical evidence supports the answer.",
                    "directness": "direct",
                    "evidence_grade": "primary",
                }
            ]
        }

    handlers = {
        **composition.gateway_handlers,
        "workspace.read_scope": read_scope,
        "workspace.update_scope": update_scope,
        "knowledge.retrieve": retrieve,
    }
    generator = Generator()
    runtime = ResearchRuntime(build_default_gateway(handlers), generator)
    return runtime, composition, generator


def _request(*, confirmed: bool) -> WorkflowRequest:
    return WorkflowRequest(
        query="How can LEO signals support navigation?",
        workspace_id="default",
        scope_version=1,
        workflow=WorkflowName.RESEARCH_BOOTSTRAP,
        bootstrap_confirmed=confirmed,
        workspace_summary={"document_count": 0},
    )


def test_empty_workspace_returns_scope_proposal_without_confirmation(
    tmp_path: Path,
) -> None:
    runtime, composition, generator = _runtime(
        tmp_path, FakeAcademicBackend(tmp_path)
    )

    result = runtime.run(_request(confirmed=False))

    assert result["answerable"] is False
    assert result["workflow_details"]["awaiting_confirmation"] is True
    assert len(result["workflow_details"]["scope_proposal"]["candidate_ids"]) == 1
    assert composition.repository.list() == ()
    assert generator.calls == 0


def test_confirmed_bootstrap_downloads_parses_updates_scope_and_resumes(
    tmp_path: Path,
) -> None:
    runtime, composition, generator = _runtime(
        tmp_path, FakeAcademicBackend(tmp_path)
    )

    waiting_download = runtime.run(_request(confirmed=True))
    assert waiting_download["workflow_details"]["pending_stage"] == "literature.download"
    composition.worker.run_until_idle()
    waiting_parse = runtime.run(_request(confirmed=True))
    assert waiting_parse["workflow_details"]["pending_stage"] == "document.parse"
    composition.worker.run_until_idle()
    completed = runtime.run(_request(confirmed=True))

    assert completed["answerable"] is True
    assert completed["workflow_details"]["resumed_original_question"] is True
    assert completed["workflow_details"]["resumed_scope_version"] == 2
    assert completed["workflow_details"]["ingested_document_ids"] == ["D_new"]
    assert generator.calls == 1


def test_bootstrap_search_empty_or_unavailable_is_safe(tmp_path: Path) -> None:
    empty_runtime, _, empty_generator = _runtime(
        tmp_path / "empty", FakeAcademicBackend(tmp_path / "empty", empty=True)
    )
    empty = empty_runtime.run(_request(confirmed=True))
    failing_runtime, _, failing_generator = _runtime(
        tmp_path / "timeout",
        FakeAcademicBackend(tmp_path / "timeout", search_error=TimeoutError("timeout")),
    )
    timeout = failing_runtime.run(_request(confirmed=True))

    assert empty["answerable"] is False
    assert empty["workflow_details"]["resumed_original_question"] is True
    assert timeout["answerable"] is False
    assert "safe_termination" in timeout["diagnostics"]["termination_reason"]
    assert empty_generator.calls == failing_generator.calls == 0


def test_download_or_parse_failure_never_updates_workspace(tmp_path: Path) -> None:
    download_runtime, download_composition, _ = _runtime(
        tmp_path / "download",
        FakeAcademicBackend(
            tmp_path / "download", download_error=ConnectionError("download failed")
        ),
    )
    download_runtime.run(_request(confirmed=True))
    download_result = download_composition.worker.run_once()
    assert download_result is not None
    assert download_result.status == "RETRY_PENDING"

    parse_runtime, parse_composition, _ = _runtime(
        tmp_path / "parse", FakeAcademicBackend(tmp_path / "parse"), parse_fail=True
    )
    parse_runtime.run(_request(confirmed=True))
    parse_composition.worker.run_until_idle()
    parse_runtime.run(_request(confirmed=True))
    parse_result = parse_composition.worker.run_once()

    assert parse_result is not None
    assert parse_result.status == "RETRY_PENDING"


def test_scope_version_conflict_refuses_after_ingestion(tmp_path: Path) -> None:
    runtime, composition, generator = _runtime(
        tmp_path, FakeAcademicBackend(tmp_path), scope_conflict=True
    )
    runtime.run(_request(confirmed=True))
    composition.worker.run_until_idle()
    runtime.run(_request(confirmed=True))
    composition.worker.run_until_idle()

    result = runtime.run(_request(confirmed=True))

    assert result["answerable"] is False
    assert "safe_termination" in result["diagnostics"]["termination_reason"]
    assert generator.calls == 0
