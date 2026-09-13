from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.scholar.console import ScholarConsoleProjection
from app.scholar.models import DraftPatch
from app.scholar.project import ScholarProjectStore
from app.session import SessionManager


def _stored_run(tmp_path: Path, *, status: str = "COMPLETED") -> tuple[ScholarProjectStore, str]:
    store = ScholarProjectStore(tmp_path)
    manager = SessionManager(tmp_path)
    session = manager.resolve("CONSOLE_SESSION", title="Console fixture", project_id=store.project_id)
    runtime = manager.open(session.session_id)
    runtime.create_run(
        "write introduction",
        run_id="CONSOLE_RUN",
        thread_id="CONSOLE_THREAD",
        trace_id="CONSOLE_TRACE",
        project_id=store.project_id,
    )
    runtime.start_run("CONSOLE_RUN", worker_id="console-test")
    runtime.complete_run(
        "CONSOLE_RUN",
        status=status,  # type: ignore[arg-type]
        answer=json.dumps({"patch": {"patch_id": "PATCH_CONSOLE", "target_section": "introduction"}}),
        citations=[],
        evidence=[],
        metadata={
            "task_type": "WRITE_INTRODUCTION",
            "selected_skill": "write-introduction",
            "result_type": "WritingResult",
            "termination_reason": "NEEDS_USER_REVIEW",
            "harness": {
                "usage": {"steps": 4, "context_tokens": 180},
                "trace": [
                    {"kind": "step", "name": "HARNESS_PLAN", "status": "succeeded", "elapsed_ms": 1.0, "details": {}},
                    {"kind": "tool", "name": "research_evidence", "status": "succeeded", "elapsed_ms": 2.0, "details": {"evidence_count": 2}},
                    {"kind": "step", "name": "EVIDENCE_VALIDATE", "status": "succeeded", "elapsed_ms": 1.0, "details": {"verified_count": 2}},
                    {"kind": "tool", "name": "review_draft", "status": "succeeded", "elapsed_ms": 3.0, "details": {}},
                ],
                "termination_reason": "NEEDS_USER_REVIEW",
            },
        },
    )
    return store, store.project_id


def test_console_maps_persisted_harness_trace_without_second_workflow_state(tmp_path: Path) -> None:
    store, project_id = _stored_run(tmp_path)
    projection = ScholarConsoleProjection(tmp_path, project_store=store, session_manager=SessionManager(tmp_path))

    snapshot = projection.run_snapshot("CONSOLE_RUN")
    events = projection.run_events("CONSOLE_RUN")

    assert snapshot["routing"]["selected_skill"] == "write-introduction"
    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "SKILL_SELECTED",
        "RESEARCH_COMPLETED",
        "EVIDENCE_VERIFIED",
        "REVIEW_COMPLETED",
        "WAITING_USER",
        "RUN_COMPLETED",
    ]
    assert events[-1]["metadata"]["termination_reason"] == "NEEDS_USER_REVIEW"
    assert projection.evidence_view(project_id, run_id="CONSOLE_RUN")["run_id"] == "CONSOLE_RUN"


def test_console_project_state_reads_current_sections_and_patch_projection(tmp_path: Path) -> None:
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text("\\documentclass{article}\n\\input{sections/introduction}\n", encoding="utf-8")
    (tmp_path / "sections" / "introduction.tex").write_text("Current text.", encoding="utf-8")
    store = ScholarProjectStore(tmp_path)
    state = ScholarConsoleProjection(tmp_path, project_store=store).project_state(store.project_id)

    assert state["root_tex"] == "main.tex"
    assert state["sections"][0]["name"] == "introduction"
    assert state["sections"][0]["stale"] is False


def test_scholar_console_sse_supports_cursor_replay(tmp_path: Path) -> None:
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text("\\documentclass{article}\n\\input{sections/introduction}\n", encoding="utf-8")
    (tmp_path / "sections" / "introduction.tex").write_text("Current text.", encoding="utf-8")
    _, project_id = _stored_run(tmp_path)

    class Runtime:
        project_root = tmp_path

        def public_status(self):
            return {"status": "ok"}

    from app.web.api import create_app

    app = create_app(tmp_path, runtime=Runtime(), scholar_harness=SimpleNamespace())
    with TestClient(app) as client:
        response = client.get("/api/scholar/runs/CONSOLE_RUN/events?after=2")
        assert response.status_code == 200
        assert "event: scholar_run" in response.text
        assert "CONSOLE_RUN:1" not in response.text
        assert "CONSOLE_RUN:2" in response.text
        assert client.get(f"/api/scholar/projects/{project_id}/state").status_code == 200


def test_console_hides_unverified_candidate_from_evidence_projection(tmp_path: Path) -> None:
    store, project_id = _stored_run(tmp_path)
    projection = ScholarConsoleProjection(
        tmp_path,
        project_store=store,
        session_manager=SessionManager(tmp_path),
    )
    snapshot = projection.run_snapshot("CONSOLE_RUN")
    snapshot["result"]["value"] = {
        "supporting_evidence": [
            {
                "evidence_id": "VERIFIED",
                "source_type": "WEB_LITERATURE",
                "canonical_id": "doi:10.1234/verified",
                "source_locator": "abstract",
                "content": "verified span",
                "content_hash": "hash",
                "validation_status": "provider_payload_verified",
            }
        ],
        "citation_requirements": [
            {
                "evidence_id": "CANDIDATE",
                "candidate": {"evidence_id": "CANDIDATE", "title": "unverified"},
            }
        ],
    }
    original = projection.run_snapshot
    projection.run_snapshot = lambda _: snapshot  # type: ignore[method-assign]
    try:
        view = projection.evidence_view(project_id, run_id="CONSOLE_RUN")
    finally:
        projection.run_snapshot = original  # type: ignore[method-assign]
    assert [value["evidence_id"] for value in view["verified_evidence"]] == ["VERIFIED"]
    assert view["citation_requirements"][0]["evidence_id"] == "CANDIDATE"


def test_console_reflects_human_patch_status_after_approval(tmp_path: Path) -> None:
    store, project_id = _stored_run(tmp_path)
    store.save_patch(
        DraftPatch(
            patch_id="PATCH_CONSOLE",
            target_section="introduction",
            base_hash="base",
            proposed_content="new",
            project_id=project_id,
        ),
        status="APPLIED",
    )
    events = ScholarConsoleProjection(tmp_path, project_store=store, session_manager=SessionManager(tmp_path)).run_events("CONSOLE_RUN")
    approval = next(event for event in events if event["node"] == "Human Approval")
    assert approval["status"] == "COMPLETED"
    assert approval["metadata"]["patch_status"] == "APPLIED"
