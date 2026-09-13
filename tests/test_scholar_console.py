from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.scholar.console import RunEvent, ScholarConsoleProjection
from app.scholar.models import DraftPatch
from app.scholar.project import ScholarProjectStore
from app.session import SessionManager


def _stored_run(
    tmp_path: Path,
    *,
    status: str = "COMPLETED",
    task_type: str = "WRITE_INTRODUCTION",
) -> tuple[ScholarProjectStore, str]:
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
    trace = [
        {"kind": "step", "name": "HARNESS_PLAN", "status": "succeeded", "elapsed_ms": 1.0, "details": {}},
    ]
    if task_type not in {"WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
        trace.extend([
            {"kind": "tool", "name": "research_evidence", "status": "succeeded", "elapsed_ms": 2.0, "details": {"evidence_count": 2}},
            {"kind": "step", "name": "EVIDENCE_VALIDATE", "status": "succeeded", "elapsed_ms": 1.0, "details": {"verified_count": 2}},
        ])
    if task_type != "SUPPORT_CLAIM":
        trace.append({"kind": "tool", "name": "review_draft", "status": "succeeded", "elapsed_ms": 3.0, "details": {}})
    answer = (
        {"support_status": "SUPPORTED"}
        if task_type == "SUPPORT_CLAIM"
        else {"patch": {"patch_id": "PATCH_CONSOLE", "target_section": "introduction"}}
    )
    runtime.complete_run(
        "CONSOLE_RUN",
        status=status,  # type: ignore[arg-type]
        answer=json.dumps(answer),
        citations=[],
        evidence=[],
        metadata={
            "task_type": task_type,
            "selected_skill": task_type.casefold().replace("_", "-"),
            "result_type": "ClaimSupportResult" if task_type == "SUPPORT_CLAIM" else "WritingResult",
            "termination_reason": "NEEDS_USER_REVIEW",
            "harness": {
                "usage": {"steps": 4, "context_tokens": 180},
                "trace": trace,
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
    assert [event["cursor"] for event in events] == list(range(1, len(events) + 1))
    assert [event["event_id"] for event in events] == [f"CONSOLE_RUN:{index}" for index in range(1, len(events) + 1)]
    assert all(isinstance(event, dict) for event in events)
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
    assert state["sections"][0]["dependencies"] == []
    assert state["sections"][0]["review_status"] == "NONE"


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
        assert '"cursor":2' not in response.text
        assert '"cursor":3' in response.text
        assert "event: end" in response.text
        assert client.get("/api/scholar/runs/CONSOLE_RUN/events?after=-1").status_code == 422
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
        "claims": [{"claim_id": "C1", "text": "Verified claim", "status": "supported", "evidence_ids": ["VERIFIED"]}],
        "supporting_evidence": [
            {
                "evidence_id": "VERIFIED",
                "source_type": "WEB_LITERATURE",
                "canonical_id": "doi:10.1234/verified",
                "source_locator": "abstract",
                "content": "verified span",
                "content_hash": hashlib.sha256(b"verified span").hexdigest(),
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
    assert view["claims"][0]["claim_id"] == "C1"
    assert view["verified_evidence"][0]["claim_ids"] == ["C1"]
    assert view["citation_requirements"][0]["evidence_id"] == "CANDIDATE"


def test_console_support_claim_and_synthesis_nodes_follow_capability_boundary(tmp_path: Path) -> None:
    store, project_id = _stored_run(tmp_path, task_type="SUPPORT_CLAIM")
    projection = ScholarConsoleProjection(tmp_path, project_store=store, session_manager=SessionManager(tmp_path))
    support_nodes = projection.run_events("CONSOLE_RUN")
    assert not any(event["node"] in {"DraftPatch", "Human Approval"} for event in support_nodes)
    assert not any(event["type"] == "PATCH_CREATED" for event in support_nodes)

    for task_type in ("WRITE_CONCLUSION", "WRITE_ABSTRACT"):
        _stored_run(tmp_path / task_type.lower(), task_type=task_type)
        synthesis_root = tmp_path / task_type.lower()
        synthesis_store = ScholarProjectStore(synthesis_root)
        synthesis_events = ScholarConsoleProjection(
            synthesis_root,
            project_store=synthesis_store,
            session_manager=SessionManager(synthesis_root),
        ).run_events("CONSOLE_RUN")
        assert not any(event["node"] == "Research Subagent" for event in synthesis_events)


def test_console_external_audit_projection_and_claim_mapping_are_complete(tmp_path: Path) -> None:
    store, project_id = _stored_run(tmp_path)
    content = "A verified external span."
    store.save_external_evidence_projection(
        project_id=project_id,
        evidence_id="EXTERNAL_01",
        identity_key="doi:10.1234/external",
        canonical_id="doi:10.1234/external",
        source_locator="https://doi.org/10.1234/external#abstract",
        provider="Crossref",
        publication_date="2025-01-01",
        retrieved_at="2026-09-13T00:00:00+00:00",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        validation_status="provider_payload_verified",
        metadata={
            "source_type": "WEB_LITERATURE",
            "title": "External paper",
            "authors": ["Author"],
            "content": content,
        },
    )
    store.save_external_evidence_projection(
        project_id=project_id,
        evidence_id="EXTERNAL_PENDING",
        identity_key="doi:10.1234/pending",
        canonical_id="doi:10.1234/pending",
        source_locator="https://doi.org/10.1234/pending#abstract",
        provider="Crossref",
        publication_date="2025-01-01",
        retrieved_at="2026-09-13T00:00:00+00:00",
        content_hash="not-a-content-hash",
        validation_status="pending",
        metadata={"source_type": "WEB_LITERATURE", "content": "not verified"},
    )
    projection = ScholarConsoleProjection(tmp_path, project_store=store, session_manager=SessionManager(tmp_path))
    view = projection.evidence_view(project_id)
    assert view["verified_evidence"][0]["source_type"] == "WEB_LITERATURE"
    assert view["verified_evidence"][0]["evidence_span"] == content
    assert view["verified_evidence"][0]["provider"] == "Crossref"
    assert [item["evidence_id"] for item in view["verified_evidence"]] == ["EXTERNAL_01"]


def test_run_event_contract_rejects_invalid_cursor_and_has_stable_shape() -> None:
    event = RunEvent(
        event_id="RUN:1",
        cursor=1,
        run_id="RUN",
        session_id=None,
        timestamp="2026-09-13T00:00:00+00:00",
        type="RUN_STARTED",
        node="Supervisor",
        status="COMPLETED",
        summary="started",
    )
    assert event.to_dict()["cursor"] == 1
    try:
        RunEvent(
            event_id="RUN:0",
            cursor=0,
            run_id="RUN",
            session_id=None,
            timestamp="2026-09-13T00:00:00+00:00",
            type="RUN_STARTED",
            node="Supervisor",
            status="COMPLETED",
            summary="started",
        )
    except ValueError as error:
        assert "cursor" in str(error)
    else:
        raise AssertionError("invalid RunEvent cursor was accepted")


def test_console_failed_and_interrupted_runs_have_terminal_event_without_completion(tmp_path: Path) -> None:
    for status, event_type in (("FAILED", "RUN_FAILED"), ("INTERRUPTED", "RUN_INTERRUPTED")):
        root = tmp_path / status.lower()
        _stored_run(root, status=status)
        store = ScholarProjectStore(root)
        events = ScholarConsoleProjection(root, project_store=store, session_manager=SessionManager(root)).run_events("CONSOLE_RUN")
        assert events[-1]["type"] == event_type
        assert not any(event["type"] == "RUN_COMPLETED" for event in events)


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
