from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.scholar import DraftPatch
from app.scholar.approval import LatexBridgeService, PatchApprovalService
from app.scholar.project import ScholarProjectStore
from app.web.api import create_app
from tests.test_scholar_foundation import make_project
from tests.test_web_api import FakeWebRuntime


def test_patch_api_preview_accept_and_build_are_separate(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    approval = PatchApprovalService(root, project_store=store)
    bridge = LatexBridgeService(root, project_store=store)
    state = approval.synchronizer.scan()
    original = approval.synchronizer.read_section(state, "introduction")
    patch = DraftPatch(
        patch_id="PATCH_API",
        target_section="introduction",
        base_hash=state.sections["introduction"].content_hash,
        original_content=original,
        proposed_content="API-applied introduction.\n",
        project_id=store.project_id,
    )
    approval.register_patch(patch)
    app = create_app(
        root,
        runtime=FakeWebRuntime(root),
        approval_service=approval,
        latex_bridge=bridge,
    )

    with TestClient(app) as client:
        preview = client.get("/api/scholar/patches/PATCH_API")
        assert preview.status_code == 200
        assert preview.json()["preview"]["status"] == "AWAITING_APPROVAL"

        accepted = client.post(
            "/api/scholar/patches/PATCH_API/accept",
            json={
                "project_id": store.project_id,
                "expected_base_hash": patch.base_hash,
                "actor": "web-human",
            },
        )
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "APPLIED"
        assert accepted.json()["apply_result"]["status"] == "APPLIED"

        build = client.post(f"/api/scholar/projects/{store.project_id}/build?patch_id=PATCH_API")
        assert build.status_code == 200
        assert build.json()["status"] == "BUILD_TRIGGERED"

        report = client.post(
            f"/api/scholar/projects/{store.project_id}/build/report",
            json={
                "build_id": build.json()["build_id"],
                "status": "FAILED",
                "diagnostics": [
                    {
                        "file": "sections/introduction.tex",
                        "line": 2,
                        "severity": "ERROR",
                        "message": "Undefined control sequence",
                    }
                ],
            },
        )
        assert report.status_code == 200
        assert report.json()["status"] == "FAILED"
        assert client.get("/api/scholar/patches/PATCH_API").json()["preview"]["status"] == "BUILD_FAILED"

        diagnostics = client.get(f"/api/scholar/projects/{store.project_id}/diagnostics")
        assert diagnostics.status_code == 200
        assert diagnostics.json()["build"]["status"] == "FAILED"


def test_patch_api_unknown_patch_is_not_an_apply_entrypoint(tmp_path: Path) -> None:
    app = create_app(tmp_path, runtime=FakeWebRuntime(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/scholar/patches/DOES_NOT_EXIST")
        assert response.status_code == 404


def test_scholar_request_api_forwards_only_to_injected_harness(tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Harness:
        def scholar_request(self, instruction: str, project_id: str, **kwargs: object) -> SimpleNamespace:
            calls.append((instruction, project_id, kwargs))
            return SimpleNamespace(to_dict=lambda: {"status": "READY", "result_type": "ClaimSupportResult"})

        def resume(self, thread_id: str, resume_value: object, project_id: str, **kwargs: object) -> SimpleNamespace:
            calls.append((thread_id, project_id, {"resume_value": resume_value, **kwargs}))
            return SimpleNamespace(to_dict=lambda: {"status": "COMPLETED", "resumed": True})

    app = create_app(tmp_path, runtime=FakeWebRuntime(tmp_path), scholar_harness=Harness())
    with TestClient(app) as client:
        response = client.post(
            "/api/scholar/requests",
            json={
                "instruction": "support this claim",
                "project_id": "PROJECT_1",
                "task_type": "SUPPORT_CLAIM",
                "session_id": "SESSION_1",
                "thread_id": "THREAD_1",
            },
        )
        resumed = client.post(
            "/api/scholar/requests/resume",
            json={
                "project_id": "PROJECT_1",
                "thread_id": "THREAD_1",
                "instruction": "support this claim",
                "resume_value": {"decisions": [{"type": "approve"}]},
                "task_type": "SUPPORT_CLAIM",
                "session_id": "SESSION_1",
            },
        )

    assert response.status_code == 200
    assert resumed.status_code == 200
    assert response.json()["result_type"] == "ClaimSupportResult"
    assert resumed.json()["resumed"] is True
    assert calls[0][2]["task_type"] == "SUPPORT_CLAIM"
    assert calls[1][2]["resume_value"] == {"decisions": [{"type": "approve"}]}
