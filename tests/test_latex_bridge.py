from __future__ import annotations

from pathlib import Path

from app.scholar.approval import LatexBridgeService, LatexDiagnostic
from app.scholar.project import ScholarProjectStore


def test_build_request_is_triggered_until_vscode_reports_result(tmp_path: Path) -> None:
    store = ScholarProjectStore(tmp_path)
    bridge = LatexBridgeService(tmp_path, project_store=store)

    requested = bridge.request_build(store.project_id)
    assert requested.status == "BUILD_TRIGGERED"
    assert requested.status != "SUCCESS"

    failed = bridge.report_build(
        store.project_id,
        build_id=requested.build_id,
        status="FAILED",
        diagnostics=(
            LatexDiagnostic("sections/introduction.tex", 4, 2, "ERROR", "Undefined control sequence"),
        ),
        message="LaTeX Workshop reported failure",
    )
    assert failed.status == "FAILED"
    assert bridge.diagnostics(store.project_id)[0].severity == "ERROR"
    assert bridge.latest(store.project_id).build_id == requested.build_id  # type: ignore[union-attr]
