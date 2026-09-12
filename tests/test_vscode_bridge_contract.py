from __future__ import annotations

import json
from pathlib import Path


def test_vscode_bridge_exposes_only_human_control_commands() -> None:
    root = Path(__file__).resolve().parents[1] / "vscode-extension"
    manifest = json.loads((root / "package.json").read_text(encoding="utf-8"))
    commands = {value["command"] for value in manifest["contributes"]["commands"]}
    assert commands == {
        "scholar.openPatchPreview",
        "scholar.acceptPatch",
        "scholar.rejectPatch",
        "scholar.buildManuscript",
        "scholar.showDiagnostics",
    }

    source = (root / "src" / "extension.ts").read_text(encoding="utf-8")
    assert '"vscode.diff"' in source
    assert '"latex-workshop.build"' in source
    assert "target_file" not in source
    assert "skip_approval" not in source
    assert "body: JSON.stringify({ proposed_content" not in source
