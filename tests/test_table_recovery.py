from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsing import table_recovery


def make_document(tmp_path: Path) -> dict[str, object]:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"table-image")
    return {
        "paper_id": "P_fixture",
        "blocks": [
            {
                "block_id": "P_fixture_p001_b001",
                "type": "table",
                "table_html": None,
                "table_html_raw": None,
                "image_path": str(image),
                "quality": {
                    "status": "image_only",
                    "issues": ["table_html_empty"],
                    "retrieval_enabled": True,
                },
            },
            {
                "block_id": "P_fixture_p001_b002",
                "type": "algorithm",
                "table_html": None,
                "image_path": str(image),
                "quality": {
                    "status": "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            },
        ],
    }


def fake_executable(tmp_path: Path) -> Path:
    executable = tmp_path / ".venv-paddleocr" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def fake_worker(tmp_path: Path) -> None:
    worker = tmp_path / "scripts" / "paddleocr_table_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# fixture worker\n", encoding="utf-8")


def test_resolve_preserves_paddle_venv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "system-python"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    executable = tmp_path / ".venv-paddleocr" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(target)

    resolved = table_recovery.resolve_paddleocr_executable(tmp_path)

    assert resolved == executable.absolute()


def test_recovery_only_targets_image_only_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = make_document(tmp_path)
    fake_worker(tmp_path)
    executable = fake_executable(tmp_path)

    def fake_run_worker(**kwargs: object) -> dict[str, object]:
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        payload = {
            "done": True,
            "paddleocr_version": "fixture",
            "pred_html": "<table><tr><td>42</td></tr></table>",
            "cell_box_list": [[0, 0, 10, 10]],
            "table_ocr_pred": [["42"]],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(table_recovery, "_run_worker", fake_run_worker)

    report = table_recovery.recover_image_only_tables(
        document=document,
        project_root=tmp_path,
        paddleocr_executable=executable,
    )

    table = document["blocks"][0]
    algorithm = document["blocks"][1]
    assert report["candidate_count"] == 1
    assert report["recovered_count"] == 1
    assert table["table_html"] == "<table><tr><td>42</td></tr></table>"
    assert table["quality"]["status"] == "usable"
    assert "table_html_empty" not in table["quality"]["issues"]
    assert algorithm.get("table_recovery") is None


def test_recovery_failure_keeps_image_only_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = make_document(tmp_path)
    fake_worker(tmp_path)
    executable = fake_executable(tmp_path)

    def failed_run_worker(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("fixture paddle failure")

    monkeypatch.setattr(table_recovery, "_run_worker", failed_run_worker)

    report = table_recovery.recover_image_only_tables(
        document=document,
        project_root=tmp_path,
        paddleocr_executable=executable,
    )

    table = document["blocks"][0]
    assert report["failed_count"] == 1
    assert table["quality"]["status"] == "image_only"
    assert table["quality"]["retrieval_enabled"] is True
    assert table["table_recovery"]["status"] == "failed"


def test_recovery_uses_cached_result_without_second_worker_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = make_document(tmp_path)
    fake_worker(tmp_path)
    executable = fake_executable(tmp_path)
    calls = 0

    def fake_run_worker(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        payload = {
            "done": True,
            "paddleocr_version": "fixture",
            "pred_html": "<table><tr><td>cached</td></tr></table>",
            "cell_box_list": [],
            "table_ocr_pred": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(table_recovery, "_run_worker", fake_run_worker)

    first = table_recovery.recover_image_only_tables(
        document=document,
        project_root=tmp_path,
        paddleocr_executable=executable,
    )
    second_document = make_document(tmp_path)
    second = table_recovery.recover_image_only_tables(
        document=second_document,
        project_root=tmp_path,
        paddleocr_executable=executable,
    )

    assert first["recovered_count"] == 1
    assert second["recovered_count"] == 1
    assert calls == 1
    assert second["records"][0]["cache_hit"] is True
