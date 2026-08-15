from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsing import formula_recovery


def equation_document(latex: object, image_path: str | None = None) -> dict[str, object]:
    return {
        "paper_id": "P_formula_fixture",
        "blocks": [
            {
                "block_id": "P_formula_fixture_p001_b001",
                "type": "equation",
                "latex": latex,
                "latex_raw": latex,
                "image_path": image_path,
                "quality": {
                    "status": "degraded" if not isinstance(latex, str) or not latex else "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            }
        ],
    }


def fake_paddle_executable(tmp_path: Path) -> Path:
    executable = tmp_path / ".venv-paddleocr" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def fake_worker(tmp_path: Path) -> None:
    worker = tmp_path / "scripts" / "paddleocr_formula_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# fixture worker\n", encoding="utf-8")


def test_hard_error_rule_detects_empty_control_and_mismatched_latex() -> None:
    assert "formula_latex_empty" in formula_recovery.formula_hard_error_issues("")
    assert "formula_control_character" in formula_recovery.formula_hard_error_issues(
        "x\x03=1"
    )
    assert "formula_environment_mismatch" in formula_recovery.formula_hard_error_issues(
        r"\begin{array}{c}x"
    )
    assert "formula_unbalanced_delimiter" in formula_recovery.formula_hard_error_issues(
        r"\frac{x}{y"
    )
    assert "formula_left_right_mismatch" in formula_recovery.formula_hard_error_issues(
        r"\left( x + y"
    )


def test_high_risk_rule_accepts_simple_formula_and_flags_matrix() -> None:
    assert not formula_recovery.is_high_risk_formula(r"x_k=Fx_{k-1}+w_k")
    assert formula_recovery.is_high_risk_formula(
        r"\begin{bmatrix}a&b\\c&d\end{bmatrix}"
    )
    assert formula_recovery.is_high_risk_formula("x" * 350)


def test_no_risk_formula_does_not_require_paddleocr(tmp_path: Path) -> None:
    document = equation_document(r"x_k=Fx_{k-1}+w_k")

    report = formula_recovery.recover_high_risk_formulas(
        document=document,
        project_root=tmp_path,
    )

    assert report["formula_count"] == 1
    assert report["hard_error_count"] == 0
    assert report["high_risk_count"] == 0
    assert report["fallback_count"] == 0
    assert report["records"] == []


def test_hard_error_without_image_is_recorded_without_worker(
    tmp_path: Path,
) -> None:
    document = equation_document("")
    fake_worker(tmp_path)
    fake_paddle_executable(tmp_path)
    report = formula_recovery.recover_high_risk_formulas(
        document=document,
        project_root=tmp_path,
    )

    assert report["hard_error_count"] == 1
    assert report["skipped_no_image_count"] == 1
    assert report["fallback_count"] == 0
    assert report["records"][0]["status"] == "skipped_no_image"


def test_successful_fallback_preserves_mineru_latex_and_updates_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "formula.jpg"
    image.write_bytes(b"formula-image")
    mineru_latex = r"\begin{bmatrix}a&b\\c&d\end{bmatrix}"
    document = equation_document(mineru_latex, "formula.jpg")
    fake_worker(tmp_path)
    executable = fake_paddle_executable(tmp_path)

    def fake_run_worker_batch(**kwargs: object) -> None:
        manifest_path = kwargs["manifest_path"]
        assert isinstance(manifest_path, Path)
        jobs = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert isinstance(jobs, list)
        output_path = Path(jobs[0]["output_json"])
        payload = {
            "done": True,
            "paddleocr_version": "fixture",
            "latex": r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        formula_recovery,
        "_run_worker_batch",
        fake_run_worker_batch,
    )
    report = formula_recovery.recover_high_risk_formulas(
        document=document,
        project_root=tmp_path,
        paddleocr_executable=executable,
    )

    block = document["blocks"][0]
    assert report["fallback_count"] == 1
    assert report["recovered_count"] == 1
    assert block["latex_mineru"] == mineru_latex
    assert block["latex_paddleocr_vl"] == r"\begin{pmatrix}1&2\\3&4\end{pmatrix}"
    assert block["latex"] == block["latex_paddleocr_vl"]
    assert block["quality"]["status"] == "usable"
    assert report["records"][0]["status"] == "recovered"


def test_cached_fallback_does_not_call_worker_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "formula.jpg"
    image.write_bytes(b"formula-image")
    latex = r"\begin{array}{cc}a&b\\c&d\end{array}"
    fake_worker(tmp_path)
    executable = fake_paddle_executable(tmp_path)
    calls = 0

    def fake_run_worker_batch(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        manifest_path = kwargs["manifest_path"]
        assert isinstance(manifest_path, Path)
        jobs = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert isinstance(jobs, list)
        output_path = Path(jobs[0]["output_json"])
        payload = {"done": True, "latex": "a+b"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        formula_recovery,
        "_run_worker_batch",
        fake_run_worker_batch,
    )
    first = formula_recovery.recover_high_risk_formulas(
        equation_document(latex, "formula.jpg"),
        tmp_path,
        paddleocr_executable=executable,
    )
    second = formula_recovery.recover_high_risk_formulas(
        equation_document(latex, "formula.jpg"),
        tmp_path,
        paddleocr_executable=executable,
    )

    assert first["recovered_count"] == 1
    assert second["recovered_count"] == 1
    assert second["records"][0]["cache_hit"] is True
    assert calls == 1
