from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.ingest import calculate_sha256
from app.parsing.pipeline import (
    MinerUExecutionError,
    PaperParseConfig,
    find_mineru_artifacts,
    parse_paper,
    resolve_mineru_executable,
    run_mineru,
)


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "A native-text test paper with enough content for precheck. " * 4,
    )
    document.save(path)
    document.close()


def write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_resolve_mineru_executable_reports_checked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEO_MINERU_EXECUTABLE", raising=False)

    with pytest.raises(FileNotFoundError) as error:
        resolve_mineru_executable(tmp_path)

    message = str(error.value)
    assert "请创建 .venv-mineru" in message
    assert ".venv-mineru/bin/mineru" in message
    assert ".venv-mineru/Scripts/mineru.exe" in message


def test_find_mineru_artifacts_rejects_incomplete_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "paper" / "txt"
    output.mkdir(parents=True)
    (output / "paper_content_list_v2.json").write_text(
        "[]",
        encoding="utf-8",
    )

    assert find_mineru_artifacts(tmp_path) is None


def test_run_mineru_failure_writes_stdout_and_stderr_logs(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fake-mineru"
    write_executable(
        executable,
        (
            "import sys\n"
            "print('simulated stdout')\n"
            "print('simulated failure', file=sys.stderr)\n"
            "raise SystemExit(7)\n"
        ),
    )
    mineru_root = tmp_path / "data" / "parsed" / "mineru"
    config = PaperParseConfig(
        project_root=tmp_path,
        mineru_executable=executable,
    )

    with pytest.raises(
        MinerUExecutionError,
        match="退出码 7",
    ):
        run_mineru(
            pdf_path=tmp_path / "paper.pdf",
            mineru_root=mineru_root,
            config=config,
        )

    log_directory = mineru_root / "_pipeline"
    assert (
        log_directory / "mineru.stdout.log"
    ).read_text(encoding="utf-8") == "simulated stdout\n"
    assert (
        log_directory / "mineru.stderr.log"
    ).read_text(encoding="utf-8") == "simulated failure\n"


def test_run_mineru_rejects_success_without_required_artifacts(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fake-mineru"
    write_executable(executable, "print('finished without output')\n")
    mineru_root = tmp_path / "data" / "parsed" / "mineru"
    config = PaperParseConfig(
        project_root=tmp_path,
        mineru_executable=executable,
    )

    with pytest.raises(
        MinerUExecutionError,
        match="没有找到 content_list_v2.json 和 middle.json",
    ):
        run_mineru(
            pdf_path=tmp_path / "paper.pdf",
            mineru_root=mineru_root,
            config=config,
        )

    assert (
        mineru_root / "_pipeline" / "mineru.stdout.log"
    ).read_text(encoding="utf-8") == "finished without output\n"


def test_corrupted_pdf_does_not_write_canonical_document(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "corrupted.pdf"
    pdf.write_bytes(b"%PDF-this-is-not-a-valid-document")
    paper_id = f"P_{calculate_sha256(pdf)[:12]}"

    with pytest.raises(pymupdf.FileDataError):
        parse_paper(
            input_path=pdf,
            config=PaperParseConfig(project_root=tmp_path),
        )

    assert not (
        tmp_path / "data" / "canonical" / paper_id / "paper.json"
    ).exists()


def test_missing_mineru_does_not_write_canonical_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEO_MINERU_EXECUTABLE", raising=False)
    pdf = tmp_path / "paper.pdf"
    create_pdf(pdf)
    paper_id = f"P_{calculate_sha256(pdf)[:12]}"

    with pytest.raises(FileNotFoundError, match="未找到 MinerU"):
        parse_paper(
            input_path=pdf,
            config=PaperParseConfig(project_root=tmp_path),
        )

    assert not (
        tmp_path / "data" / "canonical" / paper_id / "paper.json"
    ).exists()
