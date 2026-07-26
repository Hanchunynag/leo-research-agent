from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

import main as cli
from app.ingestion.batch import (
    batch_parse_directory,
    discover_pdfs,
)
from app.ingestion.ingest import calculate_sha256
from app.knowledge.catalog import load_catalog, rebuild_catalog
from app.parsing.pipeline import PaperParseConfig


def create_pdf(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), title)
    page.insert_text(
        (72, 100),
        "A technical paper fixture with enough native text for precheck. "
        * 4,
    )
    document.save(path)
    document.close()


def write_mineru_output(
    project_root: Path,
    pdf: Path,
    title: str,
) -> str:
    paper_id = f"P_{calculate_sha256(pdf)[:12]}"
    output = (
        project_root
        / "data"
        / "parsed"
        / paper_id
        / "mineru"
        / "fixture"
        / "txt"
    )
    output.mkdir(parents=True, exist_ok=True)
    content = [
        [
            {
                "type": "title",
                "bbox": [10, 10, 100, 30],
                "content": {
                    "level": 1,
                    "title_content": [
                        {"type": "text", "content": title}
                    ],
                },
            },
            {
                "type": "paragraph",
                "bbox": [10, 40, 100, 80],
                "content": {
                    "paragraph_content": [
                        {
                            "type": "text",
                            "content": "Fixture body for batch acceptance.",
                        }
                    ]
                },
            },
        ]
    ]
    middle = {
        "_backend": "pipeline",
        "_version_name": "3.4.4",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [612, 792],
                "para_blocks": [],
                "discarded_blocks": [],
            }
        ],
    }
    (output / "fixture_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )
    (output / "fixture_middle.json").write_text(
        json.dumps(middle),
        encoding="utf-8",
    )
    return paper_id


def test_discover_pdfs_supports_recursive_and_uppercase(
    tmp_path: Path,
) -> None:
    create_pdf(tmp_path / "top.pdf", "Top")
    create_pdf(tmp_path / "nested" / "paper.PDF", "Nested")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert len(discover_pdfs(tmp_path, recursive=False)) == 1
    assert len(discover_pdfs(tmp_path, recursive=True)) == 2


def test_discover_pdfs_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "paper.pdf"
    create_pdf(file_path, "Paper")

    with pytest.raises(ValueError, match="不是目录"):
        discover_pdfs(file_path, recursive=False)


def test_batch_acceptance_with_five_papers_and_one_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "论文样本"
    expected_ids: list[str] = []

    for index in range(1, 6):
        pdf = (
            input_directory
            / ("子目录" if index % 2 == 0 else "")
            / f"论文 {index}.pdf"
        )
        title = f"Batch Acceptance Paper {index}"
        create_pdf(pdf, title)
        expected_ids.append(
            write_mineru_output(tmp_path, pdf, title)
        )

    corrupt_pdf = input_directory / "损坏论文.pdf"
    corrupt_pdf.write_bytes(b"%PDF-corrupted-fixture")

    report = batch_parse_directory(
        input_directory=input_directory,
        config=PaperParseConfig(project_root=tmp_path),
        recursive=True,
    )

    assert report.total_count == 6
    assert report.reused_count == 5
    assert report.success_count == 0
    assert report.failed_count == 1
    assert report.catalog_record_count == 5
    assert report.catalog_issues == []
    assert report.report_path.exists()

    failed = [item for item in report.items if item.status == "failed"]
    assert len(failed) == 1
    assert failed[0].input_path.endswith("损坏论文.pdf")

    catalog = load_catalog(tmp_path)
    assert len(catalog.records) == 5
    assert {record.paper_id for record in catalog.records} == set(
        expected_ids
    )

    first_catalog = catalog.catalog_path.read_text(encoding="utf-8")
    rebuild_catalog(tmp_path)
    second_catalog = catalog.catalog_path.read_text(encoding="utf-8")
    assert first_catalog == second_catalog

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "batch",
                str(input_directory),
                "--recursive",
            ]
        )

    assert exit_info.value.code == 1
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["catalog_record_count"] == 5
    assert cli_report["reused_count"] == 5
    assert cli_report["failed_count"] == 1
    assert len(load_catalog(tmp_path).records) == 5
