from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import main as cli
from app.knowledge.catalog import (
    catalog_path,
    library_status,
    load_catalog,
    rebuild_catalog,
)


def paper_identity(seed: str) -> tuple[str, str]:
    sha256 = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"P_{sha256[:12]}", sha256


def write_canonical_paper(
    project_root: Path,
    seed: str,
    title: str,
    quality_issues: int = 0,
) -> str:
    paper_id, sha256 = paper_identity(seed)
    output = (
        project_root
        / "data"
        / "canonical"
        / paper_id
        / "paper.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "paper_id": paper_id,
                "metadata": {
                    "title": title,
                    "authors": [],
                    "year": None,
                    "doi": None,
                },
                "parser": {
                    "name": "mineru",
                    "version": "3.4.4",
                },
                "source": {
                    "sha256": sha256,
                    "raw_pdf": (
                        f"data/raw/{paper_id}/paper.pdf"
                    ),
                },
                "page_count": 1,
                "pipeline": {
                    "created_at": "2026-07-25T00:00:00+00:00",
                    "adapter_report": {
                        "quality_issue_counts": {
                            "fixture_issue": quality_issues,
                        },
                        "missing_asset_count": 0,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paper_id


def write_raw_and_parsed_markers(
    project_root: Path,
    paper_id: str,
) -> None:
    raw_pdf = (
        project_root / "data" / "raw" / paper_id / "paper.pdf"
    )
    raw_pdf.parent.mkdir(parents=True, exist_ok=True)
    raw_pdf.write_bytes(b"%PDF-fixture")

    output = (
        project_root
        / "data"
        / "parsed"
        / paper_id
        / "mineru"
        / "paper"
        / "txt"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_content_list_v2.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (output / "paper_middle.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_rebuild_catalog_is_sorted_and_deterministic(
    tmp_path: Path,
) -> None:
    paper_ids = [
        write_canonical_paper(
            tmp_path,
            seed=f"paper-{index}",
            title=f"Paper {index}",
            quality_issues=index,
        )
        for index in range(5)
    ]

    first = rebuild_catalog(tmp_path)
    first_content = first.catalog_path.read_text(encoding="utf-8")
    second = rebuild_catalog(tmp_path)
    second_content = second.catalog_path.read_text(encoding="utf-8")

    assert len(first.records) == 5
    assert [record.paper_id for record in first.records] == sorted(
        paper_ids
    )
    assert sum(
        record.quality_issue_count for record in first.records
    ) == 10
    assert first_content == second_content
    assert len(second.records) == 5


def test_rebuild_catalog_reports_corrupt_paper_without_blocking(
    tmp_path: Path,
) -> None:
    paper_id = write_canonical_paper(
        tmp_path,
        seed="valid",
        title="Valid Paper",
    )
    corrupt = (
        tmp_path
        / "data"
        / "canonical"
        / "P_000000000000"
        / "paper.json"
    )
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{broken", encoding="utf-8")

    result = rebuild_catalog(tmp_path)

    assert [record.paper_id for record in result.records] == [paper_id]
    assert len(result.issues) == 1
    assert result.issues[0].error_type == "JSONDecodeError"
    assert len(load_catalog(tmp_path).records) == 1


def test_load_catalog_reports_invalid_record_type(
    tmp_path: Path,
) -> None:
    write_canonical_paper(
        tmp_path,
        seed="catalog-type",
        title="Catalog Type Paper",
    )
    result = rebuild_catalog(tmp_path)
    valid_line = result.catalog_path.read_text(
        encoding="utf-8"
    ).strip()
    invalid_record = json.loads(valid_line)
    invalid_record["paper_id"] = "P_000000000000"
    invalid_record["year"] = "not-a-year"
    result.catalog_path.write_text(
        valid_line + "\n" + json.dumps(invalid_record) + "\n",
        encoding="utf-8",
    )

    loaded = load_catalog(tmp_path)

    assert len(loaded.records) == 1
    assert len(loaded.issues) == 1
    assert "year" in loaded.issues[0].message


def test_library_status_detects_consistent_store(
    tmp_path: Path,
) -> None:
    paper_id = write_canonical_paper(
        tmp_path,
        seed="complete",
        title="Complete Paper",
        quality_issues=2,
    )
    write_raw_and_parsed_markers(tmp_path, paper_id)
    rebuild_catalog(tmp_path)

    status = library_status(tmp_path)

    assert status.catalog_consistent is True
    assert status.raw_paper_count == 1
    assert status.parsed_paper_count == 1
    assert status.canonical_valid_count == 1
    assert status.catalog_record_count == 1
    assert status.quality_issue_count == 2
    assert status.unparsed_paper_ids == []


def test_library_status_detects_stale_catalog(
    tmp_path: Path,
) -> None:
    paper_id = write_canonical_paper(
        tmp_path,
        seed="stale",
        title="Original Title",
    )
    write_raw_and_parsed_markers(tmp_path, paper_id)
    rebuild_catalog(tmp_path)

    canonical = (
        tmp_path
        / "data"
        / "canonical"
        / paper_id
        / "paper.json"
    )
    document = json.loads(canonical.read_text(encoding="utf-8"))
    document["metadata"]["title"] = "Updated Title"
    canonical.write_text(json.dumps(document), encoding="utf-8")

    status = library_status(tmp_path)

    assert status.catalog_consistent is False
    assert status.catalog_stale_paper_ids == [paper_id]


def test_library_commands_rebuild_list_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paper_id = write_canonical_paper(
        tmp_path,
        seed="cli",
        title="CLI Paper",
    )
    write_raw_and_parsed_markers(tmp_path, paper_id)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    cli.main(["library", "rebuild"])
    rebuild_output = json.loads(capsys.readouterr().out)
    assert rebuild_output["record_count"] == 1

    cli.main(["library", "list"])
    list_output = json.loads(capsys.readouterr().out)
    assert list_output["records"][0]["paper_id"] == paper_id

    cli.main(["library", "status"])
    status_output = json.loads(capsys.readouterr().out)
    assert status_output["catalog_consistent"] is True
    assert catalog_path(tmp_path).exists()

    cli.main(["library", "works"])
    works_output = json.loads(capsys.readouterr().out)
    assert works_output["record_count"] == 0
    assert works_output["unresolved_paper_ids"] == [paper_id]
