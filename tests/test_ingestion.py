from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.ingest import ingest_paper


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A valid paper used by ingestion tests.")
    document.save(path)
    document.close()


def test_ingest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_paper(
            input_path=tmp_path / "missing.pdf",
            raw_dir=tmp_path / "data" / "raw",
        )


def test_ingest_rejects_non_pdf_extension(tmp_path: Path) -> None:
    text_file = tmp_path / "paper.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="只支持 PDF"):
        ingest_paper(
            input_path=text_file,
            raw_dir=tmp_path / "data" / "raw",
        )


def test_ingest_rejects_fake_pdf_header(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="不是有效 PDF"):
        ingest_paper(
            input_path=fake_pdf,
            raw_dir=tmp_path / "data" / "raw",
        )


def test_same_pdf_with_different_name_reuses_stored_file(
    tmp_path: Path,
) -> None:
    first_pdf = tmp_path / "first.pdf"
    renamed_pdf = tmp_path / "第二篇 星历修正（最终版）.pdf"
    create_pdf(first_pdf)
    renamed_pdf.write_bytes(first_pdf.read_bytes())

    raw_dir = tmp_path / "data" / "raw"
    first = ingest_paper(first_pdf, raw_dir)
    second = ingest_paper(renamed_pdf, raw_dir)

    assert second.paper_id == first.paper_id
    assert second.sha256 == first.sha256
    assert second.source_path == first.source_path
    assert second.original_filename == renamed_pdf.name
    assert len(list(first.source_path.parent.glob("*.pdf"))) == 1


def test_ingest_supports_chinese_and_spaces_in_filename(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "第二篇 星历修正 最终版.pdf"
    create_pdf(pdf)

    result = ingest_paper(
        input_path=pdf,
        raw_dir=tmp_path / "data" / "raw",
    )

    assert result.stored_filename == "第二篇_星历修正_最终版.pdf"
    assert result.source_path.is_file()
