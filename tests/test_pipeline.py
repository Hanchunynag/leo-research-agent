from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from app.ingestion.ingest import calculate_sha256
from app.normalization.mineru_adapter import (
    augment_native_pdf_text,
    filename_title_fallback,
    should_retry_with_ocr,
)
from app.parsing.pipeline import (
    PaperParseConfig,
    build_mineru_command,
    parse_paper,
    resolve_mineru_executable,
)


def create_pdf(path: Path, *, repeat: int = 4) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        (72, 72, 540, 740),
        "\n".join(
            "A native-text test paper with enough content for parsing."
            for _ in range(repeat)
        ),
    )
    document.save(path)
    document.close()


def write_existing_mineru_output(
    project_root: Path,
    paper_id: str,
) -> None:
    output = (
        project_root
        / "data"
        / "parsed"
        / paper_id
        / "mineru"
        / "test_paper"
        / "txt"
    )
    output.mkdir(parents=True)

    content = [
        [
            {
                "type": "title",
                "bbox": [10, 10, 100, 30],
                "content": {
                    "level": 1,
                    "title_content": [
                        {"type": "text", "content": "Test Paper"}
                    ],
                },
            },
            {
                "type": "paragraph",
                "bbox": [10, 40, 100, 80],
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "The model is"},
                        {"type": "equation_inline", "content": "x=1"},
                        {"type": "text", "content": "for this test."},
                    ]
                },
            },
            {
                "type": "equation_interline",
                "bbox": [10, 90, 100, 110],
                "content": {
                    "math_content": "x = 1",
                    "math_type": "latex",
                    "image_source": {"path": "images/equation.jpg"},
                },
            },
            {
                "type": "algorithm",
                "bbox": [10, 120, 100, 160],
                "content": {
                    "algorithm_caption": [],
                    "algorithm_content": [
                        {"type": "text", "content": "Algorithm 1: "},
                        {"type": "equation_inline", "content": "x\\gets1"},
                    ],
                    "algorithm_footnote": [],
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

    (output / "test_paper_content_list_v2.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )
    (output / "test_paper_middle.json").write_text(
        json.dumps(middle),
        encoding="utf-8",
    )


def test_pipeline_reuses_mineru_and_writes_one_paper_json(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "test_paper.pdf"
    create_pdf(pdf)

    paper_id = f"P_{calculate_sha256(pdf)[:12]}"
    write_existing_mineru_output(tmp_path, paper_id)

    result = parse_paper(
        input_path=pdf,
        config=PaperParseConfig(project_root=tmp_path),
    )

    assert result.paper_id == paper_id
    assert result.mineru_reused is True
    assert result.formula_count == 1
    assert result.paper_json.name == "paper.json"

    paper = json.loads(result.paper_json.read_text(encoding="utf-8"))
    assert paper["metadata"]["title"] == "Test Paper"
    assert paper["identity"]["document_id"] == f"D_{calculate_sha256(pdf)[:12]}"
    assert paper["identity"]["work_id"].startswith("W_")
    assert paper["identity"]["work_id_method"] == "document"
    assert paper["identity"]["status"] == "unresolved"
    assert paper["formulas"][0]["latex"] == "x = 1"
    assert paper["formulas"][0]["paper_id"] == paper_id
    paragraph = next(
        block for block in paper["blocks"] if block["type"] == "paragraph"
    )
    assert paragraph["text"] == "The model is $x=1$ for this test."
    assert paper["pipeline"]["mineru_reused"] is True
    assert paper["pipeline"]["formula_recovery_report"]["formula_count"] == 1
    assert paper["pipeline"]["formula_recovery_report"]["fallback_count"] == 0
    assert paper["source"]["sha256"] == calculate_sha256(pdf)

    algorithm = next(
        block for block in paper["blocks"] if block["type"] == "algorithm"
    )
    assert "Algorithm 1" in algorithm["text"]


def test_reparse_preserves_verified_external_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "test_paper.pdf"
    create_pdf(pdf)
    paper_id = f"P_{calculate_sha256(pdf)[:12]}"
    write_existing_mineru_output(tmp_path, paper_id)
    first = parse_paper(
        input_path=pdf,
        config=PaperParseConfig(project_root=tmp_path),
    )
    paper = json.loads(first.paper_json.read_text(encoding="utf-8"))
    paper["metadata"] = {
        "parser_title": "Test Paper",
        "title": "Externally Verified Test Paper",
        "authors": ["Ada Lovelace"],
        "abstract": "Verified abstract.",
        "year": 2025,
        "doi": "10.1000/test",
        "verification": {
            "status": "verified",
            "method": "academic-discovery-mcp",
        },
    }
    first.paper_json.write_text(json.dumps(paper), encoding="utf-8")

    second = parse_paper(
        input_path=pdf,
        config=PaperParseConfig(project_root=tmp_path),
    )
    reparsed = json.loads(second.paper_json.read_text(encoding="utf-8"))

    assert reparsed["metadata"]["parser_title"] == "Test Paper"
    assert reparsed["metadata"]["title"] == "Externally Verified Test Paper"
    assert reparsed["metadata"]["authors"] == ["Ada Lovelace"]
    assert reparsed["metadata"]["verification"]["status"] == "verified"


def test_native_pdf_text_fallback_recovers_low_coverage_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "native.pdf"
    create_pdf(pdf, repeat=30)
    document = {
        "paper_id": "P_aaaaaaaaaaaa",
        "blocks": [
            {
                "block_id": "P_aaaaaaaaaaaa_p001_b000",
                "paper_id": "P_aaaaaaaaaaaa",
                "page_number": 1,
                "reading_order": 0,
                "type": "paragraph",
                "text": "Short MinerU fragment.",
            }
        ],
    }

    report = augment_native_pdf_text(document, pdf)

    assert report["status"] == "applied"
    assert report["fallback_page_count"] == 1
    assert document["blocks"][-1]["source_type"] == "native_pdf_text"
    assert "native-text test paper" in document["blocks"][-1]["text"]


def test_degraded_cjk_ocr_is_detected_and_filename_title_is_safe() -> None:
    document = {
        "metadata": {
            "title": "Positionin<sub>g</sub> technolo<sub>gy</sub>",
        },
        "blocks": [
            {
                "type": "paragraph",
                "text": "Positionin<sub>g</sub> technolo<sub>gy</sub> based on IRIDIUM signals.",
                "quality": {"retrieval_enabled": True},
            }
        ],
    }

    decision = should_retry_with_ocr(
        document,
        "基于铱星机会信号的定位技术_秦红磊.pdf",
        pdf_type="native_text",
    )

    assert decision["retry"] is True
    assert decision["reason"] in {
        "cjk_filename_but_weak_cjk_text_layer",
        "malformed_title_and_weak_cjk_text_layer",
    }
    assert filename_title_fallback("基于铱星机会信号的定位技术_秦红磊.pdf") == "基于铱星机会信号的定位技术"


def test_good_cjk_text_does_not_trigger_ocr_retry() -> None:
    document = {
        "metadata": {"title": "基于铱星机会信号的定位技术"},
        "blocks": [
            {
                "type": "paragraph",
                "text": "基于铱星机会信号的定位技术采用瞬时多普勒定位方法，并通过单音信号测量多普勒频移。",
                "quality": {"retrieval_enabled": True},
            }
        ],
    }

    decision = should_retry_with_ocr(
        document,
        "基于铱星机会信号的定位技术_秦红磊.pdf",
        pdf_type="native_text",
    )

    assert decision["retry"] is False


def test_mineru_command_uses_dedicated_executable(tmp_path: Path) -> None:
    executable = tmp_path / ".venv-mineru" / "bin" / "mineru"
    config = PaperParseConfig(project_root=tmp_path)

    command = build_mineru_command(
        executable=executable,
        pdf_path=tmp_path / "paper.pdf",
        output_directory=tmp_path / "data" / "parsed",
        config=config,
    )

    assert command[0] == str(executable)
    assert "--formula" in command
    assert command[command.index("--formula") + 1] == "true"
    assert command[command.index("--table") + 1] == "true"


def test_resolve_mineru_executable_only_uses_dedicated_venv(
    tmp_path: Path,
) -> None:
    executable = tmp_path / ".venv-mineru" / "bin" / "mineru"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    assert resolve_mineru_executable(tmp_path) == executable.resolve()
