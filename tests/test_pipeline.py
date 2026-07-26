from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from app.ingestion.ingest import calculate_sha256
from app.parsing.pipeline import (
    PaperParseConfig,
    build_mineru_command,
    parse_paper,
    resolve_mineru_executable,
)


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "A native-text test paper with enough content for parsing. " * 4,
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
    assert paper["formulas"][0]["latex"] == "x = 1"
    assert paper["formulas"][0]["paper_id"] == paper_id
    paragraph = next(
        block for block in paper["blocks"] if block["type"] == "paragraph"
    )
    assert paragraph["text"] == "The model is $x=1$ for this test."
    assert paper["pipeline"]["mineru_reused"] is True
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
