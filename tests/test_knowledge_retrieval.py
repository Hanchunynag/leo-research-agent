from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import main as cli
from app.chunking.builder import build_knowledge_base
from app.chunking.chunker import build_chunks
from app.chunking.structure import (
    build_structure,
    infer_heading_level,
    is_false_heading,
)
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.retrieval.search import search_evidence
from app.storage import write_jsonl_atomic


def block(
    index: int,
    block_type: str,
    text: str = "",
    *,
    page: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "block_id": f"P_test_p{page:03d}_b{index:03d}",
        "page_number": page,
        "reading_order": index,
        "type": block_type,
        "text": text,
        "quality": {"retrieval_enabled": True},
        **extra,
    }


def canonical_document(
    *,
    paper_id: str = "P_aaaaaaaaaaaa",
    document_id: str = "D_aaaaaaaaaaaa",
    work_id: str = "W_aaaaaaaaaaaa",
    title: str = "LEO Evidence Paper",
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "paper_id": paper_id,
        "identity": {
            "document_id": document_id,
            "work_id": work_id,
        },
        "metadata": {
            "title": title,
            "authors": ["Ada Researcher"],
            "abstract": "A short orbit prediction summary.",
            "year": 2026,
            "doi": "10.1000/leo",
        },
        "source": {"sha256": "a" * 64},
        "blocks": blocks
        or [
            block(0, "title", title, title_level_raw=1),
            block(1, "list", "Ada Researcher, Example University"),
            block(2, "paragraph", "Abstract—A short orbit prediction summary."),
            block(3, "paragraph", "Index Terms—LEO, ephemeris."),
            block(4, "title", "I. INTRODUCTION", page=2, title_level_raw=2),
            block(5, "paragraph", "Introduction evidence alpha.", page=2),
            block(6, "title", "A. Method", page=2, title_level_raw=2),
            block(7, "paragraph", "Method context before equation.", page=2),
            block(8, "equation", page=2, latex="x = vt"),
            block(9, "paragraph", "Equation explanation after asset.", page=2),
            block(
                10,
                "title",
                "Figure 5<sub>.</sub>",
                page=2,
                title_level_raw=2,
            ),
            block(11, "title", "II. RESULTS", page=3, title_level_raw=2),
            block(12, "paragraph", "Results evidence beta.", page=3),
            block(13, "title", "REFERENCES", page=4, title_level_raw=2),
            block(14, "paragraph", "Secret reference-only phrase.", page=4),
            block(15, "page_metadata", "4", page=4),
        ],
    }


def write_canonical(root: Path, document: dict[str, Any]) -> Path:
    paper_id = str(document["paper_id"])
    output = root / "data" / "canonical" / paper_id / "paper.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document), encoding="utf-8")
    return output


def make_chunk(
    chunk_id: str,
    work_id: str,
    document_id: str,
    content: str,
    *,
    title: str = "Paper",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "work_id": work_id,
        "document_id": document_id,
        "paper_id": document_id.replace("D_", "P_"),
        "title": title,
        "authors": [],
        "year": 2026,
        "doi": None,
        "section_path": ["RESULTS"],
        "content_zone": "main_body",
        "page_start": 2,
        "page_end": 2,
        "block_ids": [f"{document_id}_b1"],
        "content_types": ["paragraph"],
        "content": content,
    }


def write_search_store(root: Path, chunks: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(root / "data" / "knowledge" / "chunks.jsonl", chunks)
    write_bm25_index(root, build_bm25_index(chunks))


def test_heading_levels_zones_and_false_heading_detection() -> None:
    assert infer_heading_level("IV. SIMULATION RESULTS") == 1
    assert infer_heading_level("3.2 Filter Design") == 2
    assert infer_heading_level("A. Measurement Model") == 2
    assert is_false_heading("Figure 5<sub>.</sub>") is True
    assert is_false_heading("Table II: Error Results") is True
    assert is_false_heading("TABLE OF CONTENTS") is False

    structure = build_structure(canonical_document())
    section_titles = [section["title"] for section in structure["sections"]]
    assert "Figure 5." not in section_titles
    assert section_titles == [
        "I. INTRODUCTION",
        "A. Method",
        "II. RESULTS",
        "REFERENCES",
    ]

    by_id = {value["block_id"]: value for value in structure["blocks"]}
    assert by_id["P_test_p001_b001"]["searchable"] is False
    assert by_id["P_test_p001_b002"]["content_zone"] == "abstract"
    assert by_id["P_test_p001_b003"]["content_zone"] == "abstract"
    assert by_id["P_test_p001_b003"]["searchable"] is True
    assert by_id["P_test_p004_b014"]["exclusion_reason"] == "zone:references"
    assert by_id["P_test_p004_b015"]["exclusion_reason"] == "page_metadata"


def test_asset_links_nearest_text_within_same_section() -> None:
    structure = build_structure(canonical_document())
    equation = next(
        value for value in structure["blocks"] if value["type"] == "equation"
    )
    assert equation["content"] == "[Equation]\nx = vt"
    assert equation["related_text_block_ids"] == [
        "P_test_p002_b007",
        "P_test_p002_b009",
    ]


def test_verified_abstract_aligns_split_pdf_blocks_without_including_front_matter() -> (
    None
):
    abstract_start = (
        "A framework for precise low Earth orbit satellite tracking is developed "
        "using pseudorange measurements and a robust dynamic model."
    )
    abstract_end = (
        "Simulation and experimental results demonstrate improved positioning "
        "accuracy over open-loop propagated ephemerides."
    )
    document = canonical_document(
        blocks=[
            block(0, "title", "LEO Evidence Paper", title_level_raw=1),
            block(1, "paragraph", "Ada Researcher, Example University"),
            block(2, "paragraph", abstract_start),
            block(3, "paragraph", "Received 1 January 2026. DOI 10.1000/leo."),
            block(4, "paragraph", abstract_end),
            block(5, "title", "I. INTRODUCTION", title_level_raw=2),
            block(6, "paragraph", "Main body evidence."),
        ]
    )
    document["metadata"]["abstract"] = (
        f"{abstract_start.replace('precise', 'high-precision')} "
        f"{abstract_end.replace('open-loop', 'open loop')}"
    )

    structure = build_structure(document)
    by_id = {value["block_id"]: value for value in structure["blocks"]}

    assert by_id["P_test_p001_b002"]["content_zone"] == "abstract"
    assert by_id["P_test_p001_b002"]["searchable"] is True
    assert by_id["P_test_p001_b003"]["content_zone"] == "front_matter"
    assert by_id["P_test_p001_b003"]["searchable"] is False
    assert by_id["P_test_p001_b004"]["content_zone"] == "abstract"
    assert by_id["P_test_p001_b004"]["searchable"] is True


def test_chunks_are_deterministic_bounded_and_do_not_cross_sections() -> None:
    structure = build_structure(canonical_document())
    first = build_chunks(
        structure,
        maximum_tokens=50,
        minimum_chunk_tokens=20,
        overlap_tokens=10,
    )
    second = build_chunks(
        structure,
        maximum_tokens=50,
        minimum_chunk_tokens=20,
        overlap_tokens=10,
    )

    assert first == second
    assert first["chunk_count"] >= 3
    assert [chunk["chunk_id"] for chunk in first["chunks"]] == [
        f"D_aaaaaaaaaaaa_cp02_c{index:06d}"
        for index in range(1, first["chunk_count"] + 1)
    ]
    assert all(chunk["token_count"] <= 50 for chunk in first["chunks"])
    assert not any(
        "Introduction evidence alpha" in chunk["content"]
        and "Results evidence beta" in chunk["content"]
        for chunk in first["chunks"]
    )
    assert all(
        "Secret reference-only phrase" not in chunk["content"]
        for chunk in first["chunks"]
    )


def test_small_parent_chunk_becomes_explicit_child_context() -> None:
    document = canonical_document(
        blocks=[
            block(0, "title", "LEO Evidence Paper", title_level_raw=1),
            block(1, "title", "I. INTRODUCTION", title_level_raw=2),
            block(2, "paragraph", "Short model overview."),
            block(3, "title", "A. Dynamics", title_level_raw=2),
            block(
                4,
                "paragraph",
                "Detailed satellite dynamics evidence with force and clock models.",
            ),
        ]
    )

    collection = build_chunks(
        build_structure(document),
        maximum_tokens=100,
        minimum_chunk_tokens=20,
        overlap_tokens=10,
    )

    assert collection["absorbed_parent_chunk_count"] == 1
    assert collection["chunk_count"] == 1
    chunk = collection["chunks"][0]
    assert chunk["section_path"] == ["I. INTRODUCTION", "A. Dynamics"]
    assert "Short model overview" not in chunk["content"]
    assert chunk["parent_contexts"][0]["section_path"] == ["I. INTRODUCTION"]
    assert chunk["parent_contexts"][0]["content"] == "Short model overview."
    assert chunk["parent_contexts"][0]["block_ids"] == ["P_test_p001_b002"]
    assert chunk["parent_contexts"][0]["document_id"] == chunk["document_id"]


def test_overlap_is_bounded_and_never_crosses_section_boundary() -> None:
    repeated = " ".join(f"term{index}" for index in range(35))
    document = canonical_document(
        blocks=[
            block(0, "title", "LEO Evidence Paper", title_level_raw=1),
            block(1, "title", "I. INTRODUCTION", title_level_raw=2),
            block(2, "paragraph", repeated),
            block(3, "paragraph", repeated),
            block(4, "title", "II. RESULTS", title_level_raw=2),
            block(5, "paragraph", repeated),
        ]
    )

    collection = build_chunks(
        build_structure(document),
        maximum_tokens=50,
        minimum_chunk_tokens=0,
        overlap_tokens=10,
    )
    chunks = collection["chunks"]

    assert collection["overlap_context_count"] == 1
    assert chunks[0]["overlap_context"] is None
    assert chunks[1]["overlap_context"]["section_id"] == chunks[0]["section_id"]
    assert chunks[1]["overlap_context"]["token_count"] <= 10
    assert chunks[1]["overlap_context"]["block_ids"] == chunks[0]["block_ids"]
    assert chunks[1]["overlap_context"]["source_chunk_id"] == chunks[0]["chunk_id"]
    assert chunks[1]["overlap_context"]["document_id"] == chunks[1]["document_id"]
    result_chunk = next(
        chunk for chunk in chunks if chunk["section_path"] == ["II. RESULTS"]
    )
    assert result_chunk["overlap_context"] is None


def test_knowledge_build_reuses_unchanged_document(tmp_path: Path) -> None:
    write_canonical(tmp_path, canonical_document())

    first = build_knowledge_base(tmp_path, maximum_tokens=100)
    second = build_knowledge_base(tmp_path, maximum_tokens=100)

    assert first.document_count == 1
    assert first.structure_built_count == 1
    assert first.chunks_built_count == 1
    assert second.structure_reused_count == 1
    assert second.chunks_reused_count == 1
    assert second.total_chunk_count == first.total_chunk_count
    assert (tmp_path / second.chunks_jsonl).is_file()
    assert (tmp_path / second.bm25_index).is_file()


def test_bm25_search_filters_deduplicates_and_cites(tmp_path: Path) -> None:
    chunks = [
        make_chunk("D_a_cp01_c000001", "W_same", "D_a", "ephemeris timing error"),
        make_chunk("D_a_cp01_c000002", "W_same", "D_a", "ephemeris correction"),
        make_chunk("D_b_cp01_c000001", "W_other", "D_b", "timing ephemeris model"),
        make_chunk("D_c_cp01_c000001", "W_third", "D_c", "unrelated evidence"),
    ]
    write_search_store(tmp_path, chunks)

    result = search_evidence(
        tmp_path,
        "ephemeris timing",
        limit=10,
        max_chunks_per_work=1,
    )

    assert result["result_count"] == 2
    assert [item["work_id"] for item in result["results"]] == [
        "W_same",
        "W_other",
    ]
    assert result["results"][0]["citation"] == "D_a pp. 2-2"
    assert result["results"][0]["block_ids"] == ["D_a_b1"]

    filtered = search_evidence(tmp_path, "ephemeris", document_id="D_b")
    assert [item["document_id"] for item in filtered["results"]] == ["D_b"]


def test_bm25_indexes_parent_and_overlap_context_without_hiding_boundaries(
    tmp_path: Path,
) -> None:
    chunk = make_chunk(
        "D_a_cp02_c000001",
        "W_a",
        "D_a",
        "primary dynamics evidence",
    )
    chunk["parent_contexts"] = [
        {
            "section_path": ["I. MODEL"],
            "content": "rare parent bridge phrase",
            "block_ids": ["D_a_parent"],
            "page_start": 1,
            "page_end": 1,
            "token_count": 4,
        }
    ]
    chunk["overlap_context"] = {
        "section_path": ["I. MODEL", "A. Dynamics"],
        "content": "previous covariance context",
        "block_ids": ["D_a_previous"],
        "page_start": 2,
        "page_end": 2,
        "token_count": 3,
    }
    write_search_store(tmp_path, [chunk])

    parent_result = search_evidence(tmp_path, "rare parent bridge")
    overlap_result = search_evidence(tmp_path, "previous covariance")

    assert parent_result["results"][0]["parent_contexts"][0]["block_ids"] == [
        "D_a_parent"
    ]
    assert overlap_result["results"][0]["overlap_context"]["block_ids"] == [
        "D_a_previous"
    ]


def test_search_rejects_stale_index(tmp_path: Path) -> None:
    chunks = [make_chunk("D_a_cp01_c000001", "W_a", "D_a", "orbit evidence")]
    write_search_store(tmp_path, chunks)
    chunks[0]["content"] = "changed after indexing"
    write_jsonl_atomic(tmp_path / "data" / "knowledge" / "chunks.jsonl", chunks)

    with pytest.raises(RuntimeError, match="不一致"):
        search_evidence(tmp_path, "orbit")


def test_knowledge_build_and_search_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_canonical(tmp_path, canonical_document())
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    cli.main(
        [
            "knowledge",
            "build",
            "--max-tokens",
            "100",
            "--min-chunk-tokens",
            "20",
            "--overlap-tokens",
            "10",
        ]
    )
    build_output = json.loads(capsys.readouterr().out)
    assert build_output["document_count"] == 1
    assert build_output["total_chunk_count"] > 0
    assert build_output["minimum_chunk_tokens"] == 20
    assert build_output["overlap_tokens"] == 10

    cli.main(["search", "orbit prediction", "--limit", "2"])
    search_output = json.loads(capsys.readouterr().out)
    assert search_output["query"] == "orbit prediction"
    assert search_output["result_count"] > 0


def test_invalid_chunk_limit_is_rejected_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不能小于 50"):
        build_knowledge_base(tmp_path, maximum_tokens=49)
    assert not (tmp_path / "data").exists()
