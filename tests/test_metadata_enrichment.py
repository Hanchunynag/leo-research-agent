from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.knowledge.catalog import load_catalog
from app.knowledge.metadata_enrichment import (
    enrich_paper_metadata,
    normalize_verified_paper,
)


class FakeResolver:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.titles: list[str] = []

    async def resolve_paper(
        self,
        title: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        self.titles.append(title)
        assert limit > 0
        return self.payload


def write_paper(project_root: Path, title: str = "Extracted Paper") -> str:
    pdf_bytes = f"%PDF-fixture-{title}".encode("utf-8")
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    paper_id = f"P_{sha256[:12]}"
    raw_pdf = project_root / "data" / "raw" / paper_id / "uploaded.pdf"
    raw_pdf.parent.mkdir(parents=True)
    raw_pdf.write_bytes(pdf_bytes)
    paper_json = project_root / "data" / "canonical" / paper_id / "paper.json"
    paper_json.parent.mkdir(parents=True)
    paper_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "paper_id": paper_id,
                "metadata": {
                    "title": title,
                    "authors": [],
                    "abstract": None,
                },
                "source": {
                    "sha256": sha256,
                    "original_filename": "uploaded.pdf",
                    "stored_filename": "uploaded.pdf",
                    "raw_pdf": f"data/raw/{paper_id}/uploaded.pdf",
                },
                "precheck": {
                    "source_path": f"data/raw/{paper_id}/uploaded.pdf",
                },
                "parser": {"name": "mineru", "version": "3.4.4"},
                "page_count": 1,
                "pipeline": {
                    "created_at": "2026-07-26T00:00:00+00:00",
                    "adapter_report": {
                        "quality_issue_counts": {},
                        "missing_asset_count": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return paper_id


def verified_payload() -> dict[str, Any]:
    return {
        "result_count": 2,
        "results": [
            {
                "title": "Verified Paper",
                "authors": ["Ada Lovelace", "Grace Hopper"],
                "abstract": "Verified abstract.",
                "publication_year": 2025,
                "doi": "10.1000/verified",
                "venue": "Navigation Journal",
                "external_ids": {
                    "doi": "10.1000/verified",
                    "openalex": "W123",
                },
                "sources": ["crossref", "openalex"],
                "match_score": 1.0,
            },
            {
                "title": "Different Paper",
                "authors": ["Other Author"],
                "publication_year": 2024,
                "doi": "10.1000/other",
                "sources": ["crossref"],
                "match_score": 0.4,
            },
        ],
        "provider_failures": [],
    }


def test_strict_verified_candidate_updates_paper_and_catalog(
    tmp_path: Path,
) -> None:
    paper_id = write_paper(tmp_path)
    resolver = FakeResolver(verified_payload())

    result = asyncio.run(
        enrich_paper_metadata(
            project_root=tmp_path,
            paper_id=paper_id,
            resolver=resolver,
        )
    )

    assert result.updated is True
    assert result.decision.status == "verified"
    assert result.review_path is not None and result.review_path.exists()
    assert resolver.titles == ["Extracted Paper"]

    paper = json.loads(result.paper_json.read_text(encoding="utf-8"))
    metadata = paper["metadata"]
    assert metadata["parser_title"] == "Extracted Paper"
    assert metadata["title"] == "Verified Paper"
    assert metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert metadata["doi"] == "10.1000/verified"
    assert metadata["verification"]["status"] == "verified"
    assert paper["identity"]["document_id"].startswith("D_")
    assert paper["identity"]["work_id"].startswith("W_")
    assert paper["identity"]["work_id_method"] == "doi"
    assert paper["source"]["stored_filename"] == "Verified Paper.pdf"
    assert paper["precheck"]["source_path"].endswith("/Verified Paper.pdf")
    assert not (tmp_path / "data" / "raw" / paper_id / "uploaded.pdf").exists()
    assert (tmp_path / "data" / "raw" / paper_id / "Verified Paper.pdf").is_file()

    catalog = load_catalog(tmp_path)
    assert catalog.records[0].title == "Verified Paper"
    assert catalog.records[0].year == 2025
    assert catalog.records[0].doi == "10.1000/verified"
    assert catalog.records[0].work_id == paper["identity"]["work_id"]
    works = (tmp_path / "data" / "knowledge" / "works.jsonl").read_text(
        encoding="utf-8"
    )
    assert paper["identity"]["work_id"] in works

    normalized = normalize_verified_paper(tmp_path, paper_id)
    assert normalized["renamed"] is False
    assert normalized["raw_pdf"].endswith("/Verified Paper.pdf")


def test_ambiguous_candidate_only_writes_review_report(tmp_path: Path) -> None:
    paper_id = write_paper(tmp_path)
    payload = verified_payload()
    payload["results"][0]["sources"] = ["crossref"]
    payload["results"][0]["match_score"] = 0.95

    result = asyncio.run(
        enrich_paper_metadata(
            project_root=tmp_path,
            paper_id=paper_id,
            resolver=FakeResolver(payload),
        )
    )

    assert result.updated is False
    assert result.decision.status == "review_required"
    assert result.review_path is not None and result.review_path.exists()
    paper = json.loads(result.paper_json.read_text(encoding="utf-8"))
    assert paper["metadata"]["title"] == "Extracted Paper"
    assert "verification" not in paper["metadata"]
    assert (tmp_path / "data" / "raw" / paper_id / "uploaded.pdf").is_file()


def test_explicit_candidate_selection_is_recorded(tmp_path: Path) -> None:
    paper_id = write_paper(tmp_path)
    payload = verified_payload()

    result = asyncio.run(
        enrich_paper_metadata(
            project_root=tmp_path,
            paper_id=paper_id,
            resolver=FakeResolver(payload),
            selected_index=1,
        )
    )

    assert result.updated is True
    assert result.decision.status == "selected"
    paper = json.loads(result.paper_json.read_text(encoding="utf-8"))
    assert paper["metadata"]["title"] == "Different Paper"
    assert paper["metadata"]["verification"]["status"] == "selected"
    assert paper["source"]["stored_filename"] == "Different Paper.pdf"
