from __future__ import annotations

import json
from pathlib import Path

from app.knowledge.identity import (
    MAX_FILENAME_BYTES,
    build_identity,
    canonical_pdf_filename,
)
from app.knowledge.works import rebuild_work_catalog


def verified_metadata(doi: str) -> dict[str, object]:
    return {
        "title": "A Verified Paper",
        "authors": ["Ada Lovelace"],
        "year": 2025,
        "doi": doi,
        "verification": {"status": "verified"},
    }


def test_same_doi_groups_different_pdf_documents_into_one_work() -> None:
    first = build_identity(
        paper_id="P_111111111111",
        sha256="1" * 64,
        metadata=verified_metadata("10.1000/LEO"),
    )
    second = build_identity(
        paper_id="P_222222222222",
        sha256="2" * 64,
        metadata=verified_metadata("https://doi.org/10.1000/leo"),
    )

    assert first["document_id"] != second["document_id"]
    assert first["work_id"] == second["work_id"]
    assert first["work_id_method"] == "doi"


def test_unverified_metadata_has_no_work_id() -> None:
    identity = build_identity(
        paper_id="P_111111111111",
        sha256="1" * 64,
        metadata={"title": "Unverified"},
    )

    assert identity["document_id"] == "D_111111111111"
    assert identity["work_id"] is None
    assert identity["status"] == "unresolved"


def test_canonical_pdf_filename_is_safe_readable_and_bounded() -> None:
    filename = canonical_pdf_filename('  Modeling: LEO/Signals? <A Study>  "Final"  ')
    assert filename == "Modeling LEO Signals A Study Final.pdf"
    assert not any(character in filename for character in '<>:"/\\|?*')

    long_filename = canonical_pdf_filename("低轨卫星研究" * 100)
    assert long_filename.endswith(".pdf")
    assert len(long_filename.encode("utf-8")) <= MAX_FILENAME_BYTES


def test_work_catalog_groups_two_documents_with_same_work_id(
    tmp_path: Path,
) -> None:
    identities = [
        build_identity(
            paper_id=f"P_{digit * 12}",
            sha256=digit * 64,
            metadata=verified_metadata("10.1000/leo"),
        )
        for digit in ("1", "2")
    ]
    for digit, identity in zip(("1", "2"), identities, strict=True):
        paper_id = f"P_{digit * 12}"
        canonical = tmp_path / "data" / "canonical" / paper_id / "paper.json"
        canonical.parent.mkdir(parents=True)
        canonical.write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "identity": identity,
                    "metadata": {
                        **verified_metadata("10.1000/leo"),
                        "abstract": "Evidence.",
                    },
                    "pipeline": {"created_at": f"2025-01-0{digit}T00:00:00+00:00"},
                }
            ),
            encoding="utf-8",
        )

    result = rebuild_work_catalog(tmp_path)

    assert len(result.records) == 1
    assert result.records[0].work_id == identities[0]["work_id"]
    assert len(result.records[0].document_ids) == 2
    assert result.records[0].paper_ids == [
        "P_111111111111",
        "P_222222222222",
    ]
