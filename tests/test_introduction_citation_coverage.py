from __future__ import annotations

from pathlib import Path
from typing import Any

from app.scholar.models import EvidencePack
from app.scholar.project import ScholarProjectStore
from app.scholar.writing.citation_coverage import (
    INTRODUCTION_MINIMUM_UNIQUE_PAPERS,
    citation_coverage,
)
from app.scholar.writing.models import SectionDraft, WritingContext, WritingRequest
from app.scholar.writing.service import ScholarWritingService
from tests.test_scholar_foundation import make_project


def test_same_paper_multiple_chunks_count_once() -> None:
    evidence = {
        "E1": {"paper_id": "P001"},
        "E2": {"paper_id": "P001"},
        "E3": {"paper_id": "P002"},
    }

    coverage = citation_coverage(
        ("PaperOne", "PaperTwo"),
        {"PaperOne": ("E1", "E2"), "PaperTwo": ("E3",)},
        evidence,
    )

    assert coverage.available_paper_count == 2
    assert coverage.cited_paper_count == 2
    assert coverage.cited_paper_ids == ("paper:p001", "paper:p002")


def test_default_introduction_minimum_is_five_papers() -> None:
    evidence = {f"E{i}": {"paper_id": f"P{i}"} for i in range(1, 9)}
    keys = tuple(f"Paper{i}" for i in range(1, 9))
    catalog = {key: (f"E{i}",) for i, key in enumerate(keys, start=1)}

    coverage = citation_coverage(keys, catalog, evidence)

    assert INTRODUCTION_MINIMUM_UNIQUE_PAPERS == 5
    assert coverage.minimum_unique_papers == 5
    assert coverage.sufficient is True


def test_unbound_evidence_does_not_count_as_citation_ready_source() -> None:
    evidence = {f"E{i}": {"paper_id": f"P{i}"} for i in range(1, 16)}

    coverage = citation_coverage(
        ("PaperOne",),
        {"PaperOne": ("E1",)},
        evidence,
    )

    assert coverage.available_paper_count == 1
    assert coverage.cited_paper_count == 1


def test_one_bibkey_cannot_cover_multiple_papers() -> None:
    evidence = {
        "E1": {"paper_id": "P001"},
        "E2": {"paper_id": "P002"},
    }

    coverage = citation_coverage(
        ("DuplicatedKey",),
        {"DuplicatedKey": ("E1", "E2")},
        evidence,
        minimum_unique_papers=2,
    )

    assert coverage.available_paper_count == 0
    assert coverage.cited_paper_count == 0
    assert coverage.unresolved_citation_keys == ("DuplicatedKey",)
    assert coverage.sufficient is False


class _Capability:
    workspace_id = "default"
    scope_version = 1


class _FixedDelegate:
    def __init__(self, packs: tuple[EvidencePack, ...]) -> None:
        self.packs = packs

    def research(self, request: WritingRequest, needs: Any) -> tuple[EvidencePack, ...]:
        return self.packs


class _CoverageWriter:
    def generate(self, context: WritingContext) -> SectionDraft:
        evidence_ids = tuple(
            str(item["evidence_id"])
            for pack in context.evidence_packs
            for item in pack.evidence
            if item.get("evidence_id")
        )
        citation_keys = tuple(context.citation_catalog)
        claim_ids = tuple(
            claim.claim_id
            for claim in context.claim_plan.claims
            if claim.evidence_ids
        )
        return SectionDraft(
            target_section="introduction",
            base_hash=context.current_hash,
            content="Prior work establishes the relevant research context.\n",
            claim_ids=claim_ids,
            evidence_ids=evidence_ids,
            citation_keys=citation_keys,
            original_content=context.current_introduction,
            change_summary="Add a citation-complete prior-work introduction.",
        )


def _evidence(index: int, *, paper_id: str | None = None, bibkey: str | None = None) -> dict[str, Any]:
    title = f"Verified prior-work paper {index}"
    # Keep the family name alphabetic because BibKeyGenerator derives the
    # project-local key from the final author token.
    authors = (f"Author{index} Smith",)
    value: dict[str, Any] = {
        "evidence_id": f"E{index}",
        "paper_id": paper_id or f"P{index:03d}",
        "document_id": f"D{index:03d}",
        "chunk_id": f"C{index:03d}",
        "content": f"Canonical evidence content for paper {index}.",
        "title": title,
        "authors": authors,
        "year": 2020 + index,
        "canonical_id": f"doi:10.1000/verified-paper-{index}",
    }
    if bibkey is not None:
        value["metadata"] = {"bibkey": bibkey}
    return value


def _packs(values: tuple[dict[str, Any], ...], *, candidate_count: int | None = None) -> tuple[EvidencePack, ...]:
    metadata = {"paper_candidate_count": candidate_count} if candidate_count is not None else {}
    evidence_ids = tuple(str(value["evidence_id"]) for value in values)
    return (
        EvidencePack(
            request_id="REQ:prior_work",
            query="prior work",
            claims=({
                "claim_id": "source-claim",
                "status": "supported",
                "evidence_ids": evidence_ids,
            },),
            evidence=values,
            local_coverage=1.0,
            coverage=1.0,
            metadata=metadata,
        ),
        EvidencePack(request_id="REQ:limitations", query="limitations"),
    )


def _service(
    tmp_path: Path,
    values: tuple[dict[str, Any], ...],
    *,
    candidate_count: int | None = None,
) -> tuple[ScholarWritingService, ScholarProjectStore]:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    service = ScholarWritingService(
        root,
        _Capability(),  # type: ignore[arg-type]
        _CoverageWriter(),
        project_store=store,
        delegate=_FixedDelegate(_packs(values, candidate_count=candidate_count)),  # type: ignore[arg-type]
    )
    return service, store


def _request(store: ScholarProjectStore, request_id: str = "INTRO-COVERAGE") -> WritingRequest:
    return WritingRequest(
        request_id,
        store.project_id,
        "Revise the prior work in the introduction.",
        focus="ephemeris correction",
        metadata={"minimum_unique_papers": 15},
    )


def test_three_papers_fail_before_writer_and_patch_registration(tmp_path: Path) -> None:
    service, store = _service(
        tmp_path,
        tuple(_evidence(index) for index in range(1, 4)),
        candidate_count=20,
    )

    result = service.write_introduction(_request(store))

    assert result.status == "FAILED"
    assert result.error_codes == ("INSUFFICIENT_INTRODUCTION_SOURCES",)
    assert result.patch is None
    assert result.metadata["citation_coverage"]["available_paper_count"] == 3
    assert result.metadata["citation_coverage"]["missing_paper_count"] == 12
    assert result.warnings and "已召回 20 篇候选" in result.warnings[0]


def test_fifteen_distinct_papers_and_keys_can_register_patch(tmp_path: Path) -> None:
    service, store = _service(
        tmp_path,
        tuple(_evidence(index) for index in range(1, 16)),
    )

    result = service.write_introduction(_request(store, "INTRO-15"))

    assert result.status == "READY"
    assert result.patch is not None
    assert result.metadata["citation_coverage"]["available_paper_count"] == 15
    assert result.metadata["citation_coverage"]["cited_paper_count"] == 15
    assert len(result.patch.citation_keys) == 15


def test_fifteen_evidence_items_with_one_paper_identity_fail(tmp_path: Path) -> None:
    service, store = _service(
        tmp_path,
        tuple(_evidence(index, paper_id="P-SAME") for index in range(1, 16)),
    )

    result = service.write_introduction(_request(store, "INTRO-DUPLICATE-PAPER"))

    assert result.status == "FAILED"
    assert result.error_codes == ("INSUFFICIENT_INTRODUCTION_SOURCES",)
    assert result.patch is None
    assert result.metadata["citation_coverage"]["available_paper_count"] == 1


def test_duplicate_bibkey_does_not_register_patch(tmp_path: Path) -> None:
    values = tuple(
        _evidence(index, bibkey="SameBibKey")
        for index in range(1, 16)
    )
    # Remove strong bibliographic identities so the explicit legacy key is
    # the only catalog route; the coverage function must reject its ambiguity.
    for value in values:
        value.pop("canonical_id", None)
        value.pop("title", None)
        value.pop("authors", None)
        value.pop("year", None)
    service, store = _service(tmp_path, values)

    result = service.write_introduction(_request(store, "INTRO-DUPLICATE-KEY"))

    assert result.status == "FAILED"
    assert result.error_codes == ("INSUFFICIENT_INTRODUCTION_SOURCES",)
    assert result.patch is None
    assert result.metadata["citation_coverage"]["available_paper_count"] == 0
