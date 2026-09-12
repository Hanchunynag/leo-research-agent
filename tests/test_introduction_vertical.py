from __future__ import annotations

from pathlib import Path
from typing import Any

from app.scholar import (
    Contribution,
    ManuscriptFact,
    ScholarProjectStore,
)
from app.scholar.models import EvidencePack
from app.scholar.writing import (
    IntroductionReviewer,
    SectionDraft,
    ScholarWritingService,
    WritingRequest,
)
from app.scholar.writing.models import ClaimPlan, WritingContext
from tests.test_scholar_foundation import make_project


def evidence_pack(query: str = "prior work") -> EvidencePack:
    return EvidencePack(
        request_id="WR-1:prior_work",
        query=query,
        claims=({"claim_id": "source-claim", "status": "supported", "evidence_ids": ("E1",)},),
        evidence=(
            {
                "evidence_id": "E1",
                "paper_id": "P1",
                "document_id": "D1",
                "chunk_id": "C1",
                "text": "Prior work addresses ephemeris correction.",
                "metadata": {"bibkey": "Doe2024"},
            },
        ),
        local_coverage=1.0,
        coverage=1.0,
        web_used=False,
    )


class FakeCapability:
    workspace_id = "default"
    scope_version = 1


class FakeDelegate:
    def __init__(self, packs: tuple[EvidencePack, ...]) -> None:
        self.packs = packs
        self.calls: list[WritingRequest] = []

    def research(self, request: WritingRequest, needs: Any) -> tuple[EvidencePack, ...]:
        self.calls.append(request)
        return self.packs


class FakeWriter:
    def __init__(self, *, include_citation: bool = True) -> None:
        self.include_citation = include_citation
        self.calls = 0

    def generate(self, context: WritingContext) -> SectionDraft:
        self.calls += 1
        claim_ids = tuple(value.claim_id for value in context.claim_plan.claims if value.claim_type == "existing_approaches")
        contribution_ids = tuple(value.contribution_id for value in context.contributions)
        return SectionDraft(
            target_section="introduction",
            base_hash=context.current_hash,
            content=(
                "Original introduction.\n"
                "Prior work addresses ephemeris correction. "
                "Our confirmed direction extends the correction approach.\n"
            ),
            claim_ids=claim_ids + tuple(f"contribution:{value}" for value in contribution_ids),
            evidence_ids=("E1",),
            citation_keys=("Doe2024",) if self.include_citation else (),
            contribution_ids=contribution_ids,
            change_summary="Strengthen prior-work move while preserving background.",
            original_content=context.current_introduction,
        )


def setup(tmp_path: Path, *, include_citation: bool = True) -> tuple[ScholarWritingService, ScholarProjectStore, FakeWriter, FakeDelegate, Path]:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    store.put_fact(ManuscriptFact("F1", "sampling_rate", 100, confirmed=True))
    store.put_contribution(Contribution("C1", "A user-confirmed ephemeris correction approach.", "confirmed", True))
    writer = FakeWriter(include_citation=include_citation)
    delegate = FakeDelegate((evidence_pack(),))
    service = ScholarWritingService(
        root,
        FakeCapability(),  # type: ignore[arg-type]
        writer,
        project_store=store,
        delegate=delegate,  # type: ignore[arg-type]
    )
    return service, store, writer, delegate, root


def request(store: ScholarProjectStore, instruction: str = "Revise the existing prior work in the introduction.") -> WritingRequest:
    return WritingRequest("WR-1", store.project_id, instruction, focus="ephemeris correction prior work")


def test_introduction_vertical_returns_unapplied_patch_with_latest_hash(tmp_path: Path) -> None:
    service, store, writer, delegate, root = setup(tmp_path)
    original = (root / "sections" / "introduction.tex").read_text(encoding="utf-8")

    result = service.write_introduction(request(store))

    assert result.status == "READY"
    assert result.patch is not None
    assert result.patch.base_hash == result.manuscript_hash
    assert result.patch.review_report_id == result.review_report.report_id
    assert result.patch.used_evidence_ids == ("E1",)
    assert result.patch.citation_keys == ("Doe2024",)
    assert result.claim_plan is not None
    assert result.review_report is not None and result.review_report.valid is True
    assert delegate.calls and writer.calls == 1
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") == original
    assert service.capabilities.allows("manuscript.write") is False


def test_repeating_same_writing_request_reuses_immutable_patch_id(tmp_path: Path) -> None:
    service, store, _, _, _ = setup(tmp_path)

    first = service.write_introduction(request(store))
    second = service.write_introduction(request(store))

    assert first.patch is not None and second.patch is not None
    assert second.patch.patch_id == first.patch.patch_id
    assert second.patch.proposed_content == first.patch.proposed_content


def test_unconfirmed_contribution_is_not_used_as_writing_authority(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    store.put_contribution(Contribution("CAND", "Unconfirmed idea"))
    service = ScholarWritingService(root, FakeCapability(), FakeWriter(), project_store=store, delegate=FakeDelegate((evidence_pack(),)))  # type: ignore[arg-type]

    result = service.write_introduction(WritingRequest("WR-2", store.project_id, "Revise prior work", selected_contribution_ids=("CAND",)))

    assert result.status == "CONFLICT"
    assert result.error_codes == ("CONTRIBUTION_CONFLICT",)
    assert result.patch is None


def test_missing_citation_becomes_high_review_issue_without_fabrication(tmp_path: Path) -> None:
    service, store, _, _, _ = setup(tmp_path, include_citation=False)

    result = service.write_introduction(request(store))

    assert result.status == "NEEDS_USER_REVIEW"
    assert result.patch is not None
    assert result.patch.reviewer_status == "needs_user_review"
    assert "Doe2024" not in result.patch.citation_keys
    assert any(issue.code == "CITATION_REQUIRED" and issue.severity == "HIGH" for issue in result.review_report.issues)


def test_reviewer_rejects_unknown_evidence_and_contribution_ids() -> None:
    reviewer = IntroductionReviewer()
    plan = ClaimPlan("WR", (), ())
    draft = SectionDraft(
        "introduction",
        "hash",
        "Draft.",
        evidence_ids=("FAKE-EVIDENCE",),
        contribution_ids=("FAKE-CONTRIBUTION",),
    )

    report = reviewer.review(draft, plan, {}, (), (), {})

    assert report.valid is False
    assert {issue.code for issue in report.issues} == {"FABRICATED_EVIDENCE_ID", "INVENTED_CONTRIBUTION"}


def test_stale_manuscript_during_generation_returns_conflict_and_does_not_apply(tmp_path: Path) -> None:
    service, store, _, _, root = setup(tmp_path)
    original_writer = service.writer

    class MutatingWriter(FakeWriter):
        def generate(self, context: WritingContext) -> SectionDraft:
            value = original_writer.generate(context)  # type: ignore[attr-defined]
            (root / "sections" / "introduction.tex").write_text("User changed it.\n", encoding="utf-8")
            return value

    service.writer = MutatingWriter()
    result = service.write_introduction(request(store))

    assert result.status == "CONFLICT"
    assert result.error_codes == ("STALE_BASE_HASH",)
    assert result.patch is None
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") == "User changed it.\n"
