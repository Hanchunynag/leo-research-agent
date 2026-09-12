from __future__ import annotations

from pathlib import Path
import hashlib

from app.scholar import DraftPatch, ScholarProjectStore
from app.scholar.approval import PatchApprovalRequest, PatchApprovalService
from app.scholar.citation import (
    BibliographySynchronizer,
    CitationIdentity,
    CitationResolutionService,
)
from app.scholar.writing import IntroductionReviewer, SectionDraft
from app.scholar.writing.models import ClaimPlan, PlannedClaim
from tests.test_scholar_foundation import make_project


def evidence(*, evidence_id: str = "E1", doi: str = "10.1000/ABC") -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "source_type": "WEB_LITERATURE",
        "canonical_id": f"doi:{doi.casefold()}",
        "source_locator": f"https://doi.org/{doi}",
        "locator_type": "FULLTEXT_SPAN",
        "content": "Ephemeris errors affect positioning accuracy.",
        "content_hash": hashlib.sha256(b"Ephemeris errors affect positioning accuracy.").hexdigest(),
        "verification_method": "external_source_resolver",
        "validation_status": "provider_payload_verified",
        "title": "Ephemeris Effects on Positioning",
        "authors": ["Smith, Alice"],
        "publication_date": "2025-01-01",
        "venue": "Journal of Navigation",
        "doi": doi,
        "provider": "crossref",
    }


def setup(tmp_path: Path) -> tuple[Path, ScholarProjectStore, CitationResolutionService, PatchApprovalService]:
    root = make_project(tmp_path)
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\bibliography{references}\n\\input{sections/introduction}\n",
        encoding="utf-8",
    )
    store = ScholarProjectStore(root)
    bibliography = BibliographySynchronizer(root)
    citations = CitationResolutionService(root, project_store=store, synchronizer=bibliography)
    approval = PatchApprovalService(root, project_store=store, bibliography=bibliography)
    return root, store, citations, approval


def accept(approval: PatchApprovalService, store: ScholarProjectStore, patch: DraftPatch):
    return approval.approve(
        PatchApprovalRequest(
            patch_id=patch.patch_id,
            project_id=store.project_id,
            decision="ACCEPT",
            expected_base_hash=patch.base_hash,
            actor="human:test",
        )
    )


def test_identity_prefers_doi_and_arxiv_is_versionless() -> None:
    left = CitationIdentity(doi="https://doi.org/10.1000/ABC", title="Same", authors=("Smith",), publication_date="2025-01-01")  # type: ignore[arg-type]
    right = CitationIdentity(doi="10.1000/abc", title="Different", authors=("Other",), publication_date="2024-01-01")  # type: ignore[arg-type]
    assert left.key == right.key == "doi:10.1000/abc"

    from app.scholar.citation.models import CitationIdentity as Identity

    assert Identity(arxiv_id="2401.00001v2", title="A", authors=("Smith",), publication_date="2024-01-01").key == "arxiv:2401.00001"


def test_new_verified_evidence_proposes_and_applies_bib_entry_with_patch(tmp_path: Path) -> None:
    root, store, citations, approval = setup(tmp_path)
    result = citations.resolve(evidence())
    assert result.status == "PROPOSED_NEW_ENTRY"
    assert result.candidate is not None
    assert result.binding is not None
    assert result.snapshot is not None

    change = citations.change(result)
    assert change is not None
    state = approval.synchronizer.scan()
    original = approval.synchronizer.read_section(state, "introduction")
    patch = DraftPatch(
        patch_id="PATCH_CITATION",
        target_section="introduction",
        base_hash=state.sections["introduction"].content_hash,
        original_content=original,
        proposed_content=f"Supported claim \\cite{{{result.bibkey}}}.\n",
        project_id=store.project_id,
        citation_keys=(result.bibkey or "",),
        citation_bindings=(result.binding,),
        bibliography_changes=(change,),
        bibliography_base_hash=result.snapshot.content_hash,
    )
    preview = approval.register_patch(patch)
    assert preview.patch.bibliography_changes == (change,)

    applied = accept(approval, store, patch)
    assert applied.status == "APPLIED"
    assert applied.apply_result is not None and applied.apply_result.bibliography_applied
    bib = (root / "references.bib").read_text(encoding="utf-8")
    assert f"{{{result.bibkey}," in bib
    assert f"\\cite{{{result.bibkey}}}" in (root / "sections" / "introduction.tex").read_text(encoding="utf-8")
    assert store.list_citation_bindings()[0].status == "RESOLVED_EXISTING"
    assert store.list_external_evidence_projections()[0]["evidence_id"] == "E1"


def test_user_custom_bibkey_wins_when_added_before_accept(tmp_path: Path) -> None:
    root, store, citations, approval = setup(tmp_path)
    result = citations.resolve(evidence())
    assert result.binding is not None and result.candidate is not None and result.snapshot is not None
    change = citations.change(result)
    assert change is not None
    (root / "references.bib").write_text(
        "@article{UserKey,\n"
        " author={Smith, Alice},\n"
        " title={Ephemeris Effects on Positioning},\n"
        " year={2025},\n"
        " doi={10.1000/abc}\n}\n",
        encoding="utf-8",
    )
    state = approval.synchronizer.scan()
    patch = DraftPatch(
        patch_id="PATCH_CUSTOM_KEY",
        target_section="introduction",
        base_hash=state.sections["introduction"].content_hash,
        original_content=approval.synchronizer.read_section(state, "introduction"),
        proposed_content=f"Supported claim \\cite{{{result.bibkey}}}.\n",
        project_id=store.project_id,
        citation_keys=(result.bibkey or "",),
        citation_bindings=(result.binding,),
        bibliography_changes=(change,),
        bibliography_base_hash=result.snapshot.content_hash,
    )
    approval.register_patch(patch)
    applied = accept(approval, store, patch)
    assert applied.status == "APPLIED"
    assert not applied.apply_result.bibliography_applied  # type: ignore[union-attr]
    assert "\\cite{UserKey}" in (root / "sections" / "introduction.tex").read_text(encoding="utf-8")
    assert (root / "references.bib").read_text(encoding="utf-8").count("@article") == 1
    assert store.list_citation_bindings()[0].bibkey == "UserKey"


def test_unverified_candidate_cannot_resolve_citation(tmp_path: Path) -> None:
    root, store, citations, _ = setup(tmp_path)
    value = evidence()
    value["validation_status"] = "candidate"
    result = citations.resolve(value)
    assert result.status == "INVALID"
    assert not (root / "references.bib").exists()


def test_bibliography_first_order_reports_partial_apply_without_overwriting_tex(tmp_path: Path) -> None:
    root, store, citations, approval = setup(tmp_path)
    result = citations.resolve(evidence())
    assert result.binding is not None and result.snapshot is not None
    change = citations.change(result)
    assert change is not None
    state = approval.synchronizer.scan()
    target = root / "sections" / "introduction.tex"
    patch = DraftPatch(
        patch_id="PATCH_PARTIAL_BIB",
        target_section="introduction",
        base_hash=state.sections["introduction"].content_hash,
        original_content=target.read_text(encoding="utf-8"),
        proposed_content=f"Supported claim \\cite{{{result.bibkey}}}.\n",
        project_id=store.project_id,
        citation_keys=(result.bibkey or "",),
        citation_bindings=(result.binding,),
        bibliography_changes=(change,),
        bibliography_base_hash=result.snapshot.content_hash,
    )
    approval.register_patch(patch)
    target.write_text("User manuscript edit.\n", encoding="utf-8")
    applied = accept(approval, store, patch)
    assert applied.status == "FAILED"
    assert applied.error_code == "PARTIAL_APPLY"
    assert target.read_text(encoding="utf-8") == "User manuscript edit.\n"
    assert (root / "references.bib").is_file()


def test_bibliography_sync_detects_user_rename_delete_and_duplicate_identity(tmp_path: Path) -> None:
    root, store, citations, _ = setup(tmp_path)
    bib = root / "references.bib"
    bib.write_text(
        "@article{OldKey, author={Smith, Alice}, title={Ephemeris Effects on Positioning}, year={2025}, doi={10.1000/abc}}\n",
        encoding="utf-8",
    )
    first = citations.resolve(evidence())
    assert first.status == "RESOLVED_EXISTING" and first.bibkey == "OldKey"
    bib.write_text(
        "@article{RenamedKey, author={Smith, Alice}, title={Ephemeris Effects on Positioning}, year={2025}, doi={10.1000/abc}}\n",
        encoding="utf-8",
    )
    assert citations.sync().bibkeys == ("RenamedKey",)
    assert store.list_citation_bindings()[0].bibkey == "RenamedKey"
    bib.write_text("", encoding="utf-8")
    citations.sync()
    assert store.list_citation_bindings()[0].status == "STALE"
    bib.write_text(
        "@article{A, author={Smith, Alice}, title={Ephemeris Effects on Positioning}, year={2025}, doi={10.1000/abc}}\n"
        "@article{B, author={Smith, Alice}, title={Ephemeris Effects on Positioning}, year={2025}, doi={10.1000/abc}}\n",
        encoding="utf-8",
    )
    assert citations.resolve(evidence()).status == "BIB_CONFLICT"


def test_reviewer_rejects_binding_for_a_different_identity() -> None:
    from app.scholar.citation import CitationBinding

    wrong = CitationBinding(
        binding_id="BIND_WRONG",
        project_id="PROJECT",
        citation_identity=CitationIdentity(doi="10.1000/other", title="Other", authors=("Other",), publication_date="2025-01-01"),  # type: ignore[arg-type]
        bibkey="UserKey",
        status="RESOLVED_EXISTING",
    )
    claim = PlannedClaim("C1", "background", "the claim", "LITERATURE", evidence_ids=("E1",))
    draft = SectionDraft("introduction", "hash", "the claim \\cite{UserKey}", claim_ids=("C1",), evidence_ids=("E1",), citation_keys=("UserKey",))
    report = IntroductionReviewer().review(
        draft,
        ClaimPlan("R", (claim,), ()),
        {"E1": evidence()},
        (),
        (),
        {"UserKey": ("E1",)},
        citation_bindings={"UserKey": wrong},
    )
    assert not report.valid
    assert any(issue.code == "CITATION_WRONG_IDENTITY" and issue.severity == "BLOCKER" for issue in report.issues)
