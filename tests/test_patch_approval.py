from __future__ import annotations

from pathlib import Path

from app.scholar import DraftPatch, ManuscriptSynchronizer, ReviewIssue, ReviewReport
from app.scholar.approval import (
    PatchApprovalRequest,
    PatchApprovalService,
)
from app.scholar.project import ScholarProjectStore
from tests.test_scholar_foundation import make_project


def make_service(tmp_path: Path) -> tuple[PatchApprovalService, ScholarProjectStore, Path, DraftPatch]:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    service = PatchApprovalService(root, project_store=store)
    state = service.synchronizer.scan()
    original = service.synchronizer.read_section(state, "introduction")
    patch = DraftPatch(
        patch_id="PATCH_TEST",
        target_section="introduction",
        base_hash=state.sections["introduction"].content_hash,
        original_content=original,
        proposed_content="Patched introduction.\n",
        project_id=store.project_id,
    )
    return service, store, root, patch


def accept(service: PatchApprovalService, store: ScholarProjectStore, patch: DraftPatch):
    return service.approve(
        PatchApprovalRequest(
            patch_id=patch.patch_id,
            project_id=store.project_id,
            decision="ACCEPT",
            expected_base_hash=patch.base_hash,
            actor="human:test",
        )
    )


def test_patch_lifecycle_preview_apply_and_idempotency(tmp_path: Path) -> None:
    service, store, root, patch = make_service(tmp_path)

    preview = service.register_patch(patch)
    assert preview.status == "AWAITING_APPROVAL"
    assert preview.original_content == "Original introduction.\n"
    assert service.get_preview(patch.patch_id).patch.proposed_content == "Patched introduction.\n"

    result = accept(service, store, patch)
    assert result.status == "APPLIED"
    assert result.apply_result is not None
    assert result.apply_result.new_hash != patch.base_hash
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") == "Patched introduction.\n"
    assert store.get_manuscript_state_projection()["version"] == 2  # type: ignore[index]

    replay = accept(service, store, patch)
    assert replay.apply_result is not None and replay.apply_result.idempotent is True
    assert len(store.list_patch_audits(patch.patch_id)) == 1

    next_state = service.synchronizer.scan()
    next_patch = DraftPatch(
        patch_id="PATCH_TEST_2",
        target_section="introduction",
        base_hash=next_state.sections["introduction"].content_hash,
        original_content="Patched introduction.\n",
        proposed_content="Patched introduction again.\n",
        project_id=store.project_id,
    )
    service.register_patch(next_patch)
    accept(service, store, next_patch)
    projection = store.get_manuscript_state_projection()
    assert projection["version"] == 3  # type: ignore[index]
    introduction = next(
        value for value in projection["sections"]  # type: ignore[index]
        if value["name"] == "introduction"
    )
    assert introduction["version"] == 3


def test_reject_is_terminal(tmp_path: Path) -> None:
    service, store, _, patch = make_service(tmp_path)
    service.register_patch(patch)
    rejected = service.reject(
        PatchApprovalRequest(patch.patch_id, store.project_id, "REJECT", patch.base_hash, "human:test")
    )
    assert rejected.status == "REJECTED"
    try:
        accept(service, store, patch)
    except Exception as error:
        assert "不允许 Accept" in str(error)
    else:
        raise AssertionError("rejected Patch must not be accepted")


def test_review_gate_blocks_high_issue(tmp_path: Path) -> None:
    service, store, root, patch = make_service(tmp_path)
    report = ReviewReport(
        valid=False,
        issues=(ReviewIssue("UNSUPPORTED", "HIGH", "unsupported claim"),),
        report_id="REVIEW_TEST",
    )
    service.register_patch(patch, review_report=report)
    result = accept(service, store, patch)
    assert result.error_code == "REVIEW_GATE_BLOCKED"
    assert result.status == "AWAITING_APPROVAL"
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") == patch.original_content


def test_manual_edit_causes_conflict_and_preserves_user_file(tmp_path: Path) -> None:
    service, store, root, patch = make_service(tmp_path)
    service.register_patch(patch)
    target = root / "sections" / "introduction.tex"
    target.write_text("User edit before approval.\n", encoding="utf-8")

    result = accept(service, store, patch)
    assert result.status == "CONFLICT"
    assert result.error_code == "PATCH_CONFLICT"
    assert target.read_text(encoding="utf-8") == "User edit before approval.\n"
    assert service.get_preview(patch.patch_id).status == "CONFLICT"


def test_original_content_guard_is_checked_even_when_hash_is_current(tmp_path: Path) -> None:
    service, store, _, patch = make_service(tmp_path)
    changed = DraftPatch(
        patch_id=patch.patch_id,
        target_section=patch.target_section,
        base_hash=patch.base_hash,
        original_content="not the current text",
        proposed_content=patch.proposed_content,
        project_id=patch.project_id,
    )
    service.register_patch(changed)
    result = accept(service, store, changed)
    assert result.status == "CONFLICT"
    assert result.error_code == "PATCH_CONFLICT"


def test_synchronizer_rejects_path_escape(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    synchronizer = ManuscriptSynchronizer(root)
    try:
        synchronizer._safe_path("../outside.tex")  # type: ignore[attr-defined]
    except ValueError as error:
        assert "project_root" in str(error)
    else:
        raise AssertionError("path escape must be rejected")
