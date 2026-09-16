"""Human-only Patch approval and safe manuscript Apply service."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import re
from typing import Any

from app.scholar.citation import (
    BibliographySynchronizer,
    BibStaleBaseHash,
    BibKeyCollision,
    CitationLifecycleError,
    candidate_identity,
)
from app.scholar.citation.models import BibliographyChange, CitationBinding
from app.scholar.manuscript import ManuscriptSynchronizer, PatchConflict
from app.scholar.models import DraftPatch, ReviewReport
from app.scholar.project import ScholarProjectStore
from app.scholar.approval.models import (
    PatchApprovalRequest,
    PatchApprovalResult,
    PatchApplyResult,
    PatchPreview,
    StoredPatch,
)


class PatchApprovalError(RuntimeError):
    """Patch approval/application domain error with a stable client code."""

    code = "PATCH_APPROVAL_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class PatchNotFoundError(PatchApprovalError):
    code = "PATCH_NOT_FOUND"


class PatchNotApprovableError(PatchApprovalError):
    code = "PATCH_NOT_APPROVABLE"


class PatchConflictError(PatchApprovalError):
    code = "PATCH_CONFLICT"


class ReviewGateError(PatchApprovalError):
    code = "REVIEW_GATE_BLOCKED"


class PatchApprovalService:
    """唯一负责人工决定、Review Gate、Hash Guard 和 Apply 的服务。

    Agent/Writer 只能调用 :meth:`register_patch`。没有任何公开的
    ``force``、``skip_approval`` 或客户端 supplied content 参数。
    """

    def __init__(
        self,
        project_root: Path,
        *,
        project_store: ScholarProjectStore | None = None,
        synchronizer: ManuscriptSynchronizer | None = None,
        bibliography: BibliographySynchronizer | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.synchronizer = synchronizer or ManuscriptSynchronizer(self.project_root)
        self.bibliography = bibliography or BibliographySynchronizer(self.project_root)

    def register_patch(
        self,
        patch: DraftPatch,
        *,
        review_report: ReviewReport | None = None,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
    ) -> PatchPreview:
        """Persist one immutable proposal and return its preview.

        Re-registering the exact same proposal is idempotent. A different
        proposal with the same ``patch_id`` is rejected by ProjectStore.
        """

        try:
            record = self.project_store.save_patch(
                patch,
                review_report=review_report,
                source_session_id=source_session_id,
                source_run_id=source_run_id,
            )
        except KeyError as error:
            raise PatchNotFoundError(str(error)) from error
        return self._record(record).preview()

    def get_preview(self, patch_id: str, *, project_id: str | None = None) -> PatchPreview:
        record = self._get_record(patch_id)
        self._check_project(record, project_id)
        return record.preview()

    def approve(self, request: PatchApprovalRequest) -> PatchApprovalResult:
        record = self._get_record(request.patch_id)
        self._check_project(record, request.project_id)

        if request.decision == "REJECT":
            return self._reject(record, request)

        if record.status == "APPLIED":
            return self._idempotent_applied(record, request)
        if record.status != "AWAITING_APPROVAL":
            raise PatchNotApprovableError(
                f"Patch {request.patch_id} 当前状态 {record.status} 不允许 Accept。"
            )

        if request.expected_base_hash != record.patch.base_hash:
            return self._conflict(
                record,
                request,
                previous_hash=None,
                message="expected_base_hash 与 immutable DraftPatch.base_hash 不一致。",
            )

        blocking = self._blocking_review_issues(record.review_report)
        if blocking:
            audit_id = self.project_store.record_patch_audit(
                request.patch_id,
                decision="ACCEPT_BLOCKED",
                actor=request.actor,
                decision_at=request.timestamp,
                expected_base_hash=request.expected_base_hash,
                details={"issues": blocking},
            )
            return PatchApprovalResult(
                patch_id=request.patch_id,
                project_id=request.project_id,
                decision="ACCEPT",
                status="AWAITING_APPROVAL",
                actor=request.actor,
                audit_id=audit_id,
                error_code=ReviewGateError.code,
                message="Review Report 含 BLOCKER 或 HIGH，Patch 不能 Apply。",
            )

        self.project_store.transition_patch(
            request.patch_id,
            "APPROVED",
            expected_status="AWAITING_APPROVAL",
        )
        self.project_store.transition_patch(
            request.patch_id,
            "APPLYING",
            expected_status="APPROVED",
        )
        return self._apply(record, request)

    def reject(self, request: PatchApprovalRequest) -> PatchApprovalResult:
        if request.decision != "REJECT":
            raise ValueError("reject() 只接受 decision='REJECT'。")
        return self.approve(request)

    def audits(self, patch_id: str) -> list[dict[str, object]]:
        self._get_record(patch_id)
        return self.project_store.list_patch_audits(patch_id)

    def _apply(
        self,
        record: StoredPatch,
        request: PatchApprovalRequest,
    ) -> PatchApprovalResult:
        state = self._current_state_with_project_versions()
        patch = record.patch
        bibliography_snapshot = None
        bibliography_previous_hash: str | None = None
        bibliography_new_hash: str | None = None
        bibliography_applied = False
        citation_replacements: dict[str, str] = {}
        changes = tuple(value for value in patch.bibliography_changes if isinstance(value, BibliographyChange))
        if changes or patch.citation_bindings:
            try:
                bibliography_snapshot = self.bibliography.sync()
                bibliography_previous_hash = bibliography_snapshot.content_hash
                citation_replacements = self._reconcile_bibliography(
                    patch,
                    bibliography_snapshot,
                    changes,
                )
            except (BibStaleBaseHash, BibKeyCollision, CitationLifecycleError, ValueError, OSError) as error:
                return self._bibliography_conflict(record, request, str(error), getattr(error, "code", "BIB_CONFLICT"))
            if changes and patch.bibliography_base_hash is None:
                return self._bibliography_conflict(record, request, "BIB_STALE_BASE_HASH: Patch 缺少 bibliography_base_hash。", "BIB_STALE_BASE_HASH")
            # A changed bibliography is acceptable only when every proposal
            # was already satisfied by the user's current entry.  Otherwise
            # additions must be based on the exact hash captured at preview.
            if changes and bibliography_snapshot.content_hash != patch.bibliography_base_hash:
                unresolved_changes = tuple(
                    change for change in changes
                    if not self.bibliography.find_matches(
                        bibliography_snapshot,
                        candidate_identity(change.entry),
                    )
                )
                if unresolved_changes:
                    return self._bibliography_conflict(
                        record,
                        request,
                        "BIB_STALE_BASE_HASH: references.bib 在审批前发生了未解决的修改。",
                        "BIB_STALE_BASE_HASH",
                    )
                changes = ()
        if changes:
            try:
                assert bibliography_snapshot is not None
                after_bib = self.bibliography.write_changes(bibliography_snapshot, changes)
                bibliography_new_hash = after_bib.content_hash
                bibliography_applied = after_bib.content_hash != bibliography_snapshot.content_hash
            except (BibStaleBaseHash, BibKeyCollision, CitationLifecycleError, ValueError, OSError) as error:
                return self._bibliography_conflict(record, request, str(error), getattr(error, "code", "BIB_CONFLICT"))
        elif bibliography_snapshot is not None:
            bibliography_new_hash = bibliography_snapshot.content_hash
        section = state.sections.get(record.patch.target_section)
        previous_hash = section.content_hash if section is not None else None
        if section is None or previous_hash != record.patch.base_hash:
            if bibliography_applied:
                return self._partial_apply(
                    record,
                    request,
                    previous_hash=previous_hash,
                    message="Bibliography 已应用，但 Manuscript 在人工确认后发生变化；未覆盖用户文件。",
                    bibliography_previous_hash=bibliography_previous_hash,
                    bibliography_new_hash=bibliography_new_hash,
                )
            return self._conflict(
                record,
                request,
                previous_hash=previous_hash,
                message="Manuscript 在人工确认后发生变化，Patch 未写入。",
                expected_status="APPLYING",
            )
        effective_patch = self._rewrite_patch_citations(patch, citation_replacements)
        try:
            new_state = self.synchronizer.apply_patch(state, effective_patch)
        except PatchConflict as error:
            if bibliography_applied:
                return self._partial_apply(
                    record,
                    request,
                    previous_hash=previous_hash,
                    message=str(error),
                    bibliography_previous_hash=bibliography_previous_hash,
                    bibliography_new_hash=bibliography_new_hash,
                )
            return self._conflict(
                record,
                request,
                previous_hash=previous_hash,
                message=str(error),
                expected_status="APPLYING",
            )
        except (OSError, ValueError, KeyError) as error:
            error_code = "PARTIAL_APPLY" if bibliography_applied else "PATCH_APPLY_FAILED"
            self.project_store.transition_patch(
                request.patch_id,
                "FAILED",
                expected_status="APPLYING",
                result={"error_code": error_code, "message": str(error), "bibliography_applied": bibliography_applied},
            )
            audit_id = self.project_store.record_patch_audit(
                request.patch_id,
                decision="ACCEPT",
                actor=request.actor,
                decision_at=request.timestamp,
                expected_base_hash=request.expected_base_hash,
                previous_hash=previous_hash,
                details={"error_code": error_code, "message": str(error), "bibliography_applied": bibliography_applied},
            )
            return PatchApprovalResult(
                patch_id=request.patch_id,
                project_id=request.project_id,
                decision="ACCEPT",
                status="FAILED",
                actor=request.actor,
                audit_id=audit_id,
                error_code=error_code,
                message=str(error),
                apply_result=PatchApplyResult(
                    patch_id=request.patch_id,
                    project_id=request.project_id,
                    status="FAILED",
                    error_code=error_code,
                    message=str(error),
                    bibliography_previous_hash=bibliography_previous_hash,
                    bibliography_new_hash=bibliography_new_hash,
                    bibliography_applied=bibliography_applied,
                ),
            )

        try:
            self.project_store.save_manuscript_state(new_state)
        except (OSError, ValueError, RuntimeError) as error:
            if bibliography_applied:
                return self._partial_apply(
                    record,
                    request,
                    previous_hash=previous_hash,
                    message=f"Bibliography 已应用，但 Manuscript 状态投影失败：{error}",
                    bibliography_previous_hash=bibliography_previous_hash,
                    bibliography_new_hash=bibliography_new_hash,
                )
            raise
        actual_section = new_state.sections.get(record.patch.target_section)
        new_hash = actual_section.content_hash if actual_section is not None else None
        result = PatchApplyResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            status="APPLIED",
            previous_hash=previous_hash,
            new_hash=new_hash,
            manuscript_version=new_state.version,
            message="Patch 已安全应用；Build 尚未触发。",
            bibliography_previous_hash=bibliography_previous_hash,
            bibliography_new_hash=bibliography_new_hash,
            bibliography_applied=bibliography_applied,
        )
        audit_id = self.project_store.record_patch_audit(
            request.patch_id,
            decision="ACCEPT",
            actor=request.actor,
            decision_at=request.timestamp,
            expected_base_hash=request.expected_base_hash,
            previous_hash=previous_hash,
            new_hash=new_hash,
            details={
                "manuscript_version": new_state.version,
                "bibliography_previous_hash": bibliography_previous_hash,
                "bibliography_new_hash": bibliography_new_hash,
                "bibliography_applied": bibliography_applied,
            },
        )
        result = PatchApplyResult(
            patch_id=result.patch_id,
            project_id=result.project_id,
            status=result.status,
            previous_hash=result.previous_hash,
            new_hash=result.new_hash,
            audit_id=audit_id,
            error_code=result.error_code,
            message=result.message,
            manuscript_version=result.manuscript_version,
            bibliography_previous_hash=result.bibliography_previous_hash,
            bibliography_new_hash=result.bibliography_new_hash,
            bibliography_applied=result.bibliography_applied,
        )
        self._persist_resolved_bindings(record.patch, citation_replacements)
        self.project_store.transition_patch(
            request.patch_id,
            "APPLIED",
            expected_status="APPLYING",
            result=asdict(result),
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="ACCEPT",
            status="APPLIED",
            actor=request.actor,
            applied=True,
            apply_result=result,
            audit_id=audit_id,
            message=result.message,
        )

    def _reconcile_bibliography(
        self,
        patch: DraftPatch,
        snapshot: Any,
        changes: tuple[BibliographyChange, ...],
    ) -> dict[str, str]:
        """Resolve current user BibKeys and return proposal→current rewrites."""

        replacements: dict[str, str] = {}
        change_identity_keys = {change.identity_key for change in changes}
        for binding in patch.citation_bindings:
            if not isinstance(binding, CitationBinding):
                raise CitationLifecycleError("CITATION_NOT_RESOLVED: Patch 包含无效 CitationBinding。")
            if binding.project_id != patch.project_id:
                raise CitationLifecycleError("PROJECT_CONFLICT: CitationBinding 不属于当前 Project。", code="PROJECT_CONFLICT")
            if binding.status not in {"RESOLVED_EXISTING", "PROPOSED_NEW_ENTRY"} or not binding.bibkey:
                raise CitationLifecycleError("CITATION_NOT_RESOLVED: CitationBinding 尚未解析。")
            if not binding.bibkey:
                continue
            matches = self.bibliography.find_matches(snapshot, binding.citation_identity)
            if len(matches) > 1:
                raise BibKeyCollision("BIB_CONFLICT: bibliography 中存在重复文献身份。")
            if matches and matches[0].get("ID"):
                current_key = str(matches[0]["ID"])
                if current_key != binding.bibkey:
                    replacements[binding.bibkey] = current_key
            elif binding.status == "RESOLVED_EXISTING":
                raise CitationLifecycleError("CITATION_NOT_RESOLVED: 已解析文献在当前 references.bib 中不存在。", code="CITATION_NOT_RESOLVED")
            elif binding.identity_key not in change_identity_keys:
                raise CitationLifecycleError("CITATION_NOT_RESOLVED: 新 BibKey 缺少对应 BibliographyChange。")
        for change in changes:
            matches = self.bibliography.find_matches(snapshot, candidate_identity(change.entry))
            if len(matches) > 1:
                raise BibKeyCollision("BIB_CONFLICT: bibliography 中存在重复文献身份。")
            if matches:
                current_key = str(matches[0].get("ID") or "")
                if current_key and current_key != change.entry.bibkey:
                    replacements[change.entry.bibkey] = current_key
                continue
            occupied = next((entry for entry in snapshot.entries if str(entry.get("ID")) == change.entry.bibkey), None)
            if occupied is not None:
                raise BibKeyCollision(f"BIBKEY_COLLISION: BibKey 已被另一篇文献占用：{change.entry.bibkey}")
        return replacements

    @staticmethod
    def _rewrite_patch_citations(patch: DraftPatch, replacements: dict[str, str]) -> DraftPatch:
        if not replacements:
            return patch
        content = patch.proposed_content
        for old, new in replacements.items():
            content = re.sub(
                rf"(\\cite(?:[A-Za-z]*)?\s*\{{[^}}]*?)\b{re.escape(old)}\b",
                lambda match: match.group(1) + new,
                content,
            )
        return replace(patch, proposed_content=content, citation_keys=tuple(replacements.get(value, value) for value in patch.citation_keys))

    def _persist_resolved_bindings(self, patch: DraftPatch, replacements: dict[str, str]) -> None:
        for binding in patch.citation_bindings:
            if not isinstance(binding, CitationBinding) or not binding.bibkey:
                continue
            current_key = replacements.get(binding.bibkey, binding.bibkey)
            if current_key != binding.bibkey or binding.status == "PROPOSED_NEW_ENTRY":
                self.project_store.save_citation_binding(replace(binding, bibkey=current_key, status="RESOLVED_EXISTING"))

    def _bibliography_conflict(
        self,
        record: StoredPatch,
        request: PatchApprovalRequest,
        message: str,
        error_code: str,
    ) -> PatchApprovalResult:
        self.project_store.transition_patch(
            request.patch_id,
            "CONFLICT",
            expected_status="APPLYING",
            result={"error_code": error_code, "message": message},
        )
        audit_id = self.project_store.record_patch_audit(
            request.patch_id,
            decision="ACCEPT",
            actor=request.actor,
            decision_at=request.timestamp,
            expected_base_hash=request.expected_base_hash,
            details={"error_code": error_code, "message": message},
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="ACCEPT",
            status="CONFLICT",
            actor=request.actor,
            audit_id=audit_id,
            error_code=error_code,
            message=message,
        )

    def _partial_apply(
        self,
        record: StoredPatch,
        request: PatchApprovalRequest,
        *,
        previous_hash: str | None,
        message: str,
        bibliography_previous_hash: str | None,
        bibliography_new_hash: str | None,
    ) -> PatchApprovalResult:
        result = PatchApplyResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            status="FAILED",
            previous_hash=previous_hash,
            error_code="PARTIAL_APPLY",
            message=message,
            bibliography_previous_hash=bibliography_previous_hash,
            bibliography_new_hash=bibliography_new_hash,
            bibliography_applied=True,
        )
        self.project_store.transition_patch(
            request.patch_id,
            "FAILED",
            expected_status="APPLYING",
            result=asdict(result),
        )
        audit_id = self.project_store.record_patch_audit(
            request.patch_id,
            decision="ACCEPT",
            actor=request.actor,
            decision_at=request.timestamp,
            expected_base_hash=request.expected_base_hash,
            previous_hash=previous_hash,
            details={"error_code": "PARTIAL_APPLY", "message": message, "bibliography_applied": True},
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="ACCEPT",
            status="FAILED",
            actor=request.actor,
            audit_id=audit_id,
            error_code="PARTIAL_APPLY",
            message=message,
            apply_result=replace(result, audit_id=audit_id),
        )

    def _reject(
        self,
        record: StoredPatch,
        request: PatchApprovalRequest,
    ) -> PatchApprovalResult:
        if request.expected_base_hash != record.patch.base_hash:
            raise PatchConflictError(
                "expected_base_hash 与 immutable DraftPatch.base_hash 不一致。"
            )
        if record.status == "REJECTED":
            return PatchApprovalResult(
                patch_id=request.patch_id,
                project_id=request.project_id,
                decision="REJECT",
                status="REJECTED",
                actor=request.actor,
                message="Patch 已被 Reject。",
            )
        if record.status != "AWAITING_APPROVAL":
            raise PatchNotApprovableError(
                f"Patch {request.patch_id} 当前状态 {record.status} 不允许 Reject。"
            )
        self.project_store.transition_patch(
            request.patch_id,
            "REJECTED",
            expected_status="AWAITING_APPROVAL",
        )
        audit_id = self.project_store.record_patch_audit(
            request.patch_id,
            decision="REJECT",
            actor=request.actor,
            decision_at=request.timestamp,
            expected_base_hash=request.expected_base_hash,
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="REJECT",
            status="REJECTED",
            actor=request.actor,
            audit_id=audit_id,
            message="Patch 已 Reject；如需继续请重新生成 Patch。",
        )

    def _conflict(
        self,
        record: StoredPatch,
        request: PatchApprovalRequest,
        *,
        previous_hash: str | None,
        message: str,
        expected_status: str | None = None,
    ) -> PatchApprovalResult:
        self.project_store.transition_patch(
            request.patch_id,
            "CONFLICT",
            expected_status=expected_status or record.status,
            result={
                "error_code": PatchConflictError.code,
                "message": message,
                "previous_hash": previous_hash,
            },
        )
        audit_id = self.project_store.record_patch_audit(
            request.patch_id,
            decision="ACCEPT",
            actor=request.actor,
            decision_at=request.timestamp,
            expected_base_hash=request.expected_base_hash,
            previous_hash=previous_hash,
            details={"error_code": PatchConflictError.code, "message": message},
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="ACCEPT",
            status="CONFLICT",
            actor=request.actor,
            audit_id=audit_id,
            error_code=PatchConflictError.code,
            message=message,
        )

    @staticmethod
    def _blocking_review_issues(review_report: ReviewReport | None) -> list[dict[str, str | None]]:
        if review_report is None:
            return []
        return [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "claim_id": issue.claim_id,
            }
            for issue in review_report.issues
            if issue.severity in {"BLOCKER", "HIGH"}
        ]

    def _get_record(self, patch_id: str) -> StoredPatch:
        try:
            value = self.project_store.get_patch(patch_id)
        except KeyError as error:
            raise PatchNotFoundError(str(error)) from error
        return self._record(value)

    def _current_state_with_project_versions(self):
        """Use the filesystem as content truth and DB only as version projection."""

        state = self.synchronizer.scan()
        projection = self.project_store.get_manuscript_state_projection()
        if projection is None:
            return state
        versions = {
            str(value["name"]): int(value["version"])
            for value in projection["sections"]  # type: ignore[index]
        }
        sections = {
            name: replace(section, version=versions.get(name, section.version))
            for name, section in state.sections.items()
        }
        return replace(state, version=int(projection["version"]), sections=sections)  # type: ignore[index]

    @staticmethod
    def _record(value: object) -> StoredPatch:
        if not isinstance(value, StoredPatch):
            raise TypeError("ProjectStore 返回了无效的 StoredPatch。")
        return value

    @staticmethod
    def _check_project(record: StoredPatch, project_id: str | None) -> None:
        if project_id is not None and record.patch.project_id != project_id:
            raise PatchApprovalError("PROJECT_CONFLICT: Patch 不属于当前 Project。", code="PROJECT_CONFLICT")

    @staticmethod
    def _idempotent_applied(
        record: StoredPatch,
        request: PatchApprovalRequest,
    ) -> PatchApprovalResult:
        value = dict(record.result)
        apply_result = PatchApplyResult(
            patch_id=record.patch.patch_id,
            project_id=request.project_id,
            status="APPLIED",
            previous_hash=value.get("previous_hash"),  # type: ignore[arg-type]
            new_hash=value.get("new_hash"),  # type: ignore[arg-type]
            audit_id=value.get("audit_id"),  # type: ignore[arg-type]
            error_code=value.get("error_code"),  # type: ignore[arg-type]
            message=str(value.get("message", "Patch 已应用。")),
            idempotent=True,
            manuscript_version=value.get("manuscript_version"),  # type: ignore[arg-type]
            bibliography_previous_hash=value.get("bibliography_previous_hash"),  # type: ignore[arg-type]
            bibliography_new_hash=value.get("bibliography_new_hash"),  # type: ignore[arg-type]
            bibliography_applied=bool(value.get("bibliography_applied", False)),
        )
        return PatchApprovalResult(
            patch_id=request.patch_id,
            project_id=request.project_id,
            decision="ACCEPT",
            status="APPLIED",
            actor=request.actor,
            applied=True,
            apply_result=apply_result,
            audit_id=apply_result.audit_id,
            message="Patch 已应用；本次请求为幂等重放。",
        )
