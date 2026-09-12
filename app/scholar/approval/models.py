"""Phase 2B 人工审批和 Build 的框架无关 Contract。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from app.scholar.models import DraftPatch, ReviewReport


ApprovalDecision = Literal["ACCEPT", "REJECT"]
PatchLifecycleStatus = Literal[
    "PROPOSED",
    "AWAITING_APPROVAL",
    "REJECTED",
    "APPROVED",
    "APPLYING",
    "APPLIED",
    "CONFLICT",
    "FAILED",
    "BUILD_FAILED",
]
LatexDiagnosticSeverity = Literal["ERROR", "WARNING", "INFO"]
BuildStatus = Literal[
    "SUCCESS",
    "FAILED",
    "UNAVAILABLE",
    "UNKNOWN",
    "BUILD_TRIGGERED",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PatchApprovalRequest:
    patch_id: str
    project_id: str
    decision: ApprovalDecision
    expected_base_hash: str
    actor: str
    timestamp: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.patch_id.strip() or not self.project_id.strip():
            raise ValueError("PatchApprovalRequest 必须携带 patch_id/project_id。")
        if self.decision not in {"ACCEPT", "REJECT"}:
            raise ValueError("decision 只允许 ACCEPT 或 REJECT。")
        if not self.expected_base_hash.strip():
            raise ValueError("PatchApprovalRequest 必须携带 expected_base_hash。")
        if not self.actor.strip():
            raise ValueError("PatchApprovalRequest 必须携带 actor。")


@dataclass(frozen=True, slots=True)
class PatchPreview:
    patch: DraftPatch
    status: PatchLifecycleStatus
    review_report: ReviewReport | None = None
    source_session_id: str | None = None
    source_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def patch_id(self) -> str:
        return self.patch.patch_id

    @property
    def original_content(self) -> str:
        return self.patch.original_content

    @property
    def proposed_content(self) -> str:
        return self.patch.proposed_content


@dataclass(frozen=True, slots=True)
class StoredPatch:
    patch: DraftPatch
    status: PatchLifecycleStatus
    review_report: ReviewReport | None = None
    source_session_id: str | None = None
    source_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    result: Mapping[str, Any] = field(default_factory=dict)

    def preview(self) -> PatchPreview:
        return PatchPreview(
            patch=self.patch,
            status=self.status,
            review_report=self.review_report,
            source_session_id=self.source_session_id,
            source_run_id=self.source_run_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass(frozen=True, slots=True)
class PatchApplyResult:
    patch_id: str
    project_id: str
    status: PatchLifecycleStatus
    previous_hash: str | None = None
    new_hash: str | None = None
    audit_id: str | None = None
    error_code: str | None = None
    message: str = ""
    idempotent: bool = False
    manuscript_version: int | None = None
    bibliography_previous_hash: str | None = None
    bibliography_new_hash: str | None = None
    bibliography_applied: bool = False


@dataclass(frozen=True, slots=True)
class PatchApprovalResult:
    patch_id: str
    project_id: str
    decision: ApprovalDecision
    status: PatchLifecycleStatus
    actor: str
    applied: bool = False
    apply_result: PatchApplyResult | None = None
    audit_id: str | None = None
    error_code: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class LatexDiagnostic:
    file: str | None
    line: int | None
    column: int | None
    severity: LatexDiagnosticSeverity
    message: str
    source: str = "latex-workshop"

    def __post_init__(self) -> None:
        if self.severity not in {"ERROR", "WARNING", "INFO"}:
            raise ValueError("LaTeX Diagnostic severity 不合法。")
        if not self.message.strip():
            raise ValueError("LaTeX Diagnostic message 不能为空。")


@dataclass(frozen=True, slots=True)
class BuildResult:
    build_id: str
    project_id: str
    status: BuildStatus
    started_at: str
    completed_at: str | None = None
    diagnostics: tuple[LatexDiagnostic, ...] = ()
    message: str = ""
    patch_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"SUCCESS", "FAILED", "UNAVAILABLE", "UNKNOWN", "BUILD_TRIGGERED"}:
            raise ValueError("BuildResult status 不合法。")
