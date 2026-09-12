"""ScholarHarness 的第一版结构化 Contract。

这些类型只描述领域数据，不负责调用 LLM、RAG 或写入文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


ContributionStatus = Literal["candidate_hypothesis", "confirmed", "rejected"]
FactSource = Literal["user_confirmed", "manuscript_extracted", "literature_suggested"]
ReviewSeverity = Literal["BLOCKER", "HIGH", "MEDIUM", "LOW"]


@dataclass(frozen=True, slots=True)
class ManuscriptSection:
    name: str
    relative_path: str
    content_hash: str
    version: int
    stale: bool = False


@dataclass(frozen=True, slots=True)
class ManuscriptState:
    project_root: str
    root_tex: str
    project_hash: str
    version: int
    sections: Mapping[str, ManuscriptSection]
    stale_sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManuscriptFact:
    fact_id: str
    key: str
    value: Any
    source: FactSource = "user_confirmed"
    confirmed: bool = False
    source_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fact_id.strip() or not self.key.strip():
            raise ValueError("ManuscriptFact fact_id/key 不能为空。")
        if self.source == "user_confirmed" and not self.confirmed:
            raise ValueError("user_confirmed Fact 必须 confirmed=True。")


@dataclass(frozen=True, slots=True)
class Contribution:
    contribution_id: str
    statement: str
    status: ContributionStatus = "candidate_hypothesis"
    confirmed_by_user: bool = False
    source_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.contribution_id.strip() or not self.statement.strip():
            raise ValueError("Contribution 标识和内容不能为空。")
        if self.status == "confirmed" and not self.confirmed_by_user:
            raise ValueError("只有用户确认后 Contribution 才能进入 confirmed。")


@dataclass(frozen=True, slots=True)
class EvidencePack:
    request_id: str
    query: str
    claims: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    unresolved: tuple[str, ...] = ()
    local_coverage: float = 0.0
    web_used: bool = False
    coverage: float | None = None
    counter_evidence: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DraftPatch:
    patch_id: str
    target_section: str
    base_hash: str
    proposed_content: str
    used_claim_ids: tuple[str, ...] = ()
    used_evidence_ids: tuple[str, ...] = ()
    reviewer_status: str = "pending"
    project_id: str | None = None
    citation_keys: tuple[str, ...] = ()
    contribution_ids: tuple[str, ...] = ()
    review_report_id: str | None = None
    change_summary: str = ""
    original_content: str = ""
    warnings: tuple[str, ...] = ()
    # Citation lifecycle projections.  These remain immutable proposal data;
    # the Approval service is the only component allowed to apply changes.
    bibliography_base_hash: str | None = None
    citation_bindings: tuple[Any, ...] = ()
    bibliography_changes: tuple[Any, ...] = ()
    citation_requirements: tuple[Any, ...] = ()
    citation_binding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patch_id.strip() or not self.target_section.strip():
            raise ValueError("DraftPatch 标识和目标章节不能为空。")
        if not self.base_hash.strip():
            raise ValueError("DraftPatch 必须携带 base_hash。")


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    code: str
    severity: ReviewSeverity
    message: str
    claim_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewReport:
    valid: bool
    issues: tuple[ReviewIssue, ...] = ()
    revision_round: int = 0
    report_id: str = ""
