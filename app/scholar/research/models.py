"""Research Capability 的后端无关 Contract。

这些模型只描述 capability 边界；不包含 BM25、向量、Qdrant 或具体 Tool
框架字段。底层已有 ``CandidateEvidence`` / ``VerifiedEvidence`` 继续作为
Evidence 的 canonical domain model，这里只为上层提供 paper、section 和
locator 的稳定投影。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Mapping

from app.contracts import CandidateEvidence, VerifiedEvidence
from app.scholar.research.freshness import DomainSensitivity, FreshnessMode


ResearchPurpose = Literal[
    "background",
    "prior_work",
    "support_claim",
    "experiment_reference",
    "result_comparison",
]


@dataclass(frozen=True, slots=True)
class ResearchBudget:
    max_papers: int = 10
    max_sections: int = 20
    max_evidence_items: int = 10
    max_web_queries: int = 0
    coverage_target: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_papers", "max_sections", "max_evidence_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须为正整数。")
        if isinstance(self.max_web_queries, bool) or self.max_web_queries < 0:
            raise ValueError("max_web_queries 不能为负数。")
        if self.coverage_target is not None and not 0.0 <= self.coverage_target <= 1.0:
            raise ValueError("coverage_target 必须位于 0 到 1 之间。")


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    request_id: str
    query: str
    workspace_id: str = "default"
    scope_version: int = 1
    purpose: ResearchPurpose = "background"
    target_claims: tuple[str, ...] = ()
    preferred_section_types: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None
    paper_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    freshness_mode: FreshnessMode = "LOCAL_ONLY"
    requested_from: date | datetime | str | None = None
    requested_to: date | datetime | str | None = None
    explicit_latest: bool = False
    domain_sensitivity: DomainSensitivity = "unknown"
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id 不能为空。")
        if not self.query.strip():
            raise ValueError("query 不能为空。")
        if not self.workspace_id.strip():
            raise ValueError("workspace_id 不能为空。")
        if isinstance(self.scope_version, bool) or self.scope_version < 1:
            raise ValueError("scope_version 必须为正整数。")
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from 不能大于 year_to。")
        parsed_from = self._request_date(self.requested_from, "requested_from")
        parsed_to = self._request_date(self.requested_to, "requested_to")
        if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
            raise ValueError("requested_from 不能晚于 requested_to。")
        if any(not value.strip() for value in self.target_claims):
            raise ValueError("target_claims 不能包含空字符串。")
        if self.freshness_mode not in {"LOCAL_ONLY", "LOCAL_FIRST", "FRESH_REQUIRED"}:
            raise ValueError("freshness_mode 不受支持。")
        if self.domain_sensitivity not in {"stable", "evolving", "unknown"}:
            raise ValueError("domain_sensitivity 不受支持。")

    @staticmethod
    def _request_date(value: date | datetime | str | None, name: str) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError as error:
                raise ValueError(f"{name} 必须是 ISO 日期。") from error
        raise ValueError(f"{name} 类型不受支持。")


@dataclass(frozen=True, slots=True)
class PaperCandidate:
    paper_id: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    abstract: str | None = None
    score: float = 0.0
    rank: int = 0
    work_id: str | None = None
    document_id: str | None = None
    doi: str | None = None
    publication_date: date | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("paper_id 不能为空。")


@dataclass(frozen=True, slots=True)
class SectionCandidate:
    paper_id: str
    section_id: str
    chunk_ids: tuple[str, ...] = ()
    title: str = ""
    normalized_type: str | None = None
    preview: str = ""
    score: float = 0.0
    rank: int = 0
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paper_id.strip() or not self.section_id.strip():
            raise ValueError("SectionCandidate 必须携带 paper_id 和 section_id。")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    paper_id: str | None = None
    document_id: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None
    block_id: str | None = None

    def __post_init__(self) -> None:
        if not any((self.paper_id, self.document_id, self.section_id, self.chunk_id, self.block_id)):
            raise ValueError("EvidenceRef 至少需要一个 source locator。")


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    paper_id: str
    document_id: str
    section_id: str | None
    chunk_id: str
    block_ids: tuple[str, ...]
    text: str
    page_start: int | None = None
    page_end: int | None = None
    work_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Existing CandidateEvidence is the canonical Candidate model.  This alias is
# intentional: it prevents a second source/evidence schema from emerging.
EvidenceCandidate = CandidateEvidence
VerifiedEvidenceResult = VerifiedEvidence


@dataclass(frozen=True, slots=True)
class PaperSearchResult:
    request_id: str
    query: str
    papers: tuple[PaperCandidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SectionSearchResult:
    request_id: str
    query: str
    sections: tuple[SectionCandidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiteratureSearchRequest:
    request_id: str
    query: str
    max_results: int = 10
    requested_from: date | None = None
    requested_to: date | None = None
    preferred_source_types: tuple[str, ...] = ("journal", "conference", "preprint")
    required_metadata: tuple[str, ...] = ("title", "authors", "publication_date")
    require_abstract: bool = True
    require_fulltext: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.query.strip():
            raise ValueError("LiteratureSearchRequest request_id/query 不能为空。")
        if isinstance(self.max_results, bool) or self.max_results < 1:
            raise ValueError("max_results 必须为正整数。")
        if self.requested_from and self.requested_to and self.requested_from > self.requested_to:
            raise ValueError("requested_from 不能晚于 requested_to。")


@dataclass(frozen=True, slots=True)
class LiteratureCandidate:
    candidate_id: str
    title: str
    authors: tuple[str, ...] = ()
    publication_date: date | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    canonical_id: str | None = None
    landing_url: str | None = None
    abstract: str | None = None
    provider: str = "unknown"
    retrieved_at: datetime | None = None
    metadata_confidence: str = "unknown"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.title.strip():
            raise ValueError("LiteratureCandidate candidate_id/title 不能为空。")


@dataclass(frozen=True, slots=True)
class LiteratureSearchResult:
    request_id: str
    query: str
    candidates: tuple[LiteratureCandidate, ...] = ()
    provider_failures: tuple[Mapping[str, Any], ...] = ()
    metadata_conflicts: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
