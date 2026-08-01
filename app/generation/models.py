"""回答、引用和校验结果的稳定数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.context.models import ContextBundle


ANSWER_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AnswerClaim:
    claim_id: str
    text: str
    source_ids: list[str]
    category: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerDraft:
    answerable: bool
    claims: list[AnswerClaim]
    refusal_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "claims": [claim.to_dict() for claim in self.claims],
            "refusal_reason": self.refusal_reason,
        }


@dataclass(frozen=True)
class CitationRecord:
    claim_id: str
    source_id: str
    chunk_id: str
    work_id: str
    document_id: str
    title: str
    section_path: list[str]
    page_start: int
    page_end: int
    block_ids: list[str]
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitationValidationIssue:
    code: str
    message: str
    claim_id: str | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitationValidationReport:
    valid: bool
    issues: list[CitationValidationIssue]
    citations: list[CitationRecord]

    def to_dict(self, *, include_citations: bool = True) -> dict[str, Any]:
        payload = {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if include_citations:
            payload["citations"] = [
                citation.to_dict() for citation in self.citations
            ]
        return payload


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    answerable: bool
    answer: str
    claims: list[AnswerClaim]
    citations: list[CitationRecord]
    refusal_reason: str | None
    validation: CitationValidationReport
    context: ContextBundle
    diagnostics: dict[str, Any]
    schema_version: str = ANSWER_SCHEMA_VERSION

    def to_dict(self, *, include_context: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "query": self.query,
            "answerable": self.answerable,
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "citations": [citation.to_dict() for citation in self.citations],
            "refusal_reason": self.refusal_reason,
            "validation": self.validation.to_dict(
                include_citations=include_context,
            ),
            "diagnostics": self.diagnostics,
        }
        if include_context:
            payload["context"] = self.context.to_dict()
        return payload
