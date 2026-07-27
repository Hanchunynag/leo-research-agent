"""证据上下文的稳定数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CONTEXT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    rank: int
    score: float | None
    retrieval_source: str
    chunk_id: str
    work_id: str
    document_id: str
    paper_id: str | None
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    section_path: list[str]
    page_start: int
    page_end: int
    primary_block_ids: list[str]
    block_ids: list[str]
    content_types: list[str]
    content: str
    truncated: bool
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextBundle:
    query: str
    retrieval_mode: str
    evidence: list[EvidenceItem]
    context_text: str
    token_budget: int
    token_count: int
    diagnostics: dict[str, Any]
    schema_version: str = CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "retrieval_mode": self.retrieval_mode,
            "evidence_count": len(self.evidence),
            "token_budget": self.token_budget,
            "token_count": self.token_count,
            "diagnostics": self.diagnostics,
            "evidence": [item.to_dict() for item in self.evidence],
            "context_text": self.context_text,
        }
