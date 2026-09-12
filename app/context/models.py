"""证据上下文的稳定数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONTEXT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    rank: int
    score: float | None = None
    retrieval_source: str = "unknown"
    chunk_id: str = ""
    work_id: str = ""
    document_id: str = ""
    paper_id: str | None = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    section_path: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    primary_block_ids: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    content: str = ""
    truncated: bool = False
    token_count: int = 0
    evidence_id: str | None = None
    origin: str = "newly_retrieved"
    source_type: str = "LOCAL_CORPUS"
    canonical_id: str | None = None
    source_locator: str | None = None
    locator_type: str | None = None
    publication_date: str | None = None
    retrieved_at: str | None = None
    provider: str | None = None

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
