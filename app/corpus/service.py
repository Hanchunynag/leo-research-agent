"""Canonical Corpus 的唯一服务门面。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.contracts import Document
from app.corpus.repository import ChunkRepository, CitationRepository, DocumentRepository, SectionRepository


@dataclass(frozen=True, slots=True)
class CanonicalLocator:
    document_id: str
    chunk_id: str
    section_id: str | None
    page_start: int
    page_end: int
    block_ids: tuple[str, ...]
    content: str
    work_id: str | None
    metadata: dict[str, Any]


class CanonicalCorpusService:
    def __init__(self, project_root: Path) -> None:
        self.documents = DocumentRepository(project_root)
        self.sections = SectionRepository(project_root)
        self.chunks = ChunkRepository(project_root)
        self.citations = CitationRepository(project_root, self.documents)

    def list_documents(self) -> tuple[Document, ...]:
        return self.documents.list()

    def duplicate_for_hash(self, content_hash: str) -> Document | None:
        return self.documents.find_by_content_hash(content_hash)

    def require_document(self, document_id: str) -> Document:
        value = self.documents.get(document_id)
        if value is None:
            raise KeyError(f"Canonical Document 不存在：{document_id}")
        return value

    def locate_chunk(self, chunk_id: str) -> CanonicalLocator | None:
        value = self.chunks.get(chunk_id)
        return self._locator(value) if value is not None else None

    def locate_document_chunks(self, document_id: str) -> tuple[CanonicalLocator, ...]:
        return tuple(self._locator(value) for value in self.chunks.list_for_document(document_id))

    @staticmethod
    def _locator(value: dict[str, Any]) -> CanonicalLocator:
        return CanonicalLocator(
            document_id=str(value.get("document_id") or ""),
            chunk_id=str(value.get("chunk_id") or ""),
            section_id=str(value.get("section_id")) if value.get("section_id") else None,
            page_start=int(value.get("page_start") or 0),
            page_end=int(value.get("page_end") or 0),
            block_ids=tuple(str(item) for item in value.get("block_ids", []) if isinstance(item, str)),
            content=str(value.get("content") or ""),
            work_id=str(value.get("work_id")) if value.get("work_id") else None,
            metadata={key: value.get(key) for key in ("paper_id", "title", "authors", "year", "doi", "section_path", "content_types")},
        )
