"""Canonical Corpus 的只读 Repository 与统一服务。"""

from app.corpus.repository import (
    ChunkRepository,
    CitationRepository,
    DocumentRepository,
    SectionRepository,
)
from app.corpus.service import CanonicalCorpusService, CanonicalLocator
from app.corpus.versions import (
    DocumentVersionReceipt,
    DocumentVersionRepository,
    ExtractionDecision,
)

__all__ = [
    "CanonicalCorpusService",
    "CanonicalLocator",
    "DocumentVersionReceipt",
    "DocumentVersionRepository",
    "ExtractionDecision",
    "ChunkRepository",
    "CitationRepository",
    "DocumentRepository",
    "SectionRepository",
]
