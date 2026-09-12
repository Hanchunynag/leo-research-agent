"""Deterministic Citation Lifecycle contracts.

The bibliography file remains the project authority.  These contracts are
only the identity, audit and change-proposal projections around that file.
They intentionally contain no LLM or filesystem side effects.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Mapping

from app.knowledge.identity import normalize_doi, normalize_identity_text


CitationStatus = Literal[
    "RESOLVED_EXISTING",
    "PROPOSED_NEW_ENTRY",
    "AMBIGUOUS",
    "INSUFFICIENT_METADATA",
    "BIB_CONFLICT",
    "STALE",
    "INVALID",
]
BibliographyChangeAction = Literal["ADD"]


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    if isinstance(value, int) and not isinstance(value, bool) and 1000 <= value <= 3000:
        return date(value, 1, 1)
    return None


def _authors(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in re.split(r"\s+and\s+", value, flags=re.IGNORECASE) if item.strip())
    if isinstance(value, (tuple, list)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _year(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1000 <= value <= 3000 else None
    if isinstance(value, str):
        match = re.search(r"\b(\d{4})\b", value)
        return int(match.group(1)) if match else None
    return None


def _normal_arxiv(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().casefold()
    cleaned = re.sub(r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", cleaned)
    cleaned = re.sub(r"\.pdf$", "", cleaned)
    return re.sub(r"v\d+$", "", cleaned) or None


def metadata_hash(metadata: Mapping[str, Any]) -> str:
    """Hash only deterministic JSON values used in an identity/proposal."""

    return hashlib.sha256(
        json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CitationIdentity:
    """The work identity, independent of a project-local BibKey."""

    canonical_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    title: str = ""
    authors: tuple[str, ...] = ()
    publication_date: date | None = None
    venue: str | None = None
    identity_method: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        doi = normalize_doi(self.doi)
        arxiv = _normal_arxiv(self.arxiv_id)
        canonical = self.canonical_id.strip() if isinstance(self.canonical_id, str) and self.canonical_id.strip() else None
        if canonical and canonical.casefold().startswith("doi:"):
            doi = normalize_doi(canonical[4:]) or doi
            canonical = f"doi:{doi}" if doi else canonical.casefold()
        elif canonical and canonical.casefold().startswith("arxiv:"):
            arxiv = _normal_arxiv(canonical[6:]) or arxiv
            canonical = f"arxiv:{arxiv}" if arxiv else canonical.casefold()
        if doi:
            canonical = f"doi:{doi}"
        elif arxiv:
            canonical = f"arxiv:{arxiv}"
        if not self.title.strip() and not canonical:
            raise ValueError("CitationIdentity 至少需要 canonical_id 或 title。")
        object.__setattr__(self, "doi", doi)
        object.__setattr__(self, "arxiv_id", arxiv)
        object.__setattr__(self, "canonical_id", canonical)
        object.__setattr__(self, "authors", _authors(self.authors))
        object.__setattr__(self, "publication_date", _date(self.publication_date))

    @property
    def key(self) -> str:
        """Stable comparison key; never a project BibKey."""

        if self.canonical_id:
            return self.canonical_id
        if not self.title.strip() or not self.authors or self.publication_date is None:
            raise ValueError("CitationIdentity 缺少可比较的 canonical 或完整 metadata。")
        value = ":".join(
            (
                normalize_identity_text(self.title),
                normalize_identity_text(self.authors[0]),
                str(self.publication_date.year),
            )
        )
        return "bib:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def from_evidence(cls, evidence: Mapping[str, Any]) -> "CitationIdentity | None":
        metadata = evidence.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        def first(*names: str) -> Any:
            for name in names:
                if evidence.get(name) not in (None, "", ()):
                    return evidence.get(name)
                if metadata.get(name) not in (None, "", ()):
                    return metadata.get(name)
            return None

        title = str(first("title") or "").strip()
        authors = _authors(first("authors", "author"))
        publication = _date(first("publication_date", "published_date", "year", "publication_year"))
        doi = normalize_doi(first("doi"))
        arxiv = _normal_arxiv(first("arxiv_id", "arxiv", "eprint"))
        canonical = first("canonical_id")
        if canonical is not None:
            canonical = str(canonical).strip() or None
        if canonical is None:
            work_id = first("work_id")
            if work_id:
                canonical = f"work:{work_id}"
        if not any((canonical, doi, arxiv, title)):
            return None
        return cls(
            canonical_id=canonical,
            doi=doi,
            arxiv_id=arxiv,
            title=title,
            authors=authors,
            publication_date=publication,
            venue=str(first("venue", "journal", "booktitle") or "").strip() or None,
            identity_method="doi" if doi else "arxiv" if arxiv else "canonical_id" if canonical else "metadata",
            provenance={
                "source_type": evidence.get("source_type", "LOCAL_CORPUS"),
                "provider": evidence.get("provider") or metadata.get("provider"),
                "evidence_id": evidence.get("evidence_id"),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": list(self.authors),
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "venue": self.venue,
            "identity_method": self.identity_method,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class CitationRequirement:
    evidence_id: str
    paper_id: str | None = None
    reason: str = "verified evidence has no stable BibKey in the current bibliography"
    identity_key: str | None = None


@dataclass(frozen=True, slots=True)
class BibEntrySnapshot:
    """A read-on-demand, hash-bound view of the actual references.bib."""

    relative_path: str
    content_hash: str
    entries: tuple[Mapping[str, Any], ...] = ()
    exists: bool = True
    duplicate_identity_keys: tuple[str, ...] = ()

    @property
    def bibkeys(self) -> tuple[str, ...]:
        return tuple(str(value.get("ID")) for value in self.entries if value.get("ID"))

    @property
    def bibliography_hash(self) -> str:
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "entries": [dict(value) for value in self.entries],
            "exists": self.exists,
            "duplicate_identity_keys": list(self.duplicate_identity_keys),
        }


@dataclass(frozen=True, slots=True)
class BibEntryCandidate:
    bibkey: str
    entry_type: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str | None = None
    arxiv_id: str | None = None
    canonical_id: str | None = None
    venue: str | None = None
    url: str | None = None
    metadata_provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata_hash: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:\-]*", self.bibkey):
            raise ValueError("BibEntryCandidate bibkey 不合法。")
        if not self.title.strip() or not self.authors or self.year is None:
            raise ValueError("BibEntryCandidate 缺少 title/authors/year。")
        if not self.metadata_hash:
            value = {
                "entry_type": self.entry_type,
                "title": self.title,
                "authors": list(self.authors),
                "year": self.year,
                "doi": self.doi,
                "arxiv_id": self.arxiv_id,
                "canonical_id": self.canonical_id,
                "venue": self.venue,
                "url": self.url,
            }
            object.__setattr__(self, "metadata_hash", metadata_hash(value))

    @property
    def proposed_bibkey(self) -> str:
        return self.bibkey

    def to_bibtex(self) -> str:
        fields: list[tuple[str, str]] = [
            ("author", " and ".join(self.authors)),
            ("title", self.title),
            ("year", str(self.year)),
        ]
        if self.venue:
            fields.append(("journal" if self.entry_type.lower() == "article" else "booktitle", self.venue))
        if self.doi:
            fields.append(("doi", self.doi))
        if self.arxiv_id:
            fields.append(("eprint", self.arxiv_id))
            fields.append(("archivePrefix", "arXiv"))
        if self.canonical_id and not (self.doi or self.arxiv_id):
            fields.append(("canonical_id", self.canonical_id))
        if self.url:
            fields.append(("url", self.url))
        body = ",\n".join(f"  {key} = {{{value}}}" for key, value in fields)
        return f"@{self.entry_type}{{{self.bibkey},\n{body}\n}}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bibkey": self.bibkey,
            "entry_type": self.entry_type,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "canonical_id": self.canonical_id,
            "venue": self.venue,
            "url": self.url,
            "metadata_provenance": dict(self.metadata_provenance),
            "metadata_hash": self.metadata_hash,
        }


@dataclass(frozen=True, slots=True)
class BibliographyChange:
    relative_path: str
    identity_key: str
    entry: BibEntryCandidate
    action: BibliographyChangeAction = "ADD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "relative_path": self.relative_path,
            "identity_key": self.identity_key,
            "entry": self.entry.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CitationBinding:
    binding_id: str
    project_id: str
    citation_identity: CitationIdentity
    bibkey: str | None
    evidence_ids: tuple[str, ...] = ()
    status: CitationStatus = "INVALID"
    metadata_hash: str = ""
    source_type: str | None = None

    def __post_init__(self) -> None:
        if not self.binding_id.strip() or not self.project_id.strip():
            raise ValueError("CitationBinding 必须携带 binding_id/project_id。")
        if self.status in {"RESOLVED_EXISTING", "PROPOSED_NEW_ENTRY"} and not self.bibkey:
            raise ValueError("可用 CitationBinding 必须携带 bibkey。")

    @property
    def identity_key(self) -> str:
        return self.citation_identity.key

    @property
    def bib_key(self) -> str | None:
        return self.bibkey

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "project_id": self.project_id,
            "citation_identity": self.citation_identity.to_dict(),
            "bibkey": self.bibkey,
            "evidence_ids": list(self.evidence_ids),
            "status": self.status,
            "metadata_hash": self.metadata_hash,
            "source_type": self.source_type,
        }


@dataclass(frozen=True, slots=True)
class CitationResolutionResult:
    identity: CitationIdentity | None
    status: CitationStatus
    binding: CitationBinding | None = None
    candidate: BibEntryCandidate | None = None
    requirement: CitationRequirement | None = None
    snapshot: BibEntrySnapshot | None = None
    errors: tuple[str, ...] = ()

    @property
    def bibkey(self) -> str | None:
        return self.binding.bibkey if self.binding else None

    @property
    def citation_binding(self) -> CitationBinding | None:
        return self.binding

    @property
    def bib_entry_candidate(self) -> BibEntryCandidate | None:
        return self.candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "identity": self.identity.to_dict() if self.identity else None,
            "binding": self.binding.to_dict() if self.binding else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "requirement": {
                "evidence_id": self.requirement.evidence_id,
                "paper_id": self.requirement.paper_id,
                "reason": self.requirement.reason,
                "identity_key": self.requirement.identity_key,
            } if self.requirement else None,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "errors": list(self.errors),
        }
