"""Deterministic access to the project's user-managed ``references.bib``.

``bibtexparser`` is deliberately used only as a parser.  The synchronizer
never rewrites existing entries; additions are rendered by the deterministic
``BibEntryCandidate`` contract and appended atomically.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from app.knowledge.identity import normalize_doi
from app.scholar.citation.errors import BibKeyCollision, BibPathInvalid, BibStaleBaseHash
from app.scholar.citation.models import (
    BibEntryCandidate,
    BibEntrySnapshot,
    CitationIdentity,
)


def _arxiv(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().casefold()
    value = re.sub(r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", value)
    value = re.sub(r"\.pdf$", "", value)
    return re.sub(r"v\d+$", "", value) or None


def _first_author(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    first = re.split(r"\s+and\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return first or None


def _year(value: Any) -> int | None:
    match = re.search(r"\b(\d{4})\b", str(value or ""))
    return int(match.group(1)) if match else None


def _field(entry: Mapping[str, Any], name: str) -> Any:
    for key, value in entry.items():
        if str(key).casefold() == name.casefold():
            return value
    return None


def identity_from_bib_entry(entry: Mapping[str, Any]) -> CitationIdentity | None:
    """Derive a conservative identity from a parsed BibTeX entry."""

    doi = normalize_doi(_field(entry, "doi"))
    arxiv = _arxiv(_field(entry, "eprint") or _field(entry, "arxiv_id"))
    archive = str(_field(entry, "archiveprefix") or "").casefold()
    if not arxiv and archive == "arxiv":
        arxiv = _arxiv(_field(entry, "url"))
    canonical = _field(entry, "canonical_id")
    title = str(_field(entry, "title") or "").strip()
    authors = _first_author(_field(entry, "author"))
    year = _year(_field(entry, "year"))
    if not any((doi, arxiv, canonical, title)):
        return None
    return CitationIdentity(
        canonical_id=str(canonical).strip() if canonical else None,
        doi=doi,
        arxiv_id=arxiv,
        title=title,
        authors=(authors,) if authors else (),
        publication_date=None if year is None else date(year, 1, 1),
        venue=str(_field(entry, "journal") or _field(entry, "booktitle") or "").strip() or None,
        identity_method="doi" if doi else "arxiv" if arxiv else "canonical_id" if canonical else "metadata",
        provenance={"source": "references.bib", "bibkey": str(_field(entry, "ID") or "")},
    )


class BibKeyGenerator:
    """Project-local deterministic BibKey proposal and collision handling."""

    @staticmethod
    def _author_token(identity: CitationIdentity) -> str:
        value = identity.authors[0] if identity.authors else "Source"
        value = value.split(",", 1)[0].strip().split()[-1] if value.strip() else "Source"
        value = re.sub(r"[^A-Za-z0-9]+", "", value)
        return value[:24].capitalize() or "Source"

    @staticmethod
    def _title_token(identity: CitationIdentity) -> str:
        stop = {"the", "a", "an", "of", "and", "for", "on", "in", "to", "with"}
        words = re.findall(r"[A-Za-z0-9]+", identity.title)
        for word in words:
            if word.casefold() not in stop:
                return word[:18].capitalize()
        return "Work"

    def propose(self, identity: CitationIdentity, occupied: Iterable[str] = ()) -> str:
        year = identity.publication_date.year if identity.publication_date else ""
        base = f"{self._author_token(identity)}{year}{self._title_token(identity)}"
        occupied_set = set(occupied)
        if base not in occupied_set:
            return base
        for index, suffix in enumerate("abcdefghijklmnopqrstuvwxyz", 1):
            candidate = f"{base}{suffix}"
            if candidate not in occupied_set:
                return candidate
        index = 1
        while f"{base}{index}" in occupied_set:
            index += 1
        return f"{base}{index}"

    generate = propose


class BibliographySynchronizer:
    """Read and reconcile the actual bibliography file for one project."""

    def __init__(self, project_root: Path, bibliography_path: str | Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self._configured_path = bibliography_path

    @property
    def path(self) -> Path:
        raw = self._configured_path
        if raw is not None:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
        else:
            candidate = self._discover_path()
        return self._safe_path(candidate)

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(self.project_root).as_posix()

    def _discover_path(self) -> Path:
        # The LaTeX project is authoritative.  Prefer a declared resource,
        # then the conventional root references.bib path.
        pattern = re.compile(r"\\(?:bibliography|addbibresource)\s*\{([^}]+)\}")
        for tex in sorted(self.project_root.rglob("*.tex")):
            if not tex.is_file():
                continue
            try:
                text = tex.read_text(encoding="utf-8")
            except OSError:
                continue
            match = pattern.search(text)
            if match:
                value = match.group(1).split(",", 1)[0].strip()
                if value:
                    return self.project_root / (value if value.endswith(".bib") else f"{value}.bib")
        return self.project_root / "references.bib"

    def _safe_path(self, value: Path) -> Path:
        try:
            resolved = value.expanduser().resolve()
        except OSError as error:
            raise BibPathInvalid(f"无法解析 bibliography 路径：{value}") from error
        if resolved != self.project_root and self.project_root not in resolved.parents:
            raise BibPathInvalid("BIB_PATH_INVALID: bibliography 不能越出 project_root。")
        return resolved

    @staticmethod
    def _hash(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _parse(raw: str) -> tuple[Mapping[str, Any], ...]:
        try:
            import bibtexparser
            from bibtexparser.bparser import BibTexParser
        except ImportError as error:  # pragma: no cover - dependency is declared in pyproject
            raise RuntimeError("BibTeX parser dependency is unavailable") from error
        parser = BibTexParser(common_strings=False, ignore_nonstandard_types=False)
        database = bibtexparser.loads(raw, parser=parser)
        return tuple(dict(entry) for entry in database.entries)

    def sync(self) -> BibEntrySnapshot:
        path = self.path
        raw = path.read_bytes() if path.is_file() else b""
        entries = self._parse(raw.decode("utf-8")) if raw else ()
        by_identity: dict[str, int] = {}
        duplicate: set[str] = set()
        for entry in entries:
            identity = identity_from_bib_entry(entry)
            if identity is None:
                continue
            try:
                key = identity.key
            except ValueError:
                continue
            by_identity[key] = by_identity.get(key, 0) + 1
        duplicate.update(key for key, count in by_identity.items() if count > 1)
        return BibEntrySnapshot(
            relative_path=path.relative_to(self.project_root).as_posix(),
            content_hash=self._hash(raw),
            entries=entries,
            exists=path.is_file(),
            duplicate_identity_keys=tuple(sorted(duplicate)),
        )

    synchronize = sync
    snapshot = sync

    @staticmethod
    def matches(identity: CitationIdentity, entry: Mapping[str, Any]) -> bool:
        other = identity_from_bib_entry(entry)
        if other is None:
            return False
        # Strong identifiers never fall back to title matching.  This is what
        # prevents two different DOI records with similar titles merging.
        if identity.doi:
            return other.doi == identity.doi
        if identity.arxiv_id:
            return other.arxiv_id == identity.arxiv_id
        if identity.canonical_id and not identity.canonical_id.startswith("work:"):
            return other.canonical_id == identity.canonical_id
        try:
            return other.key == identity.key
        except ValueError:
            return False

    def find_matches(self, snapshot: BibEntrySnapshot, identity: CitationIdentity) -> tuple[Mapping[str, Any], ...]:
        return tuple(entry for entry in snapshot.entries if self.matches(identity, entry))

    def key_for_identity(self, snapshot: BibEntrySnapshot, identity: CitationIdentity) -> tuple[str, ...]:
        return tuple(str(entry.get("ID")) for entry in self.find_matches(snapshot, identity) if entry.get("ID"))

    def write_changes(
        self,
        snapshot: BibEntrySnapshot,
        changes: Iterable[Any],
    ) -> BibEntrySnapshot:
        """Apply only ADD proposals against the exact snapshot hash."""

        current = self.sync()
        if current.content_hash != snapshot.content_hash:
            raise BibStaleBaseHash("BIB_STALE_BASE_HASH: references.bib 已被修改。")
        additions: list[str] = []
        occupied = set(current.bibkeys)
        added_identity_keys: set[str] = set()
        for change in changes:
            if getattr(change, "relative_path", None) != snapshot.relative_path or getattr(change, "action", None) != "ADD":
                raise BibPathInvalid("BIB_PATH_INVALID: bibliography change 目标不匹配。")
            candidate = change.entry
            identity = candidate_identity(candidate)
            identity_key = identity.key
            if identity_key in added_identity_keys:
                continue
            matches = self.find_matches(current, identity)
            if len(matches) > 1:
                raise BibKeyCollision("BIB_CONFLICT: bibliography 中存在重复文献身份。")
            if matches:
                # A retry is idempotent when the identity is already present.
                continue
            if candidate.bibkey in occupied:
                raise BibKeyCollision(f"BIBKEY_COLLISION: BibKey 已被占用：{candidate.bibkey}")
            occupied.add(candidate.bibkey)
            added_identity_keys.add(identity_key)
            additions.append(candidate.to_bibtex())
        if not additions:
            return current
        path = self.path
        original = path.read_text(encoding="utf-8") if path.is_file() else ""
        rendered = original
        if rendered and not rendered.endswith("\n"):
            rendered += "\n"
        rendered += ("\n" if rendered.strip() else "").join(additions) + "\n"
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return self.sync()


def candidate_identity(candidate: BibEntryCandidate) -> CitationIdentity:
    return CitationIdentity(
        canonical_id=candidate.canonical_id,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        title=candidate.title,
        authors=candidate.authors,
        publication_date=date(candidate.year, 1, 1) if candidate.year else None,
        venue=candidate.venue,
        identity_method="doi" if candidate.doi else "arxiv" if candidate.arxiv_id else "metadata",
    )
