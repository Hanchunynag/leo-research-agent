"""Deterministic Evidence → BibKey resolution service."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from typing import Any
from urllib.parse import urlparse

from app.scholar.citation.bibliography import BibKeyGenerator, BibliographySynchronizer
from app.scholar.citation.models import (
    BibEntryCandidate,
    BibliographyChange,
    CitationBinding,
    CitationIdentity,
    CitationRequirement,
    CitationResolutionResult,
    metadata_hash,
)
from app.scholar.project import ScholarProjectStore


def _date_year(identity: CitationIdentity, evidence: Mapping[str, Any]) -> int | None:
    if identity.publication_date is not None:
        return identity.publication_date.year
    metadata = evidence.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for value in (evidence.get("year"), evidence.get("publication_year"), metadata.get("year"), metadata.get("publication_year")):
        if isinstance(value, int) and not isinstance(value, bool) and 1000 <= value <= 3000:
            return value
        if isinstance(value, str) and value[:4].isdigit():
            return int(value[:4])
    return None


def _binding_id(project_id: str, identity_key: str) -> str:
    digest = hashlib.sha256(f"{project_id}:{identity_key}".encode("utf-8")).hexdigest()[:24]
    return f"BIND_{digest}"


class CitationResolutionService:
    """Resolve current Project BibKeys without writing the bibliography."""

    def __init__(
        self,
        project_root: Any,
        *,
        project_store: ScholarProjectStore | None = None,
        synchronizer: BibliographySynchronizer | None = None,
        key_generator: BibKeyGenerator | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.synchronizer = synchronizer or BibliographySynchronizer(self.project_root)
        self.key_generator = key_generator or BibKeyGenerator()

    def sync(self) -> Any:
        """Re-read .bib and reconcile previously observed Project bindings."""

        snapshot = self.synchronizer.sync()
        for stored in self.project_store.list_citation_bindings():
            if stored.status not in {"RESOLVED_EXISTING", "STALE"}:
                continue
            matches = self.synchronizer.find_matches(snapshot, stored.citation_identity)
            if len(matches) == 1 and matches[0].get("ID"):
                current_key = str(matches[0]["ID"])
                self.project_store.save_citation_binding(replace(
                    stored,
                    bibkey=current_key,
                    status="RESOLVED_EXISTING",
                ))
            elif len(matches) > 1:
                self.project_store.save_citation_binding(replace(stored, status="BIB_CONFLICT"))
            elif not matches:
                self.project_store.save_citation_binding(replace(stored, status="STALE"))
        return snapshot

    @staticmethod
    def _mapping(evidence: Any) -> Mapping[str, Any]:
        if isinstance(evidence, Mapping):
            return evidence
        if is_dataclass(evidence):
            return asdict(evidence)
        raise TypeError("CitationResolutionService 需要 Evidence mapping 或 VerifiedEvidence。")

    @staticmethod
    def _verified(evidence: Mapping[str, Any]) -> bool:
        if not str(evidence.get("evidence_id") or "").strip() or not str(evidence.get("content") or evidence.get("text") or "").strip():
            return False
        source_type = str(evidence.get("source_type") or "LOCAL_CORPUS")
        if source_type == "WEB_LITERATURE":
            content = str(evidence.get("content") or evidence.get("text") or "")
            content_hash = str(evidence.get("content_hash") or "")
            return bool(
                evidence.get("canonical_id")
                and evidence.get("source_locator")
                and evidence.get("locator_type") in {"ABSTRACT", "FULLTEXT_SPAN"}
                and evidence.get("verification_method")
                and str(evidence.get("validation_status") or "").casefold() not in {"candidate", "unverified", ""}
                and urlparse(str(evidence.get("source_locator"))).scheme in {"http", "https", "doi", "arxiv"}
                and len(content_hash) == 64
                and content_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()
            )
        validation_status = str(evidence.get("validation_status") or "").casefold()
        if validation_status in {"candidate", "unverified", "rejected"}:
            return False
        return bool(
            evidence.get("verification_method")
            or evidence.get("validation_status") in {"canonical_locator_verified", "verified"}
            or (evidence.get("document_id") and evidence.get("chunk_id"))
        )

    @staticmethod
    def _metadata_conflicts(evidence: Mapping[str, Any]) -> tuple[Any, ...]:
        metadata = evidence.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        values = evidence.get("metadata_conflicts") or metadata.get("metadata_conflicts")
        if not isinstance(values, (list, tuple)):
            return ()
        # Publication date disagreements have already passed the deterministic
        # resolver when a chosen date is recorded.  Keep their provenance but
        # do not treat a resolved online/print date as an identity conflict.
        return tuple(
            value for value in values
            if not (
                isinstance(value, Mapping)
                and value.get("field") == "publication_date"
                and value.get("chosen")
            )
        )

    @staticmethod
    def _metadata(evidence: Mapping[str, Any], identity: CitationIdentity, year: int | None) -> dict[str, Any]:
        source_metadata = evidence.get("metadata")
        source_metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
        return {
            "canonical_id": identity.canonical_id,
            "doi": identity.doi,
            "arxiv_id": identity.arxiv_id,
            "title": identity.title,
            "authors": list(identity.authors),
            "year": year,
            "venue": identity.venue,
            "source_type": evidence.get("source_type", "LOCAL_CORPUS"),
            "provider": evidence.get("provider") or source_metadata.get("provider"),
            "source_locator": evidence.get("source_locator"),
            "metadata_provenance": source_metadata.get("provenance") or source_metadata.get("metadata_provenance") or {},
        }

    def _requirement(
        self,
        evidence: Mapping[str, Any],
        identity: CitationIdentity | None,
        reason: str,
    ) -> CitationResolutionResult:
        evidence_id = str(evidence.get("evidence_id") or "")
        paper_id = str(evidence.get("paper_id")) if evidence.get("paper_id") else None
        try:
            identity_key = identity.key if identity is not None else None
        except ValueError:
            identity_key = None
        requirement = CitationRequirement(
            evidence_id=evidence_id,
            paper_id=paper_id,
            reason=reason,
            identity_key=identity_key,
        )
        return CitationResolutionResult(identity, "INSUFFICIENT_METADATA", requirement=requirement, errors=(reason,))

    def resolve(
        self,
        evidence: Mapping[str, Any] | Any,
        *,
        project_id: str | None = None,
    ) -> CitationResolutionResult:
        """Resolve one *VerifiedEvidence* projection against current .bib."""

        evidence = self._mapping(evidence)
        resolved_project = project_id or self.project_store.project_id
        if resolved_project != self.project_store.project_id:
            return CitationResolutionResult(None, "INVALID", errors=("PROJECT_CONFLICT",))
        if not self._verified(evidence):
            return CitationResolutionResult(None, "INVALID", errors=("CITATION_REQUIRES_VERIFIED_EVIDENCE",))
        identity = CitationIdentity.from_evidence(evidence)
        if identity is None:
            return self._requirement(evidence, None, "INSUFFICIENT_BIB_METADATA: 缺少文献身份元数据。")
        try:
            identity_key = identity.key
        except ValueError:
            return self._requirement(evidence, identity, "INSUFFICIENT_BIB_METADATA: 缺少可比较的文献身份。")
        snapshot = self.sync()
        if identity_key in set(snapshot.duplicate_identity_keys):
            status = "AMBIGUOUS" if identity.identity_method == "metadata" else "BIB_CONFLICT"
            error = "CITATION_IDENTITY_AMBIGUOUS" if status == "AMBIGUOUS" else "BIB_METADATA_CONFLICT"
            return CitationResolutionResult(identity, status, snapshot=snapshot, errors=(error,))  # type: ignore[arg-type]
        conflicts = self._metadata_conflicts(evidence)
        if conflicts:
            return CitationResolutionResult(identity, "BIB_CONFLICT", snapshot=snapshot, errors=("BIB_METADATA_CONFLICT",))
        matches = self.synchronizer.find_matches(snapshot, identity)
        if len(matches) > 1:
            status = "AMBIGUOUS" if identity.identity_method == "metadata" else "BIB_CONFLICT"
            error = "CITATION_IDENTITY_AMBIGUOUS" if status == "AMBIGUOUS" else "BIB_METADATA_CONFLICT"
            return CitationResolutionResult(identity, status, snapshot=snapshot, errors=(error,))  # type: ignore[arg-type]
        evidence_id = str(evidence["evidence_id"])
        source_type = str(evidence.get("source_type") or "LOCAL_CORPUS")
        if matches:
            bibkey = str(matches[0].get("ID"))
            binding_id = _binding_id(resolved_project, identity_key)
            prior = None
            try:
                prior = self.project_store.get_citation_binding(binding_id)
            except KeyError:
                pass
            binding = CitationBinding(
                binding_id=binding_id,
                project_id=resolved_project,
                citation_identity=identity,
                bibkey=bibkey,
                evidence_ids=tuple(dict.fromkeys((*(prior.evidence_ids if prior is not None else ()), evidence_id))),  # type: ignore[union-attr]
                status="RESOLVED_EXISTING",
                metadata_hash=metadata_hash(identity.to_dict()),
                source_type=source_type,
            )
            self.project_store.save_citation_binding(binding)
            self._project_evidence(evidence, identity, resolved_project)
            return CitationResolutionResult(identity, "RESOLVED_EXISTING", binding=binding, snapshot=snapshot)

        year = _date_year(identity, evidence)
        if not identity.title.strip() or not identity.authors or year is None:
            return self._requirement(evidence, identity, "INSUFFICIENT_BIB_METADATA: title/authors/year 不完整。")
        metadata = self._metadata(evidence, identity, year)
        candidate = BibEntryCandidate(
            bibkey=self.key_generator.propose(identity, snapshot.bibkeys),
            entry_type="article" if identity.venue else "misc",
            title=identity.title,
            authors=identity.authors,
            year=year,
            doi=identity.doi,
            arxiv_id=identity.arxiv_id,
            canonical_id=identity.canonical_id,
            venue=identity.venue,
            url=str(evidence.get("landing_url") or evidence.get("url") or "").strip() or None,
            metadata_provenance={
                "evidence_id": evidence_id,
                "provider": evidence.get("provider"),
                "source": metadata.get("metadata_provenance", {}),
            },
        )
        binding_id = _binding_id(resolved_project, identity_key)
        prior = None
        try:
            prior = self.project_store.get_citation_binding(binding_id)
        except KeyError:
            pass
        binding = CitationBinding(
            binding_id=binding_id,
            project_id=resolved_project,
            citation_identity=identity,
            bibkey=candidate.bibkey,
            evidence_ids=tuple(dict.fromkeys((*(prior.evidence_ids if prior is not None else ()), evidence_id))),  # type: ignore[union-attr]
            status="PROPOSED_NEW_ENTRY",
            metadata_hash=candidate.metadata_hash,
            source_type=source_type,
        )
        self.project_store.save_citation_binding(binding)
        self._project_evidence(evidence, identity, resolved_project)
        return CitationResolutionResult(identity, "PROPOSED_NEW_ENTRY", binding=binding, candidate=candidate, snapshot=snapshot)

    def resolve_many(
        self,
        evidence: Sequence[Mapping[str, Any]],
        *,
        project_id: str | None = None,
    ) -> tuple[CitationResolutionResult, ...]:
        return tuple(self.resolve(value, project_id=project_id) for value in evidence)

    resolve_evidence = resolve

    @staticmethod
    def render_citation(binding: CitationBinding, *, command: str = "cite") -> str:
        """Render only an actually resolved/proposed binding, never a free key."""

        if binding.status not in {"RESOLVED_EXISTING", "PROPOSED_NEW_ENTRY"} or not binding.bibkey:
            raise ValueError("CITATION_NOT_RESOLVED: CitationBinding 没有可渲染的 BibKey。")
        if not command.isidentifier() or command.casefold() not in {"cite", "citep", "citet", "citeauthor"}:
            raise ValueError("不支持的 LaTeX citation command。")
        return f"\\{command}{{{binding.bibkey}}}"

    @staticmethod
    def change(result: CitationResolutionResult, *, relative_path: str | None = None) -> BibliographyChange | None:
        if result.status != "PROPOSED_NEW_ENTRY" or result.candidate is None or result.identity is None:
            return None
        path = relative_path or (result.snapshot.relative_path if result.snapshot else "references.bib")
        return BibliographyChange(
            relative_path=path,
            identity_key=result.identity.key,
            entry=result.candidate,
        )

    def _project_evidence(self, evidence: Mapping[str, Any], identity: CitationIdentity, project_id: str) -> None:
        if str(evidence.get("source_type") or "LOCAL_CORPUS") != "WEB_LITERATURE":
            return
        self.project_store.save_external_evidence_projection(
            project_id=project_id,
            evidence_id=str(evidence.get("evidence_id")),
            identity_key=identity.key,
            canonical_id=identity.canonical_id,
            source_locator=str(evidence.get("source_locator") or ""),
            provider=str(evidence.get("provider") or "unknown"),
            publication_date=str(evidence.get("publication_date")) if evidence.get("publication_date") else None,
            retrieved_at=str(evidence.get("retrieved_at")) if evidence.get("retrieved_at") else None,
            content_hash=str(evidence.get("content_hash") or hashlib.sha256(str(evidence.get("content") or evidence.get("text") or "").encode("utf-8")).hexdigest()),
            validation_status=str(evidence.get("validation_status") or ""),
            metadata=dict(evidence),
        )
