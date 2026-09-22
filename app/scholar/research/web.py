"""ToolGateway-backed Web Literature adapter for Scholar Research Capability."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import replace
from datetime import date, datetime, timezone
from time import perf_counter
from typing import Any, Mapping, Protocol

from app.contracts import CandidateEvidence, ExternalEvidenceResolution
from app.knowledge.identity import normalize_doi, normalize_identity_text
from app.scholar.research.budget import (
    CapabilityBudget,
    CapabilityBudgetPolicy,
)
from app.scholar.research.cache import LiteratureDiscoveryCache, literature_request_fingerprint
from app.scholar.research.errors import (
    WebProviderRateLimited,
    WebProviderTimeout,
    WebProviderUnavailable,
)
from app.scholar.research.models import LiteratureCandidate, LiteratureSearchRequest, LiteratureSearchResult


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


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def canonical_identity(raw: Mapping[str, Any]) -> str:
    title = str(raw.get("title") or "").strip()
    authors = _strings(raw.get("authors"))
    publication_date = _date(raw.get("publication_date") or raw.get("publication_year") or raw.get("year"))
    return _canonical_id(raw, title, authors, publication_date)


def _canonical_id(raw: Mapping[str, Any], title: str, authors: tuple[str, ...], publication_date: date | None) -> str:
    doi = normalize_doi(raw.get("doi"))
    if doi:
        return f"doi:{doi}"
    external_ids = raw.get("external_ids")
    external_arxiv = external_ids.get("arxiv") if isinstance(external_ids, Mapping) else None
    arxiv = str(raw.get("arxiv_id") or external_arxiv or "").strip().casefold()
    if arxiv:
        versionless = re.sub(r"v\d+$", "", arxiv)
        return f"arxiv:{versionless}"
    # A title-only hit is not a safe cross-provider identity.  Keep it
    # request/provider scoped until a stronger identifier is resolved.
    if not authors or publication_date is None:
        raw_identity = str(raw.get("paper_id") or raw.get("id") or raw.get("url") or "")
        normalized = ":".join((normalize_identity_text(title), normalize_identity_text(raw_identity)))
        return "unresolved:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    author = authors[0]
    year = str(publication_date.year)
    normalized = ":".join((normalize_identity_text(title), normalize_identity_text(author), year))
    return "bib:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


class LiteratureSearcher(Protocol):
    def __call__(self, request: LiteratureSearchRequest, budget: CapabilityBudget | None) -> Mapping[str, Any] | LiteratureSearchResult: ...


class WebLiteratureAdapter:
    """Normalizes discovery and owns request-scoped external source snapshots."""

    def __init__(
        self,
        *,
        gateway: Any | None = None,
        searcher: LiteratureSearcher | None = None,
        cache: LiteratureDiscoveryCache | None = None,
    ) -> None:
        if gateway is None and searcher is None:
            raise TypeError("WebLiteratureAdapter 需要 ToolGateway 或 searcher。")
        self.gateway = gateway
        self.searcher = searcher
        self.cache = cache
        self._sources: dict[str, ExternalEvidenceResolution] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def _raw_search(self, request: LiteratureSearchRequest, budget: CapabilityBudget | None) -> Mapping[str, Any] | LiteratureSearchResult:
        if self.searcher is not None:
            return self.searcher(request, budget)
        assert self.gateway is not None
        if budget is None:
            budget = CapabilityBudget(
                "scholar_research",
                CapabilityBudgetPolicy(
                    max_steps=8,
                    max_tool_calls=max(2, request.max_results + 1),
                    max_external_searches=1,
                    max_retrieval_rounds=1,
                    max_context_tokens=1_000,
                    max_total_tokens=2_000,
                ),
            )
            budget.transition("context_preparing")
            budget.transition("planning")
            budget.transition("executing")
        return self.gateway.invoke(
            "literature.search",
            {
                "query": request.query,
                "limit": request.max_results,
                **({"year_from": request.requested_from.year} if request.requested_from else {}),
                **({"year_to": request.requested_to.year} if request.requested_to else {}),
            },
            context={
                "workflow": "scholar_research",
                "workspace_id": request.metadata.get("workspace_id", "default"),
                "scope_version": int(request.metadata.get("scope_version", 1)),
                "permissions": ("literature.search", "literature.read"),
            },
            budget=budget,
        )

    @staticmethod
    def _publication_values(item: Mapping[str, Any]) -> tuple[date, ...]:
        values: list[date] = []
        keys = (
            "publication_date",
            "published_date",
            "published_online",
            "published_print",
            "online_date",
            "issued_date",
            "publication_year",
            "year",
        )
        for key in keys:
            parsed = _date(item.get(key))
            if parsed is not None:
                values.append(parsed)
        raw_dates = item.get("publication_dates")
        if isinstance(raw_dates, (list, tuple)):
            for value in raw_dates:
                parsed = _date(value)
                if parsed is not None:
                    values.append(parsed)
        return tuple(dict.fromkeys(values))

    @classmethod
    def _resolve_publication_date(cls, item: Mapping[str, Any]) -> tuple[date | None, tuple[Mapping[str, Any], ...]]:
        values = cls._publication_values(item)
        if not values:
            return None, ()
        # Prefer the explicitly resolved/exact date, then the earliest
        # provider date.  A disagreement is retained as provenance; it is
        # never silently overwritten.
        preferred = _date(item.get("resolved_publication_date") or item.get("publication_date"))
        chosen = preferred or min(values)
        conflicts = ()
        if len(set(values)) > 1:
            conflicts = ({"field": "publication_date", "values": [value.isoformat() for value in values], "chosen": chosen.isoformat()},)
        return chosen, conflicts

    @staticmethod
    def _provider_error(error: Exception) -> Exception:
        if isinstance(error, (WebProviderTimeout, WebProviderRateLimited, WebProviderUnavailable)):
            return error
        name = type(error).__name__.casefold()
        message = str(error)
        if isinstance(error, TimeoutError) or "timeout" in name or "timeout" in message.casefold():
            return WebProviderTimeout(message)
        if "rate" in name or "429" in message or "rate limit" in message.casefold():
            return WebProviderRateLimited(message)
        return WebProviderUnavailable(message)

    @staticmethod
    def _normalize(raw: Mapping[str, Any], request_id: str, query: str) -> LiteratureSearchResult:
        values = raw.get("results")
        candidates: list[LiteratureCandidate] = []
        by_identity: dict[str, LiteratureCandidate] = {}
        conflicts: list[Mapping[str, Any]] = [
            value for value in (raw.get("metadata_conflicts") or ()) if isinstance(value, Mapping)
        ]
        for _ordinal, item in enumerate(values if isinstance(values, list) else (), 1):
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            authors = _strings(item.get("authors"))
            publication_date, date_conflicts = WebLiteratureAdapter._resolve_publication_date(item)
            conflicts.extend({"candidate_id": item.get("paper_id") or item.get("id"), **dict(value)} for value in date_conflicts)
            canonical_id = canonical_identity(item)
            provider_values = item.get("sources") or item.get("provider") or item.get("providers")
            provider = _strings(provider_values)[0] if _strings(provider_values) else str(item.get("provider") or "academic-mcp")
            retrieved_at = _datetime(raw.get("retrieved_at")) or datetime.now(timezone.utc)
            abstract = str(item.get("abstract") or "").strip() or None
            candidate = LiteratureCandidate(
                    candidate_id=str(item.get("paper_id") or item.get("id") or f"WEB_{secrets.token_hex(6)}"),
                    title=title,
                    authors=authors,
                    publication_date=publication_date,
                    venue=str(item.get("venue") or "").strip() or None,
                    doi=str(item.get("doi") or "").strip() or None,
                    arxiv_id=str(item.get("arxiv_id") or ((item.get("external_ids") or {}).get("arxiv") if isinstance(item.get("external_ids"), Mapping) else "") or "").strip() or None,
                    canonical_id=canonical_id,
                    landing_url=str(item.get("landing_page_url") or item.get("url") or item.get("source_url") or "").strip() or None,
                    abstract=abstract,
                    provider=provider,
                    retrieved_at=retrieved_at,
                    metadata_confidence="high" if item.get("doi") or item.get("arxiv_id") else "medium",
                    provenance={"provider": provider, "raw_paper_id": item.get("paper_id"), "sources": item.get("sources", ())},
                    metadata={
                        **dict(item),
                        "publication_date_resolution": {
                            "chosen": publication_date.isoformat() if publication_date else None,
                            "conflicts": [dict(value) for value in date_conflicts],
                        },
                    },
                )
            previous = by_identity.get(canonical_id)
            if previous is None:
                by_identity[canonical_id] = candidate
                continue
            if previous.publication_date and candidate.publication_date and previous.publication_date != candidate.publication_date:
                conflicts.append({"canonical_id": canonical_id, "field": "publication_date", "values": [previous.publication_date.isoformat(), candidate.publication_date.isoformat()], "chosen": previous.publication_date.isoformat()})
            for field in ("title", "authors", "doi", "venue"):
                previous_value = getattr(previous, field)
                candidate_value = getattr(candidate, field)
                if previous_value and candidate_value and str(previous_value).casefold() != str(candidate_value).casefold():
                    conflicts.append({"canonical_id": canonical_id, "field": field, "values": [previous_value, candidate_value]})
            by_identity[canonical_id] = replace(
                previous,
                abstract=previous.abstract or candidate.abstract,
                venue=previous.venue or candidate.venue,
                doi=previous.doi or candidate.doi,
                arxiv_id=previous.arxiv_id or candidate.arxiv_id,
                landing_url=previous.landing_url or candidate.landing_url,
                metadata={**dict(candidate.metadata), **dict(previous.metadata)},
                provenance={"merged_providers": tuple(dict.fromkeys((*previous.provenance.get("sources", ()), *candidate.provenance.get("sources", ())))), "canonical_id": canonical_id},
            )
        candidates = list(by_identity.values())
        failures = tuple(value for value in (raw.get("provider_failures") or ()) if isinstance(value, Mapping))
        return LiteratureSearchResult(request_id, query, tuple(candidates), failures, tuple(conflicts), metadata={"candidate_count": len(candidates)})

    def search(self, request: LiteratureSearchRequest, *, budget: CapabilityBudget | None = None) -> LiteratureSearchResult:
        fingerprint = literature_request_fingerprint(
            request.query,
            provider_set=tuple(str(value) for value in request.metadata.get("provider_set", ())),
            requested_from=request.requested_from.isoformat() if request.requested_from else None,
            requested_to=request.requested_to.isoformat() if request.requested_to else None,
            max_results=request.max_results,
            source_policy=str(request.metadata.get("source_policy") or "default"),
        )
        cached = self.cache.get(fingerprint) if self.cache is not None else None
        if cached is not None:
            result = self._normalize(cached, request.request_id, request.query)
            self.last_diagnostics = {"cache_hit": True, "fingerprint": fingerprint, "candidate_count": len(result.candidates), "metadata_conflicts": list(result.metadata_conflicts)}
            if budget is not None:
                budget.record_provider_usage("web_discovery", {"status": "cache_hit", "candidate_count": len(result.candidates), "metadata_conflict_count": len(result.metadata_conflicts)})
            return result
        started = perf_counter()
        try:
            raw = self._raw_search(request, budget)
        except Exception as error:
            wrapped = self._provider_error(error)
            if budget is not None and self.searcher is not None:
                budget.record_tool("literature.search", "failed", (perf_counter() - started) * 1000, {"error_code": getattr(wrapped, "code", "WEB_PROVIDER_UNAVAILABLE"), "error_type": type(error).__name__})
                budget.record_provider_usage("web_discovery", {"status": "failed", "error_code": getattr(wrapped, "code", "WEB_PROVIDER_UNAVAILABLE")})
            raise wrapped from error
        if isinstance(raw, LiteratureSearchResult):
            result = raw
        else:
            result = self._normalize(raw, request.request_id, request.query)
        if budget is not None and self.searcher is not None:
            budget.record_tool(
                "literature.search",
                "succeeded",
                (perf_counter() - started) * 1000,
                {"candidate_count": len(result.candidates)},
            )
        if self.cache is not None:
            self.cache.put(
                fingerprint,
                {
                    "results": [dict(candidate.metadata) for candidate in result.candidates],
                    "provider_failures": list(result.provider_failures),
                    "metadata_conflicts": list(result.metadata_conflicts),
                },
            )
        self.last_diagnostics = {"cache_hit": False, "fingerprint": fingerprint, "candidate_count": len(result.candidates), "provider_failures": list(result.provider_failures), "metadata_conflicts": list(result.metadata_conflicts), "elapsed_ms": round((perf_counter() - started) * 1000, 3)}
        if budget is not None:
            budget.record_provider_usage("web_discovery", {"status": "cache_miss", "candidate_count": len(result.candidates), "metadata_conflict_count": len(result.metadata_conflicts)})
        return result

    def evidence_candidates(
        self,
        request: LiteratureSearchRequest,
        result: LiteratureSearchResult,
        *,
        workspace_id: str,
        scope_version: int,
    ) -> tuple[CandidateEvidence, ...]:
        values: list[CandidateEvidence] = []
        for candidate in result.candidates:
            raw = candidate.metadata
            fulltext = str(raw.get("fulltext") or raw.get("content") or "").strip()
            snippet = str(raw.get("snippet") or "").strip()
            if request.require_fulltext and not fulltext:
                continue
            if request.require_abstract and not fulltext and not candidate.abstract:
                continue
            content = fulltext or candidate.abstract or ""
            if not content or (not fulltext and snippet and not candidate.abstract):
                continue
            locator_type = "FULLTEXT_SPAN" if fulltext else "ABSTRACT"
            locator = str(
                raw.get("source_locator")
                or raw.get("fulltext_locator")
                or raw.get("fulltext_url")
                or ""
            ).strip()
            if not locator and candidate.landing_url:
                locator = f"{candidate.landing_url}#{'fulltext' if fulltext else 'abstract'}"
            if not candidate.canonical_id or not locator:
                continue
            evidence_id = f"{request.request_id}:WEB:{candidate.canonical_id}"
            source = ExternalEvidenceResolution(
                candidate_id=evidence_id,
                canonical_id=candidate.canonical_id,
                source_locator=locator,
                locator_type=locator_type,  # type: ignore[arg-type]
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                provider=candidate.provider,
                validation_status="provider_payload_verified",
                publication_date=candidate.publication_date,
                retrieved_at=candidate.retrieved_at,
                work_id=str(raw.get("work_id") or ""),
                metadata={"title": candidate.title, "authors": list(candidate.authors), "venue": candidate.venue, "canonical_id": candidate.canonical_id},
            )
            self._sources[evidence_id] = source
            values.append(
                CandidateEvidence(
                    candidate_id=evidence_id,
                    request_id=request.request_id,
                    workspace_id=workspace_id,
                    scope_version=scope_version,
                    content=content,
                    score=float(raw.get("score") or 0.0),
                    retrieval_source="research_capability_web",
                    evidence_grade="candidate" if locator_type == "ABSTRACT" else "primary",
                    metadata={
                        **dict(raw),
                        "metadata_conflicts": [dict(value) for value in result.metadata_conflicts],
                        "paper_id": raw.get("paper_id") or candidate.candidate_id,
                        "title": candidate.title,
                        "authors": list(candidate.authors),
                        "source_type": "WEB_LITERATURE",
                        "evidence_role": raw.get("evidence_role") or raw.get("support_type") or "support",
                        "retrieval_source": "research_capability_web",
                    },
                    source_type="WEB_LITERATURE",
                    canonical_id=candidate.canonical_id,
                    source_locator=locator,
                    locator_type=locator_type,  # type: ignore[arg-type]
                    publication_date=candidate.publication_date,
                    retrieved_at=candidate.retrieved_at,
                    provider=candidate.provider,
                    validation_status="candidate",
                )
            )
        return tuple(values)

    def resolve(self, candidate: CandidateEvidence) -> ExternalEvidenceResolution | None:
        source = self._sources.get(candidate.candidate_id)
        if source is None or source.content != candidate.content:
            return None
        return source
