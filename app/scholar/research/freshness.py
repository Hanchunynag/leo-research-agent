"""Deterministic freshness policy for Scholar Research Capability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.scholar.research.models import ResearchRequest


FreshnessMode = Literal["LOCAL_ONLY", "LOCAL_FIRST", "FRESH_REQUIRED"]
DomainSensitivity = Literal["stable", "evolving", "unknown"]


def _date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class FreshnessDecision:
    mode: FreshnessMode
    reason: str
    requested_time_range: tuple[date | None, date | None] = (None, None)
    local_latest_date: date | None = None
    local_coverage: float | None = None
    local_sufficient: bool = False
    web_required: bool = False
    freshness_uncertain: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "requested_time_range": [
                value.isoformat() if isinstance(value, date) else None
                for value in self.requested_time_range
            ],
            "local_latest_date": self.local_latest_date.isoformat() if self.local_latest_date else None,
            "local_coverage": self.local_coverage,
            "local_sufficient": self.local_sufficient,
            "web_required": self.web_required,
            "freshness_uncertain": self.freshness_uncertain,
        }


class FreshnessPolicy:
    """A pure policy function; it never calls a provider or a tool."""

    @staticmethod
    def requested_range(request: "ResearchRequest") -> tuple[date | None, date | None]:
        """Return the normalized publication-date window for a request.

        ``publication_date`` is deliberately kept separate from retrieval
        time.  A missing publication date therefore remains unknown rather
        than being inferred from ``retrieved_at``.
        """

        requested_from = _date(request.requested_from) or (
            date(request.year_from, 1, 1) if request.year_from is not None else None
        )
        requested_to = _date(request.requested_to) or (
            date(request.year_to, 12, 31) if request.year_to is not None else None
        )
        return requested_from, requested_to

    @staticmethod
    def decide(
        request: "ResearchRequest",
        *,
        local_latest_date: date | None = None,
        local_coverage: float | None = None,
        unresolved: tuple[str, ...] = (),
        conflicts: tuple[object, ...] = (),
        local_time_satisfied: bool | None = None,
    ) -> FreshnessDecision:
        requested_from, requested_to = FreshnessPolicy.requested_range(request)
        explicit_range = requested_from is not None or requested_to is not None
        explicit_freshness = bool(request.explicit_latest or explicit_range)
        mode: FreshnessMode = "FRESH_REQUIRED" if explicit_freshness else request.freshness_mode
        coverage_ok = (
            not unresolved
            and not conflicts
            and (local_coverage is None or local_coverage >= (request.budget.coverage_target or 1.0))
        )
        time_ok = local_time_satisfied
        uncertain = False
        if explicit_freshness:
            if local_latest_date is None or time_ok is False:
                uncertain = local_latest_date is None or time_ok is None
            local_sufficient = coverage_ok and time_ok is not False and not uncertain
            return FreshnessDecision(
                mode="FRESH_REQUIRED",
                reason="EXPLICIT_FRESHNESS_REQUIREMENT",
                requested_time_range=(requested_from, requested_to),
                local_latest_date=local_latest_date,
                local_coverage=local_coverage,
                local_sufficient=local_sufficient,
                web_required=True,
                freshness_uncertain=uncertain,
            )

        local_sufficient = coverage_ok
        if mode == "FRESH_REQUIRED":
            return FreshnessDecision(
                mode="FRESH_REQUIRED",
                reason="EXPLICIT_FRESHNESS_MODE",
                requested_time_range=(requested_from, requested_to),
                local_latest_date=local_latest_date,
                local_coverage=local_coverage,
                local_sufficient=local_sufficient,
                web_required=True,
                freshness_uncertain=local_latest_date is None,
            )
        if mode == "LOCAL_ONLY":
            return FreshnessDecision(
                mode=mode,
                reason="LOCAL_ONLY_POLICY",
                local_latest_date=local_latest_date,
                local_coverage=local_coverage,
                local_sufficient=local_sufficient,
                web_required=False,
            )
        # Evolving domains cannot claim that an otherwise complete local
        # corpus is current when its publication horizon is unknown.  This is
        # intentionally a conservative uncertainty signal, not a hidden age
        # threshold and not an invitation to always browse the Web.
        if mode == "LOCAL_FIRST" and request.domain_sensitivity == "evolving" and local_latest_date is None:
            return FreshnessDecision(
                mode="LOCAL_FIRST",
                reason="LOCAL_FRESHNESS_UNKNOWN_FOR_EVOLVING_DOMAIN",
                local_latest_date=None,
                local_coverage=local_coverage,
                local_sufficient=False,
                web_required=True,
                freshness_uncertain=True,
            )
        return FreshnessDecision(
            mode="LOCAL_FIRST",
            reason="LOCAL_EVIDENCE_SUFFICIENT" if local_sufficient else "LOCAL_EVIDENCE_INSUFFICIENT",
            local_latest_date=local_latest_date,
            local_coverage=local_coverage,
            local_sufficient=local_sufficient,
            web_required=not local_sufficient,
            freshness_uncertain=False,
        )
