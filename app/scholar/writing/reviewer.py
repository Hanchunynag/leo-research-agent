"""Introduction Reviewer：确定性 provenance 检查和可选语义 Judge。"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.scholar.models import Contribution, ManuscriptFact, ReviewIssue, ReviewReport
from app.scholar.citation.models import CitationBinding, CitationIdentity
from app.scholar.writing.models import ClaimPlan, SectionDraft
from app.scholar.writing.citation_coverage import (
    INTRODUCTION_MINIMUM_UNIQUE_PAPERS,
    citation_coverage,
    insufficient_introduction_message,
)


ReviewPolicy = str


class SemanticReviewJudge(Protocol):
    def review(self, payload: Mapping[str, Any]) -> Sequence[ReviewIssue]: ...


def _issue(code: str, severity: str, message: str, claim_id: str | None = None) -> ReviewIssue:
    return ReviewIssue(code, severity, message, claim_id)  # type: ignore[arg-type]


class IntroductionReviewer:
    def __init__(self, semantic_judge: SemanticReviewJudge | None = None) -> None:
        self.semantic_judge = semantic_judge

    def review(
        self,
        draft: SectionDraft,
        claim_plan: ClaimPlan,
        evidence: Mapping[str, Mapping[str, Any]],
        facts: Sequence[ManuscriptFact],
        contributions: Sequence[Contribution],
        citation_catalog: Mapping[str, tuple[str, ...]],
        *,
        revision_round: int = 0,
        citation_bindings: Mapping[str, CitationBinding] | None = None,
        minimum_unique_papers: int | None = None,
    ) -> ReviewReport:
        issues: list[ReviewIssue] = []
        claim_by_id = {value.claim_id: value for value in claim_plan.claims}
        contribution_ids = {value.contribution_id for value in contributions if value.status == "confirmed" and value.confirmed_by_user}
        evidence_ids = set(evidence)

        if not draft.content.strip():
            issues.append(_issue("EMPTY_DRAFT", "BLOCKER", "Draft content is empty."))
        unknown_claims = set(draft.claim_ids) - set(claim_by_id)
        if unknown_claims:
            issues.append(_issue("UNKNOWN_CLAIM_ID", "HIGH", f"Draft references unknown claim IDs: {sorted(unknown_claims)}."))
        unknown_evidence = set(draft.evidence_ids) - evidence_ids
        if unknown_evidence:
            issues.append(_issue("FABRICATED_EVIDENCE_ID", "HIGH", f"Draft references unknown evidence IDs: {sorted(unknown_evidence)}."))
        unknown_contributions = set(draft.contribution_ids) - contribution_ids
        if unknown_contributions:
            issues.append(_issue("INVENTED_CONTRIBUTION", "BLOCKER", f"Draft references unconfirmed contributions: {sorted(unknown_contributions)}."))

        for claim in claim_plan.claims:
            if claim.claim_id not in draft.claim_ids:
                continue
            if claim.source_type == "LITERATURE":
                if not claim.evidence_ids:
                    issues.append(_issue("UNSUPPORTED_EXTERNAL_CLAIM", "HIGH", "Literature claim has no verified evidence.", claim.claim_id))
                missing = set(claim.evidence_ids) - set(draft.evidence_ids)
                if missing:
                    issues.append(_issue("CLAIM_EVIDENCE_MISMATCH", "HIGH", f"Literature claim is missing evidence: {sorted(missing)}.", claim.claim_id))
                if not draft.citation_keys:
                    issues.append(_issue("CITATION_REQUIRED", "HIGH", "Literature claim has no stable citation key.", claim.claim_id))
                elif not any(set(citation_catalog.get(key, ())) & set(claim.evidence_ids) for key in draft.citation_keys):
                    issues.append(_issue("CITATION_NOT_GROUNDED", "HIGH", "Citation keys do not resolve to this claim's evidence.", claim.claim_id))
            if claim.source_type == "CONFIRMED_CONTRIBUTION" and not set(claim.contribution_ids) <= contribution_ids:
                issues.append(_issue("CONTRIBUTION_CONFLICT", "BLOCKER", "Contribution claim is not backed by the confirmed registry.", claim.claim_id))
            if claim.source_type == "CONFIRMED_CONTRIBUTION" and claim.claim_id in draft.claim_ids and not set(claim.contribution_ids) <= set(draft.contribution_ids):
                issues.append(_issue("CONTRIBUTION_CLAIM_MISMATCH", "BLOCKER", "Contribution claim is missing its registry source.", claim.claim_id))

        known_citation_keys = set(citation_catalog)
        unknown_citations = set(draft.citation_keys) - known_citation_keys
        if unknown_citations:
            issues.append(_issue("FABRICATED_CITATION_KEY", "HIGH", f"Citation keys are not present in Citation Store: {sorted(unknown_citations)}."))

        if minimum_unique_papers is not None:
            coverage = citation_coverage(
                draft.citation_keys,
                citation_catalog,
                evidence,
                minimum_unique_papers=max(INTRODUCTION_MINIMUM_UNIQUE_PAPERS, minimum_unique_papers),
            )
            if not coverage.sufficient:
                issues.append(_issue(
                    "INSUFFICIENT_INTRODUCTION_CITATIONS",
                    "BLOCKER",
                    insufficient_introduction_message(coverage),
                ))

        # A known BibKey is not enough: the binding must still point to the
        # same work as the Evidence used by the claim.  This is intentionally
        # deterministic and catches a valid BibKey attached to the wrong
        # paper as a BLOCKER.
        if citation_bindings:
            evidence_by_id = evidence
            for key in draft.citation_keys:
                binding = citation_bindings.get(key)
                if binding is None:
                    continue
                for evidence_id in citation_catalog.get(key, ()):
                    item = evidence_by_id.get(evidence_id)
                    if item is None:
                        continue
                    identity = CitationIdentity.from_evidence(item)
                    if identity is not None and identity.key != binding.identity_key:
                        issues.append(_issue(
                            "CITATION_WRONG_IDENTITY",
                            "BLOCKER",
                            f"BibKey {key} points to a different bibliographic identity than Evidence {evidence_id}.",
                        ))

        lowered = draft.content.casefold()
        for word in ("groundbreaking", "revolutionary", "unprecedented"):
            if re.search(rf"\b{word}\b", lowered):
                issues.append(_issue("EXAGGERATED_STYLE", "LOW", f"Avoid unsupported academic superlative: {word}."))

        for fact in facts:
            if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool):
                key_tokens = [token for token in re.findall(r"\w+", fact.key.casefold()) if len(token) > 2]
                if key_tokens and all(token in lowered for token in key_tokens) and str(fact.value) not in lowered:
                    issues.append(_issue("FACT_CONTRADICTION", "BLOCKER", f"Draft may contradict Manuscript Fact {fact.key}."))

        if self.semantic_judge is not None:
            issues.extend(self.semantic_judge.review({
                "draft": draft,
                "claim_plan": claim_plan,
                "evidence": evidence,
                "facts": tuple(facts),
                "confirmed_contributions": tuple(contributions),
            }))

        blocking = {"BLOCKER", "HIGH"}
        return ReviewReport(
            valid=not any(value.severity in blocking for value in issues),
            issues=tuple(issues),
            revision_round=revision_round,
            report_id=f"REVIEW_{secrets.token_hex(8)}",
        )


class SharedManuscriptReviewer:
    """Shared deterministic reviewer for synthesis Skills.

    The reviewer consumes a manuscript snapshot and never writes it. Semantic
    judgement can be injected later without changing the Skill boundary.
    """

    _NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")

    def review_synthesis(
        self,
        draft: SectionDraft,
        *,
        policy: ReviewPolicy,
        manuscript_sections: Mapping[str, str],
        facts: Sequence[ManuscriptFact],
        contributions: Sequence[Contribution],
    ) -> ReviewReport:
        issues: list[ReviewIssue] = []
        allowed_contributions = {
            value.contribution_id
            for value in contributions
            if value.status == "confirmed" and value.confirmed_by_user
        }
        unknown_contributions = set(draft.contribution_ids) - allowed_contributions
        if unknown_contributions:
            issues.append(_issue(
                "INVENTED_CONTRIBUTION",
                "BLOCKER",
                f"Draft references unconfirmed contributions: {sorted(unknown_contributions)}.",
            ))
        method_results = " ".join(
            manuscript_sections.get(name, "")
            for name in ("method", "experiment", "results")
        ).casefold()
        for contribution in contributions:
            if contribution.contribution_id not in draft.contribution_ids:
                continue
            tokens = {
                value
                for value in re.findall(r"\w+", contribution.statement.casefold())
                if len(value) > 3
            }
            if tokens and not tokens & set(re.findall(r"\w+", method_results)):
                issues.append(_issue(
                    "CONTRIBUTION_NOT_SUPPORTED_BY_RESULTS",
                    "HIGH",
                    f"Contribution {contribution.contribution_id} is not evidenced by Method/Results.",
                    f"contribution:{contribution.contribution_id}",
                ))
        if draft.evidence_ids:
            issues.append(_issue(
                "UNEXPECTED_EXTERNAL_EVIDENCE",
                "BLOCKER",
                f"{policy} synthesis must not introduce external Evidence IDs.",
            ))
        if draft.citation_keys:
            issues.append(_issue(
                "CITATION_NOT_ALLOWED",
                "BLOCKER",
                f"{policy} synthesis has citation_keys although Citation is disabled.",
            ))

        source_text = "\n".join(manuscript_sections.values())
        allowed_numbers = set(self._NUMBER.findall(source_text))
        for value in facts:
            if value.value is not None:
                allowed_numbers.update(self._NUMBER.findall(str(value.value)))
        new_numbers = set(self._NUMBER.findall(draft.content)) - allowed_numbers
        if new_numbers:
            issues.append(_issue(
                "NEW_MANUSCRIPT_RESULT",
                "BLOCKER",
                f"Draft introduces numeric results absent from the manuscript: {sorted(new_numbers)}.",
            ))

        lowered = draft.content.casefold()
        for fact in facts:
            if not isinstance(fact.value, (int, float)) or isinstance(fact.value, bool):
                continue
            key_tokens = [token for token in re.findall(r"\w+", fact.key.casefold()) if len(token) > 2]
            if key_tokens and all(token in lowered for token in key_tokens) and str(fact.value) not in lowered:
                issues.append(_issue("FACT_CONTRADICTION", "BLOCKER", f"Draft may contradict Manuscript Fact {fact.key}."))

        if policy in {"CONCLUSION", "ABSTRACT"}:
            for word in ("significantly", "significant", "显著", "proves", "证明"):
                if re.search(rf"\b{re.escape(word)}\b", lowered) and word.casefold() not in source_text.casefold():
                    issues.append(_issue("UNSUPPORTED_STRONG_CLAIM", "HIGH", f"Strong result wording is not present in source manuscript: {word}."))
        if policy == "ABSTRACT" and len(draft.content.split()) > 350:
            issues.append(_issue("ABSTRACT_TOO_VERBOSE", "LOW", "Abstract exceeds the concise V1 review budget."))

        return ReviewReport(
            valid=not any(value.severity in {"BLOCKER", "HIGH"} for value in issues),
            issues=tuple(issues),
            revision_round=0,
            report_id=f"REVIEW_{secrets.token_hex(8)}",
        )
