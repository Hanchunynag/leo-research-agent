"""Research-only support-claim Skill; it never creates a manuscript Patch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from app.scholar.models import EvidencePack
from app.scholar.citation import CitationResolutionService
from app.scholar.project import ScholarProjectStore
from app.scholar.research import ResearchCapabilityService
from app.scholar.writing.models import ResearchNeed
from app.scholar.writing.runtime import CapabilityProfile, SkillRuntimeError
from app.scholar.writing.service import MetadataCitationResolver, ResearchDelegate


ClaimSupportStatus = str


@dataclass(frozen=True, slots=True)
class SupportClaimRequest:
    request_id: str
    project_id: str
    claim: str
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.project_id.strip() or not self.claim.strip():
            raise ValueError("SupportClaimRequest request_id/project_id/claim 不能为空。")


@dataclass(frozen=True, slots=True)
class ClaimSupportSubclaim:
    subclaim_id: str
    text: str
    support_status: ClaimSupportStatus
    supporting_evidence_ids: tuple[str, ...] = ()
    counter_evidence_ids: tuple[str, ...] = ()
    unresolved: bool = False


@dataclass(frozen=True, slots=True)
class ClaimSupportResult:
    original_claim: str
    normalized_claim: str
    subclaims: tuple[ClaimSupportSubclaim, ...]
    support_status: ClaimSupportStatus
    supporting_evidence: tuple[Mapping[str, Any], ...] = ()
    counter_evidence: tuple[Mapping[str, Any], ...] = ()
    qualifying_evidence: tuple[Mapping[str, Any], ...] = ()
    coverage: float | None = None
    unresolved: tuple[str, ...] = ()
    citation_requirements: tuple[Mapping[str, Any], ...] = ()
    citation_bindings: tuple[Any, ...] = ()
    evidence_packs: tuple[EvidencePack, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def creates_draft_patch(self) -> bool:
        return False


class ClaimNormalizer(Protocol):
    def normalize(self, claim: str) -> tuple[str, tuple[str, ...]]: ...


class ConservativeClaimNormalizer:
    """Only cleans whitespace and splits explicit conjunctions; it adds no claim."""

    def normalize(self, claim: str) -> tuple[str, tuple[str, ...]]:
        normalized = re.sub(r"\s+", " ", claim).strip()
        pieces = tuple(
            value.strip(" ，,;；")
            for value in re.split(r"\s+(?:and|以及|并且)\s+", normalized, flags=re.IGNORECASE)
            if value.strip(" ，,;；")
        )
        return normalized, pieces or (normalized,)


class ClaimSupportService:
    """Shared Research Capability adapter for the SUPPORT_CLAIM task."""

    skill_name = "support-claim"

    def __init__(
        self,
        research: ResearchCapabilityService,
        *,
        delegate: ResearchDelegate | None = None,
        normalizer: ClaimNormalizer | None = None,
        citation_resolver: MetadataCitationResolver | None = None,
        project_store: ScholarProjectStore | None = None,
        citation_service: CitationResolutionService | None = None,
    ) -> None:
        self.research = research
        self.normalizer = normalizer or ConservativeClaimNormalizer()
        self.citation_resolver = citation_resolver or MetadataCitationResolver()
        self.project_store = project_store
        self.citation_service = citation_service or (
            CitationResolutionService(project_store.project_root, project_store=project_store)
            if project_store is not None
            else None
        )
        self.capabilities = CapabilityProfile(
            skill_name=self.skill_name,
            allowed=frozenset({"research.local", "research.web", "evidence.read", "citation.resolve"}),
        )
        self.delegate = delegate or ResearchDelegate(research, capabilities=self.capabilities)

    def support_claim(self, request: SupportClaimRequest) -> ClaimSupportResult:
        if self.project_store is not None and request.project_id != self.project_store.project_id:
            raise SkillRuntimeError("PROJECT_CONFLICT", "Support Claim 不属于当前 Scholar Project。")
        self.capabilities.require("LOCAL_RESEARCH")
        normalized, pieces = self.normalizer.normalize(request.claim)
        needs = tuple(
            ResearchNeed(
                need_id=f"subclaim_{index}",
                rhetorical_move="claim_support",
                query=piece,
                target_claims=(piece,),
                purpose="support_claim",
                preferred_section_types=(),
            )
            for index, piece in enumerate(pieces, 1)
        )
        try:
            # The CrewAI Research Agent may already have completed the
            # bounded ResearchCapability call.  This private, request-scoped
            # handoff avoids a second search while keeping EvidencePack
            # assessment in this shared Skill service.  It is never persisted
            # as business metadata.
            precomputed = request.metadata.get("_precomputed_evidence_packs")
            if isinstance(precomputed, (list, tuple)) and all(
                isinstance(value, EvidencePack) for value in precomputed
            ):
                packs = tuple(precomputed)
                if len(packs) != len(pieces):
                    raise SkillRuntimeError(
                        "RESEARCH_FAILED",
                        "Research Subagent 未覆盖全部 Support Claim 子断言。",
                    )
            else:
                packs = self.delegate.research_needs(
                    request.request_id,
                    needs,
                    request_metadata=request.metadata,
                )
        except Exception as error:
            raise SkillRuntimeError("RESEARCH_FAILED", str(error)) from error

        for pack in packs:
            for evidence in pack.evidence:
                if not evidence.get("evidence_id") or (
                    evidence.get("source_type") != "WEB_LITERATURE"
                    and (not evidence.get("paper_id") or not evidence.get("section_id"))
                ):
                    raise SkillRuntimeError(
                        "EVIDENCE_VALIDATION_ERROR",
                        "Support Claim 只接受带 paper/section provenance 的 Verified Evidence。",
                    )
                if evidence.get("source_type") != "WEB_LITERATURE" and not evidence.get("chunk_id") and not evidence.get("block_id"):
                    raise SkillRuntimeError(
                        "EVIDENCE_VALIDATION_ERROR",
                        "Verified Evidence 缺少 chunk/block locator。",
                    )

        subclaims: list[ClaimSupportSubclaim] = []
        support_items: dict[str, Mapping[str, Any]] = {}
        counter_items: dict[str, Mapping[str, Any]] = {}
        qualifying_items: dict[str, Mapping[str, Any]] = {}
        citation_requirements: list[Mapping[str, Any]] = []
        citation_bindings: dict[str, Any] = {}
        unresolved: list[str] = []
        for index, (piece, pack) in enumerate(zip(pieces, packs, strict=False), 1):
            claim = pack.claims[0] if pack.claims else {}
            status = str(claim.get("status") or "unresolved")
            evidence_ids = tuple(str(value) for value in claim.get("evidence_ids", ()) if value)
            support_type = str(claim.get("support_type") or "")
            if status in {"supported", "partially_supported"} and support_type not in {"contradict", "contradicted"}:
                support_status = "SUPPORTED" if status == "supported" else "PARTIALLY_SUPPORTED"
                for value in pack.evidence:
                    if str(value.get("evidence_id")) in evidence_ids:
                        support_items[str(value["evidence_id"])] = value
            elif status in {"contradicted", "contradictory"} or support_type in {"contradict", "contradicted"}:
                support_status = "CONTRADICTED"
                for value in pack.evidence:
                    if str(value.get("evidence_id")) in evidence_ids:
                        counter_items[str(value["evidence_id"])] = value
            elif status == "qualifies" or support_type == "qualify":
                support_status = "PARTIALLY_SUPPORTED"
                for value in pack.evidence:
                    if str(value.get("evidence_id")) in evidence_ids:
                        qualifying_items[str(value["evidence_id"])] = value
            else:
                support_status = "INSUFFICIENT_EVIDENCE"
                unresolved.append(piece)
            if pack.counter_evidence:
                for value in pack.counter_evidence:
                    evidence_id = str(value.get("evidence_id") or f"counter:{index}")
                    counter_items[evidence_id] = value
                if support_status == "SUPPORTED":
                    support_status = "PARTIALLY_SUPPORTED"
                elif support_status == "INSUFFICIENT_EVIDENCE":
                    support_status = "CONTRADICTED"
            if not evidence_ids and support_status == "INSUFFICIENT_EVIDENCE":
                unresolved.append(piece)
            subclaims.append(
                ClaimSupportSubclaim(
                    subclaim_id=f"{request.request_id}:C{index}",
                    text=piece,
                    support_status=support_status,
                    supporting_evidence_ids=evidence_ids if support_status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} else (),
                    counter_evidence_ids=evidence_ids if support_status == "CONTRADICTED" else (),
                    unresolved=support_status == "INSUFFICIENT_EVIDENCE",
                )
            )
            for value in pack.evidence:
                resolution = (
                    self.citation_service.resolve(value, project_id=self.project_store.project_id)
                    if self.citation_service is not None and self.project_store is not None
                    else None
                )
                if resolution is not None and resolution.binding is not None and resolution.status == "RESOLVED_EXISTING":
                    citation_bindings[resolution.binding.binding_id] = resolution.binding
                elif resolution is not None and resolution.status == "PROPOSED_NEW_ENTRY":
                    citation_requirements.append(
                        {
                            "evidence_id": value.get("evidence_id"),
                            "paper_id": value.get("paper_id"),
                            "identity_key": resolution.identity.key if resolution.identity else None,
                            "reason": "CitationRequirement: bibliography entry requires a Human-approved BibliographyChange",
                            "candidate": resolution.candidate.to_dict() if resolution.candidate else None,
                        }
                    )
                elif self.citation_resolver.resolve(value) is None:
                    citation_requirements.append(
                        {
                            "evidence_id": value.get("evidence_id"),
                            "paper_id": value.get("paper_id"),
                            "identity_key": resolution.requirement.identity_key if resolution and resolution.requirement else None,
                            "reason": resolution.requirement.reason if resolution and resolution.requirement else "verified evidence has no stable BibKey in the current bibliography",
                        }
                    )

        statuses = {value.support_status for value in subclaims}
        if "CONTRADICTED" in statuses and not statuses & {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
            overall = "CONTRADICTED"
        elif statuses and statuses <= {"INSUFFICIENT_EVIDENCE"}:
            overall = "INSUFFICIENT_EVIDENCE"
        elif "PARTIALLY_SUPPORTED" in statuses or "INSUFFICIENT_EVIDENCE" in statuses:
            overall = "PARTIALLY_SUPPORTED"
        else:
            overall = "SUPPORTED"
        coverage = (
            sum(value.support_status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} for value in subclaims) / len(subclaims)
            if subclaims
            else None
        )
        return ClaimSupportResult(
            original_claim=request.claim,
            normalized_claim=normalized,
            subclaims=tuple(subclaims),
            support_status=overall,
            supporting_evidence=tuple(support_items.values()),
            counter_evidence=tuple(counter_items.values())
            + tuple(item for pack in packs for item in pack.counter_evidence),
            qualifying_evidence=tuple(qualifying_items.values()),
            coverage=coverage,
            unresolved=tuple(dict.fromkeys(unresolved)),
            citation_requirements=tuple(citation_requirements),
            citation_bindings=tuple(citation_bindings.values()),
            evidence_packs=packs,
            metadata={
                "local_only": not any(pack.web_used for pack in packs),
                "web_used": any(pack.web_used for pack in packs),
            },
        )
