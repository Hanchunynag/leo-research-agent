"""Manuscript Supervisor 的 Introduction Vertical。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from app.scholar.approval.service import PatchApprovalService
from app.scholar.citation import CitationResolutionService
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.models import Contribution, DraftPatch, EvidencePack, ManuscriptFact, ReviewReport
from app.scholar.project import ScholarProjectStore
from app.scholar.research import ResearchBudget, ResearchCapabilityService, ResearchRequest
from app.scholar.writing.models import (
    CapabilitySet,
    ClaimPlan,
    CitationRequirement,
    PlannedClaim,
    ResearchNeed,
    SectionDraft,
    WritingContext,
    WritingRequest,
    WritingResult,
    deterministic_patch_id,
    deterministic_review_report_id,
)
from app.scholar.writing.reviewer import IntroductionReviewer
from app.scholar.writing.skill import IntroductionSkill


class IntroductionWriter(Protocol):
    def generate(self, context: WritingContext) -> SectionDraft: ...


class CitationResolver(Protocol):
    def resolve(self, evidence: Mapping[str, Any]) -> str | None: ...


class MetadataCitationResolver:
    """Phase 2C compatibility adapter for legacy test/project metadata.

    Real Evidence with bibliographic identity is resolved by
    :class:`CitationResolutionService` before this adapter is consulted.
    """

    def resolve(self, evidence: Mapping[str, Any]) -> str | None:
        metadata = evidence.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        for name in ("bibkey", "bib_key", "citation_key"):
            value = metadata.get(name)
            if value:
                return str(value)
        return None


class ResearchDelegate:
    """只把 Skill 的 ResearchNeed 转成现有 ResearchRequest。"""

    def __init__(self, capability: ResearchCapabilityService, *, capabilities: CapabilitySet | None = None) -> None:
        self.capability = capability
        self.capabilities = capabilities

    def research_needs(
        self,
        request_id: str,
        needs: Sequence[ResearchNeed],
        *,
        workspace_id: str | None = None,
        scope_version: int | None = None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[EvidencePack, ...]:
        packs: list[EvidencePack] = []
        resolved_workspace = workspace_id or self.capability.workspace_id
        resolved_scope = scope_version or self.capability.scope_version
        for need in needs:
            metadata = {**dict(request_metadata or {}), **dict(need.metadata)}
            freshness_mode = str(metadata.get("freshness_mode") or "LOCAL_ONLY")
            budget = ResearchBudget()
            max_web_queries = metadata.get("max_web_queries")
            if isinstance(max_web_queries, int) and not isinstance(max_web_queries, bool):
                budget = replace(budget, max_web_queries=max(0, max_web_queries))
            research_request = ResearchRequest(
                request_id=f"{request_id}:{need.need_id}",
                query=need.query,
                workspace_id=resolved_workspace,
                scope_version=resolved_scope,
                purpose=need.purpose,  # type: ignore[arg-type]
                target_claims=need.target_claims,
                preferred_section_types=need.preferred_section_types,
                freshness_mode=freshness_mode,  # type: ignore[arg-type]
                requested_from=metadata.get("requested_from"),
                requested_to=metadata.get("requested_to"),
                explicit_latest=bool(metadata.get("explicit_latest", False)),
                domain_sensitivity=str(metadata.get("domain_sensitivity") or "unknown"),  # type: ignore[arg-type]
                budget=budget,
                metadata=metadata,
            )
            allow_web = bool(self.capabilities and self.capabilities.allows("WEB_RESEARCH"))
            try:
                pack = self.capability.research(research_request, allow_web=allow_web)
            except TypeError as error:
                # Phase 2C test doubles and legacy capability facades expose the
                # original one-argument contract; preserve that adapter path.
                if "allow_web" not in str(error):
                    raise
                pack = self.capability.research(research_request)
            packs.append(pack)
        return tuple(packs)

    def research(self, request: WritingRequest, needs: Sequence[ResearchNeed]) -> tuple[EvidencePack, ...]:
        return self.research_needs(request.request_id, needs, request_metadata=request.metadata)


class ScholarWritingService:
    """唯一 Introduction 应用编排边界；没有 Manuscript 写权限。"""

    def __init__(
        self,
        project_root: Path,
        research: ResearchCapabilityService,
        writer: IntroductionWriter,
        *,
        project_store: ScholarProjectStore | None = None,
        synchronizer: ManuscriptSynchronizer | None = None,
        skill: IntroductionSkill | None = None,
        delegate: ResearchDelegate | None = None,
        reviewer: IntroductionReviewer | None = None,
        citation_resolver: CitationResolver | None = None,
        citation_service: CitationResolutionService | None = None,
        approval_service: PatchApprovalService | None = None,
        max_revision_rounds: int = 2,
    ) -> None:
        if max_revision_rounds < 0 or max_revision_rounds > 2:
            raise ValueError("Introduction Revision Loop 最多允许 2 轮。")
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.synchronizer = synchronizer or ManuscriptSynchronizer(self.project_root)
        self.research = research
        self.writer = writer
        self.skill = skill or IntroductionSkill(self.project_root / "skills" / "write-introduction" / "SKILL.md")
        self.capabilities = CapabilitySet()
        self.delegate = delegate or ResearchDelegate(research, capabilities=self.capabilities)
        self.reviewer = reviewer or IntroductionReviewer()
        self.citation_resolver = citation_resolver or MetadataCitationResolver()
        self.citation_service = citation_service or CitationResolutionService(
            self.project_root,
            project_store=self.project_store,
        )
        self.approval_service = approval_service or PatchApprovalService(
            self.project_root,
            project_store=self.project_store,
            synchronizer=self.synchronizer,
        )
        self.max_revision_rounds = max_revision_rounds

    def _confirmed_contributions(self, request: WritingRequest) -> tuple[Contribution, ...]:
        all_values = tuple(value for value in self.project_store.list_contributions() if value.status == "confirmed" and value.confirmed_by_user)
        if not request.selected_contribution_ids:
            return all_values
        by_id = {value.contribution_id: value for value in all_values}
        missing = set(request.selected_contribution_ids) - set(by_id)
        if missing:
            raise ValueError(f"CONTRIBUTION_CONFLICT: 未确认或不存在的 Contribution：{sorted(missing)}")
        return tuple(by_id[value] for value in request.selected_contribution_ids)

    @staticmethod
    def _fact_conflicts(content: str, facts: Sequence[ManuscriptFact]) -> tuple[str, ...]:
        lowered = content.casefold()
        conflicts: list[str] = []
        for fact in facts:
            if not isinstance(fact.value, (int, float)) or isinstance(fact.value, bool):
                continue
            tokens = [token for token in fact.key.casefold().replace("_", " ").split() if len(token) > 2]
            if tokens and all(token in lowered for token in tokens) and str(fact.value) not in lowered:
                conflicts.append(fact.key)
        return tuple(conflicts)

    @staticmethod
    def _claim_plan(request: WritingRequest, needs: Sequence[ResearchNeed], packs: Sequence[EvidencePack], contributions: Sequence[Contribution]) -> ClaimPlan:
        claims: list[PlannedClaim] = []
        packs_by_query = {pack.query: pack for pack in packs}
        for index, need in enumerate(needs):
            # ResearchDelegate preserves need order.  The positional fallback
            # also keeps injected delegates from having to reproduce query text
            # exactly when testing or batching requests.
            pack = packs[index] if index < len(packs) else packs_by_query.get(need.query)
            evidence_ids = tuple(
                str(evidence_id)
                for claim in (pack.claims if pack else ())
                if claim.get("status") in {"supported", "partially_supported"}
                for evidence_id in claim.get("evidence_ids", ())
                if evidence_id
            )
            if pack is not None and pack.evidence and not pack.claims:
                evidence_ids = tuple(
                    str(value.get("evidence_id"))
                    for value in pack.evidence
                    if value.get("evidence_id")
                )
            for index, meaning in enumerate(need.target_claims, 1):
                claims.append(PlannedClaim(
                    claim_id=f"{need.need_id}:C{index}",
                    claim_type=need.rhetorical_move,
                    proposed_meaning=meaning,
                    source_type="LITERATURE",
                    evidence_ids=evidence_ids,
                    required=need.required,
                ))
        for contribution in contributions:
            claims.append(PlannedClaim(
                claim_id=f"contribution:{contribution.contribution_id}",
                claim_type="confirmed_contribution",
                proposed_meaning=contribution.statement,
                source_type="CONFIRMED_CONTRIBUTION",
                contribution_ids=(contribution.contribution_id,),
            ))
        return ClaimPlan(request.request_id, tuple(claims), tuple(needs))

    def _citation_catalog(
        self,
        packs: Sequence[EvidencePack],
    ) -> tuple[
        dict[str, tuple[str, ...]],
        tuple[CitationRequirement, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        str | None,
    ]:
        catalog: dict[str, tuple[str, ...]] = {}
        requirements: list[CitationRequirement] = []
        bindings: dict[str, Any] = {}
        changes: dict[str, Any] = {}
        bibliography_hash: str | None = None
        for pack in packs:
            for item in pack.evidence:
                evidence_id = str(item.get("evidence_id") or "")
                resolution = self.citation_service.resolve(item, project_id=self.project_store.project_id)
                if resolution.snapshot is not None:
                    bibliography_hash = resolution.snapshot.content_hash
                if resolution.binding is not None and resolution.bibkey:
                    key = resolution.bibkey
                    catalog[key] = (*catalog.get(key, ()), evidence_id)
                    prior = bindings.get(resolution.binding.binding_id)
                    if prior is None:
                        bindings[resolution.binding.binding_id] = resolution.binding
                    else:
                        bindings[resolution.binding.binding_id] = replace(
                            prior,
                            evidence_ids=tuple(dict.fromkeys((*prior.evidence_ids, evidence_id))),
                        )
                    change = self.citation_service.change(resolution)
                    if change is not None:
                        changes[change.identity_key] = change
                    continue
                # Phase 2C accepted an explicitly supplied BibKey in test
                # doubles and legacy Evidence metadata.  Keep that adapter
                # path, but all real bibliographic identity resolution above
                # is now handled by the project bibliography service.
                legacy_key = self.citation_resolver.resolve(item)
                if legacy_key and resolution.identity is None and resolution.status != "INVALID":
                    catalog[legacy_key] = (*catalog.get(legacy_key, ()), evidence_id)
                else:
                    requirements.append(
                        resolution.requirement
                        or CitationRequirement(
                            evidence_id,
                            str(item.get("paper_id")) if item.get("paper_id") else None,
                        )
                    )
        return catalog, tuple(requirements), tuple(bindings.values()), tuple(changes.values()), bibliography_hash

    @staticmethod
    def _evidence_mapping(packs: Sequence[EvidencePack]) -> dict[str, Mapping[str, Any]]:
        return {
            str(item.get("evidence_id")): item
            for pack in packs
            for item in pack.evidence
            if item.get("evidence_id")
        }

    def _result(self, request: WritingRequest, status: str, **kwargs: Any) -> WritingResult:
        return WritingResult(status=status, request=request, **kwargs)  # type: ignore[arg-type]

    def write_introduction(
        self,
        request: WritingRequest,
        *,
        writer: IntroductionWriter | None = None,
    ) -> WritingResult:
        if request.project_id != self.project_store.project_id:
            return self._result(request, "CONFLICT", error_codes=("PROJECT_CONFLICT",))
        try:
            self.capabilities.require("READ_MANUSCRIPT")
            self.capabilities.require("READ_FACTS")
            self.capabilities.require("READ_CONTRIBUTIONS")
            self.capabilities.require("LOCAL_RESEARCH")
            self.capabilities.require("READ_EVIDENCE")
            self.capabilities.require("RESOLVE_CITATION")
            self.capabilities.require("REVIEW")
            self.capabilities.require("CREATE_DRAFT_PATCH")
        except PermissionError:
            return self._result(request, "FAILED", error_codes=("CAPABILITY_DENIED",))
        # Runtime may replace the profile for this Skill invocation.  Keep the
        # delegate on the same object so WEB_RESEARCH is enforced by the
        # Research boundary rather than by prompt instructions.
        if hasattr(self.delegate, "capabilities"):
            self.delegate.capabilities = self.capabilities
        active_writer = writer or self.writer
        state = self.synchronizer.scan()
        section = state.sections.get("introduction")
        if section is None:
            return self._result(request, "FAILED", error_codes=("MANUSCRIPT_CONFLICT",))
        current = self.synchronizer.read_section(state, "introduction")
        facts = tuple(self.project_store.list_facts())
        try:
            contributions = self._confirmed_contributions(request)
        except ValueError:
            return self._result(request, "CONFLICT", manuscript_hash=section.content_hash, error_codes=("CONTRIBUTION_CONFLICT",))
        fact_conflicts = self._fact_conflicts(current, facts)
        if fact_conflicts:
            return self._result(request, "CONFLICT", manuscript_hash=section.content_hash, error_codes=("FACT_CONFLICT",), warnings=fact_conflicts)

        needs = self.skill.build_research_needs(request)
        try:
            precomputed = request.metadata.get("_precomputed_evidence_packs")
            if isinstance(precomputed, (list, tuple)) and all(
                isinstance(value, EvidencePack) for value in precomputed
            ):
                if len(precomputed) != len(needs):
                    return self._result(
                        request,
                        "FAILED",
                        manuscript_hash=section.content_hash,
                        error_codes=("RESEARCH_FAILED",),
                        warnings=("Research Subagent 未覆盖全部 Introduction ResearchNeed。",),
                    )
                # This is a request-scoped handoff from the isolated Research
                # Subagent.  The shared Writing Service still owns ClaimPlan,
                # citation resolution and review; it simply does not repeat
                # the same ResearchCapability call.
                packs = tuple(precomputed)
            else:
                packs = self.delegate.research(request, needs)
        except Exception:
            return self._result(request, "FAILED", manuscript_hash=section.content_hash, error_codes=("RESEARCH_FAILED",))
        plan = self._claim_plan(request, needs, packs, contributions)
        catalog, citation_requirements, citation_bindings, bibliography_changes, bibliography_hash = self._citation_catalog(packs)
        context = WritingContext(
            request,
            current,
            section.content_hash,
            facts,
            contributions,
            plan,
            packs,
            catalog,
            citation_requirements,
            self.capabilities,
            self.skill.text,
            citation_bindings=citation_bindings,
            bibliography_changes=bibliography_changes,
            bibliography_base_hash=bibliography_hash,
        )
        try:
            draft = active_writer.generate(context)
            if not isinstance(draft, SectionDraft):
                raise TypeError("Introduction Writer 必须返回 SectionDraft。")
            if draft.target_section != "introduction" or draft.base_hash != section.content_hash:
                return self._result(request, "CONFLICT", manuscript_hash=section.content_hash, claim_plan=plan, evidence_packs=packs, error_codes=("STALE_BASE_HASH",))
            if draft.original_content and draft.original_content != current:
                return self._result(request, "CONFLICT", manuscript_hash=section.content_hash, draft=draft, claim_plan=plan, evidence_packs=packs, error_codes=("STALE_BASE_HASH",))
            if not draft.original_content:
                draft = replace(draft, original_content=current)
        except Exception:
            return self._result(request, "FAILED", manuscript_hash=section.content_hash, claim_plan=plan, evidence_packs=packs, error_codes=("DRAFT_GENERATION_FAILED",))

        review: ReviewReport | None = None
        for revision_round in range(self.max_revision_rounds + 1):
            try:
                review = self.reviewer.review(
                    draft,
                    plan,
                    self._evidence_mapping(packs),
                    facts,
                    contributions,
                    catalog,
                    revision_round=revision_round,
                    citation_bindings={
                        str(value.bibkey): value
                        for value in citation_bindings
                        if getattr(value, "bibkey", None)
                    },
                )
                review = replace(
                    review,
                    report_id=deterministic_review_report_id(
                        request.project_id,
                        request.request_id,
                        "introduction",
                        revision_round,
                    ),
                )
            except Exception:
                return self._result(request, "FAILED", manuscript_hash=section.content_hash, draft=draft, claim_plan=plan, evidence_packs=packs, error_codes=("REVIEW_FAILED",))
            if review.valid or revision_round >= self.max_revision_rounds or not callable(getattr(active_writer, "revise", None)):
                break
            context = replace(context, review_report=review)
            draft = active_writer.revise(context, review)  # type: ignore[attr-defined]

        latest = self.synchronizer.scan()
        latest_section = latest.sections.get("introduction")
        if latest_section is None or latest_section.content_hash != section.content_hash:
            return self._result(request, "CONFLICT", manuscript_hash=section.content_hash, draft=draft, claim_plan=plan, evidence_packs=packs, review_report=review, error_codes=("STALE_BASE_HASH",))
        assert review is not None
        patch = DraftPatch(
            patch_id=deterministic_patch_id(
                request.project_id,
                request.request_id,
                "introduction",
            ),
            target_section="introduction",
            base_hash=section.content_hash,
            proposed_content=draft.content,
            used_claim_ids=draft.claim_ids,
            used_evidence_ids=draft.evidence_ids,
            reviewer_status="approved" if review.valid else "needs_user_review",
            project_id=request.project_id,
            citation_keys=draft.citation_keys,
            contribution_ids=draft.contribution_ids,
            review_report_id=review.report_id,
            change_summary=draft.change_summary,
            original_content=draft.original_content or current,
            warnings=draft.warnings,
            bibliography_base_hash=bibliography_hash if (citation_bindings or citation_requirements or bibliography_changes) else None,
            citation_bindings=tuple(citation_bindings),
            bibliography_changes=tuple(bibliography_changes),
            citation_requirements=tuple(citation_requirements),
            citation_binding_ids=tuple(
                str(getattr(value, "binding_id")) for value in citation_bindings
                if getattr(value, "binding_id", None)
            ),
        )
        try:
            self.approval_service.register_patch(
                patch,
                review_report=review,
                source_session_id=request.session_id,
                source_run_id=(
                    str(request.metadata["run_id"])
                    if isinstance(request.metadata.get("run_id"), str)
                    else None
                ),
            )
        except (OSError, ValueError, RuntimeError):
            return self._result(
                request,
                "FAILED",
                manuscript_hash=section.content_hash,
                draft=draft,
                claim_plan=plan,
                evidence_packs=packs,
                review_report=review,
                error_codes=("PATCH_PERSISTENCE_FAILED",),
                warnings=draft.warnings,
            )
        return self._result(
            request,
            "READY" if review.valid else "NEEDS_USER_REVIEW",
            manuscript_hash=section.content_hash,
            draft=draft,
            claim_plan=plan,
            evidence_packs=packs,
            review_report=review,
            patch=patch,
            warnings=draft.warnings,
        )


ManuscriptSupervisor = ScholarWritingService
