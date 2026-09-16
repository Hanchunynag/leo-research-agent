"""Shared Manuscript Synthesis Runtime for Conclusion and Abstract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from app.scholar.approval.service import PatchApprovalService
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.models import Contribution, DraftPatch, ManuscriptFact, ReviewReport
from app.scholar.project import ScholarProjectStore
from app.scholar.writing.models import (
    SectionDraft,
    WritingRequest,
    deterministic_patch_id,
    deterministic_review_report_id,
    WritingResult,
)
from app.scholar.writing.reviewer import SharedManuscriptReviewer
from app.scholar.writing.runtime import SkillDefinition, SkillExecutionContext


@dataclass(frozen=True, slots=True)
class AbstractFactSet:
    problem: str
    method: str
    confirmed_contribution_ids: tuple[str, ...]
    experiment_context: str
    main_numeric_results: tuple[str, ...]
    main_qualitative_result: str
    limitations: str = ""


class SynthesisWriter(Protocol):
    def generate(self, context: SkillExecutionContext) -> SectionDraft: ...


class SynthesisWritingService:
    """One shared Draft/Review/Persist pipeline for Conclusion and Abstract."""

    def __init__(
        self,
        project_root: Path,
        *,
        project_store: ScholarProjectStore | None = None,
        synchronizer: ManuscriptSynchronizer | None = None,
        approval_service: PatchApprovalService | None = None,
        reviewer: SharedManuscriptReviewer | None = None,
        max_revision_rounds: int = 2,
    ) -> None:
        if max_revision_rounds < 0 or max_revision_rounds > 2:
            raise ValueError("Synthesis Revision Loop 最多允许 2 轮。")
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.synchronizer = synchronizer or ManuscriptSynchronizer(self.project_root)
        self.approval_service = approval_service or PatchApprovalService(
            self.project_root,
            project_store=self.project_store,
            synchronizer=self.synchronizer,
        )
        self.reviewer = reviewer or SharedManuscriptReviewer()
        self.max_revision_rounds = max_revision_rounds

    def write(
        self,
        request: WritingRequest,
        writer: SynthesisWriter,
        definition: SkillDefinition,
    ) -> WritingResult:
        if request.task_type not in {"WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            return self._result(request, "FAILED", error_codes=("UNSUPPORTED_TASK",))
        capabilities = definition.capabilities
        try:
            for capability in ("READ_MANUSCRIPT", "READ_FACTS", "READ_CONTRIBUTIONS", "REVIEW", "CREATE_DRAFT_PATCH"):
                capabilities.require(capability)
        except PermissionError:
            return self._result(request, "FAILED", error_codes=("CAPABILITY_DENIED",))

        if request.project_id != self.project_store.project_id:
            return self._result(request, "CONFLICT", error_codes=("PROJECT_CONFLICT",))
        previous = self.project_store.load_manuscript_state()
        # Keep synthesis usable for a newly created project while preserving
        # the PatchApprovalService write boundary. This only creates missing
        # empty template files.
        state = self.synchronizer.ensure_initialized()
        target = state.sections.get(request.target_section)
        if target is None:
            return self._result(request, "FAILED", error_codes=("INSUFFICIENT_MANUSCRIPT_STATE",))
        required = self._required_sections(request.task_type)
        missing = tuple(value for value in required if value not in state.sections)
        if missing:
            return self._result(request, "FAILED", manuscript_hash=target.content_hash, error_codes=("INSUFFICIENT_MANUSCRIPT_STATE",), warnings=missing)
        upstream = set(required) - {request.target_section}
        changed = {
            name
            for name, section in state.sections.items()
            if previous is None
            or name not in previous.sections
            or previous.sections[name].content_hash != section.content_hash
        }
        # A freshly changed section is current content; only its deterministic
        # downstream stale projection blocks synthesis.
        stale = tuple(sorted(upstream & (set(state.stale_sections) - changed)))
        if stale:
            return self._result(request, "CONFLICT", manuscript_hash=target.content_hash, error_codes=("UPSTREAM_SECTION_STALE",), warnings=stale)

        sections = {
            name: self.synchronizer.read_section(state, name)
            for name in state.sections
            if name in required or name == request.target_section
        }
        facts = tuple(self.project_store.list_facts())
        all_contributions = tuple(
            value
            for value in self.project_store.list_contributions()
            if value.status == "confirmed" and value.confirmed_by_user
        )
        if request.selected_contribution_ids:
            by_id = {value.contribution_id: value for value in all_contributions}
            missing_contributions = set(request.selected_contribution_ids) - set(by_id)
            if missing_contributions:
                return self._result(
                    request,
                    "CONFLICT",
                    manuscript_hash=target.content_hash,
                    error_codes=("CONTRIBUTION_CONFLICT",),
                    warnings=tuple(sorted(missing_contributions)),
                )
            contributions = tuple(by_id[value] for value in request.selected_contribution_ids)
        else:
            contributions = all_contributions
        fact_conflicts = self._fact_result_conflicts(sections.get("results", ""), facts)
        if fact_conflicts:
            return self._result(
                request,
                "CONFLICT",
                manuscript_hash=target.content_hash,
                error_codes=("FACT_CONFLICT",),
                warnings=fact_conflicts,
            )
        fact_set = self._abstract_fact_set(sections, facts, contributions) if request.task_type == "WRITE_ABSTRACT" else None
        if request.task_type == "WRITE_ABSTRACT" and fact_set is None:
            return self._result(request, "FAILED", manuscript_hash=target.content_hash, error_codes=("INSUFFICIENT_MANUSCRIPT_STATE",))
        context = SkillExecutionContext(
            project_id=request.project_id,
            task_type=request.task_type,
            target_section=request.target_section,
            session_id=request.session_id,
            user_instruction=request.instruction,
            current_section=sections[request.target_section],
            current_hash=target.content_hash,
            manuscript_sections=sections,
            facts=facts,
            contributions=contributions,
            abstract_fact_set=fact_set,
            skill_metadata={"skill": definition.name, "required_sections": required},
            capabilities=capabilities,
        )
        try:
            draft = writer.generate(context)
            draft = self._validate_draft(draft, request, target.content_hash, sections[request.target_section])
        except Exception:
            return self._result(request, "FAILED", manuscript_hash=target.content_hash, error_codes=("DRAFT_GENERATION_FAILED",))

        report: ReviewReport | None = None
        for revision_round in range(self.max_revision_rounds + 1):
            try:
                report = self.reviewer.review_synthesis(
                    draft,
                    policy="CONCLUSION" if request.task_type == "WRITE_CONCLUSION" else "ABSTRACT",
                    manuscript_sections=sections,
                    facts=facts,
                    contributions=contributions,
                )
                report = replace(
                    report,
                    revision_round=revision_round,
                    report_id=deterministic_review_report_id(
                        request.project_id,
                        request.request_id,
                        request.target_section,
                        revision_round,
                    ),
                )
            except Exception:
                return self._result(request, "FAILED", manuscript_hash=target.content_hash, draft=draft, review_report=report, error_codes=("REVIEW_FAILED",))
            if report.valid or revision_round >= self.max_revision_rounds or not callable(getattr(writer, "revise", None)):
                break
            draft = writer.revise(context, report)  # type: ignore[attr-defined]
            draft = self._validate_draft(draft, request, target.content_hash, sections[request.target_section])

        latest = self.synchronizer.scan(previous=state)
        latest_target = latest.sections.get(request.target_section)
        if latest_target is None or latest_target.content_hash != target.content_hash:
            return self._result(request, "CONFLICT", manuscript_hash=target.content_hash, draft=draft, review_report=report, error_codes=("STALE_BASE_HASH",))
        assert report is not None
        patch = DraftPatch(
            patch_id=deterministic_patch_id(
                request.project_id,
                request.request_id,
                request.target_section,
            ),
            target_section=request.target_section,
            base_hash=target.content_hash,
            proposed_content=draft.content,
            used_claim_ids=draft.claim_ids,
            used_evidence_ids=(),
            reviewer_status="approved" if report.valid else "needs_user_review",
            project_id=request.project_id,
            citation_keys=(),
            contribution_ids=draft.contribution_ids,
            review_report_id=report.report_id,
            change_summary=draft.change_summary,
            original_content=draft.original_content or sections[request.target_section],
            warnings=draft.warnings,
        )
        if not report.valid and any(value.severity in {"BLOCKER", "HIGH"} for value in report.issues):
            # Store the proposal for Human Review even when it cannot yet pass Apply Gate.
            pass
        try:
            self.approval_service.register_patch(
                patch,
                review_report=report,
                source_session_id=request.session_id,
                source_run_id=str(request.metadata["run_id"]) if isinstance(request.metadata.get("run_id"), str) else None,
            )
        except (OSError, ValueError, RuntimeError):
            return self._result(request, "FAILED", manuscript_hash=target.content_hash, draft=draft, review_report=report, error_codes=("PATCH_PERSISTENCE_FAILED",))
        return self._result(
            request,
            "READY" if report.valid else "NEEDS_USER_REVIEW",
            manuscript_hash=target.content_hash,
            draft=draft,
            review_report=report,
            patch=patch,
            warnings=draft.warnings,
        )

    @staticmethod
    def _required_sections(task_type: str) -> tuple[str, ...]:
        if task_type == "WRITE_CONCLUSION":
            return ("method", "results", "conclusion")
        return ("introduction", "method", "results", "conclusion", "abstract")

    @staticmethod
    def _validate_draft(
        draft: SectionDraft,
        request: WritingRequest,
        base_hash: str,
        original: str,
    ) -> SectionDraft:
        if not isinstance(draft, SectionDraft):
            raise TypeError("Synthesis Writer 必须返回 SectionDraft。")
        if draft.target_section != request.target_section or draft.base_hash != base_hash:
            raise ValueError("STALE_BASE_HASH")
        if draft.original_content and draft.original_content != original:
            raise ValueError("STALE_BASE_HASH")
        return replace(draft, original_content=original) if not draft.original_content else draft

    @staticmethod
    def _abstract_fact_set(
        sections: dict[str, str],
        facts: tuple[ManuscriptFact, ...],
        contributions: tuple[Contribution, ...],
    ) -> AbstractFactSet | None:
        if not sections.get("results", "").strip() or not sections.get("method", "").strip():
            return None
        results = sections["results"]
        numeric = tuple(dict.fromkeys(SharedManuscriptReviewer._NUMBER.findall(results)))
        problem = sections.get("introduction", "").strip()[:1200]
        method = sections.get("method", "").strip()[:1600]
        experiment = sections.get("experiment", results).strip()[:1200]
        qualitative = results.strip()[:1600]
        return AbstractFactSet(
            problem=problem,
            method=method,
            confirmed_contribution_ids=tuple(value.contribution_id for value in contributions),
            experiment_context=experiment,
            main_numeric_results=numeric,
            main_qualitative_result=qualitative,
            limitations=sections.get("conclusion", "").strip()[:800],
        )

    @staticmethod
    def _fact_result_conflicts(results: str, facts: tuple[ManuscriptFact, ...]) -> tuple[str, ...]:
        lowered = results.casefold()
        conflicts: list[str] = []
        for fact in facts:
            if not isinstance(fact.value, (int, float, str)) or isinstance(fact.value, bool):
                continue
            tokens = [value for value in fact.key.casefold().replace("_", " ").split() if len(value) > 2]
            if tokens and all(value in lowered for value in tokens) and str(fact.value).casefold() not in lowered:
                conflicts.append(fact.key)
        return tuple(conflicts)

    @staticmethod
    def _result(request: WritingRequest, status: str, **kwargs: Any) -> WritingResult:
        return WritingResult(status=status, request=request, **kwargs)  # type: ignore[arg-type]
