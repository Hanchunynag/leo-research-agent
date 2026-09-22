"""Scholar Writing Runtime 的框架无关 Contract。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Mapping

from app.scholar.models import Contribution, DraftPatch, EvidencePack, ManuscriptFact, ReviewReport
from app.scholar.citation.models import CitationBinding, CitationRequirement


WritingMode = Literal["WRITE", "REVISE"]
WritingStatus = Literal["READY", "NEEDS_USER_REVIEW", "CONFLICT", "FAILED"]
TaskType = Literal[
    "WRITE_INTRODUCTION",
    "SUPPORT_CLAIM",
    "WRITE_CONCLUSION",
    "WRITE_ABSTRACT",
]


def deterministic_patch_id(project_id: str, request_id: str, target_section: str) -> str:
    """Return the idempotency key for one writing request's Patch proposal.

    A resumed Scholar run reuses its ``request_id``/Run ID. Keeping the
    proposal id stable lets ProjectStore's immutable write provide the final
    idempotency guard if a process dies after the Domain write but before the
    framework checkpoint is committed.
    """

    payload = "\x1f".join((project_id, request_id, target_section)).encode("utf-8")
    return f"PATCH_{sha256(payload).hexdigest()[:24]}"


def deterministic_review_report_id(
    project_id: str,
    request_id: str,
    target_section: str,
    revision_round: int,
) -> str:
    """Return the stable review artifact id used by a Patch-producing run."""

    payload = "\x1f".join(
        (project_id, request_id, target_section, str(revision_round))
    ).encode("utf-8")
    return f"REVIEW_{sha256(payload).hexdigest()[:24]}"
ClaimSource = Literal[
    "LITERATURE",
    "MANUSCRIPT_FACT",
    "CONFIRMED_CONTRIBUTION",
    "USER_INSTRUCTION",
]


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Skill 的代码级能力集合，而不是 Markdown Prompt 中的约定。"""

    skill_name: str = "write-introduction"
    allowed: frozenset[str] = frozenset()

    _ALIASES = {
        "READ_MANUSCRIPT": "manuscript.read",
        "READ_FACTS": "facts.read",
        "READ_CONTRIBUTIONS": "contributions.read",
        "LOCAL_RESEARCH": "research.local",
        "READ_EVIDENCE": "evidence.read",
        "RESOLVE_CITATION": "citation.resolve",
        "REVIEW": "reviewer.run",
        "CREATE_DRAFT_PATCH": "patch.create",
        "WEB_RESEARCH": "research.web",
        "WRITE_MANUSCRIPT": "manuscript.write",
        "WRITE_FACTS": "facts.write",
        "WRITE_CONTRIBUTIONS": "contributions.write",
        "READ_BIBLIOGRAPHY": "bibliography.read",
        "PROPOSE_BIB_ENTRY": "bibliography.propose",
        "WRITE_BIBLIOGRAPHY": "bibliography.write",
    }

    def allows(self, capability: str) -> bool:
        return capability in self.allowed or self._ALIASES.get(capability) in self.allowed

    def require(self, capability: str) -> None:
        if not self.allows(capability):
            raise PermissionError(
                f"CAPABILITY_DENIED: {self.skill_name} 不允许 {capability}。"
            )


@dataclass(frozen=True, slots=True)
class CapabilitySet(CapabilityProfile):
    """Introduction Vertical 的向后兼容能力别名。"""

    skill_name: str = "write-introduction"
    allowed: frozenset[str] = frozenset(
        {
            "workspace.read",
            "manuscript.read",
            "facts.read",
            "contributions.read",
            "research.local",
            "research.web",
            "evidence.read",
            "citation.resolve",
            "bibliography.read",
            "bibliography.propose",
            "reviewer.run",
            "patch.create",
        }
    )

@dataclass(frozen=True, slots=True)
class WritingRequest:
    request_id: str
    project_id: str
    instruction: str
    target_section: str = "introduction"
    session_id: str | None = None
    mode: WritingMode = "REVISE"
    selected_contribution_ids: tuple[str, ...] = ()
    focus: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_type: TaskType | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.project_id.strip():
            raise ValueError("WritingRequest request_id/project_id 不能为空。")
        if not self.instruction.strip():
            raise ValueError("WritingRequest instruction 不能为空。")
        section_to_task = {
            "introduction": "WRITE_INTRODUCTION",
            "conclusion": "WRITE_CONCLUSION",
            "abstract": "WRITE_ABSTRACT",
        }
        if self.target_section not in section_to_task:
            raise ValueError("WritingRequest target_section 必须是 introduction/conclusion/abstract。")
        if self.mode not in {"WRITE", "REVISE"}:
            raise ValueError("mode 必须是 WRITE 或 REVISE。")
        inferred = section_to_task[self.target_section]
        if self.task_type is None:
            object.__setattr__(self, "task_type", inferred)
        elif self.task_type != inferred:
            raise ValueError("task_type 与 target_section 不一致。")


@dataclass(frozen=True, slots=True)
class ResearchNeed:
    need_id: str
    rhetorical_move: str
    query: str
    target_claims: tuple[str, ...]
    purpose: str = "background"
    preferred_section_types: tuple[str, ...] = ("Introduction", "Related Work")
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannedClaim:
    claim_id: str
    claim_type: str
    proposed_meaning: str
    source_type: ClaimSource
    evidence_ids: tuple[str, ...] = ()
    contribution_ids: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True, slots=True)
class ClaimPlan:
    request_id: str
    claims: tuple[PlannedClaim, ...]
    research_needs: tuple[ResearchNeed, ...]

    @property
    def claim_ids(self) -> frozenset[str]:
        return frozenset(value.claim_id for value in self.claims)

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(
            evidence_id
            for claim in self.claims
            for evidence_id in claim.evidence_ids
        )


@dataclass(frozen=True, slots=True)
class SectionDraft:
    target_section: str
    base_hash: str
    content: str
    claim_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()
    citation_requirements: tuple[CitationRequirement, ...] = ()
    contribution_ids: tuple[str, ...] = ()
    change_summary: str = ""
    warnings: tuple[str, ...] = ()
    original_content: str = ""
    citation_binding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_section not in {"introduction", "conclusion", "abstract"}:
            raise ValueError("SectionDraft target_section 不支持。")
        if not self.base_hash.strip() or not self.content.strip():
            raise ValueError("SectionDraft 必须携带 base_hash 和 content。")


@dataclass(frozen=True, slots=True)
class WritingResult:
    status: WritingStatus
    request: WritingRequest
    manuscript_hash: str | None = None
    draft: SectionDraft | None = None
    claim_plan: ClaimPlan | None = None
    evidence_packs: tuple[EvidencePack, ...] = ()
    review_report: ReviewReport | None = None
    patch: DraftPatch | None = None
    error_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WritingContext:
    request: WritingRequest
    current_introduction: str
    current_hash: str
    facts: tuple[ManuscriptFact, ...]
    contributions: tuple[Contribution, ...]
    claim_plan: ClaimPlan
    evidence_packs: tuple[EvidencePack, ...]
    citation_catalog: Mapping[str, tuple[str, ...]]
    citation_requirements: tuple[CitationRequirement, ...]
    capabilities: CapabilitySet
    skill_text: str
    citation_bindings: tuple[CitationBinding, ...] = ()
    bibliography_changes: tuple[Any, ...] = ()
    bibliography_base_hash: str | None = None
    citation_coverage: Mapping[str, Any] = field(default_factory=dict)
    review_report: ReviewReport | None = None

    def writer_mapping(self) -> dict[str, Any]:
        """Return a bounded, citation-preserving payload for the LLM Writer.

        ``public_mapping`` is the domain-facing projection and intentionally
        retains complete evidence metadata for deterministic validation.  It
        is not a safe wire payload: three research packs can contain repeated
        chunks, full abstracts and pipeline diagnostics, easily exceeding the
        provider context window.  The Writer only needs a representative
        excerpt plus stable IDs/identity metadata to draft and cite prose;
        the full objects remain in the Writing Runtime as the authority.
        """

        raw_evidence: list[dict[str, Any]] = []
        for pack in self.evidence_packs:
            raw_evidence.extend(
                dict(value) for value in pack.evidence if isinstance(value, Mapping)
            )

        def identity(value: Mapping[str, Any]) -> str:
            metadata = value.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            for key in ("paper_id", "canonical_id", "doi", "arxiv_id", "work_id", "document_id"):
                candidate = value.get(key) or metadata.get(key)
                if candidate not in (None, "", ()):
                    return f"{key}:{candidate}"
            return str(value.get("evidence_id") or "unknown")

        # First keep one excerpt per paper so the 5-paper contract is visible
        # to the model even when retrieval returned many chunks from one work.
        selected: list[dict[str, Any]] = []
        selected_identities: set[str] = set()
        for value in raw_evidence:
            key = identity(value)
            if key in selected_identities:
                continue
            selected_identities.add(key)
            selected.append(value)
            if len(selected) >= 12:
                break
        # Add a small number of supporting excerpts after paper coverage has
        # been established.  This helps the model bind different claims while
        # keeping the request comfortably below common gateway limits.
        selected_ids = {str(value.get("evidence_id") or "") for value in selected}
        for value in raw_evidence:
            evidence_id = str(value.get("evidence_id") or "")
            if evidence_id in selected_ids:
                continue
            selected.append(value)
            selected_ids.add(evidence_id)
            if len(selected) >= 18:
                break

        compact_evidence: list[dict[str, Any]] = []
        for value in selected:
            metadata = value.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            compact_metadata = {
                key: metadata[key]
                for key in (
                    "title",
                    "authors",
                    "year",
                    "doi",
                    "arxiv_id",
                    "venue",
                    "verification",
                )
                if metadata.get(key) is not None
            }
            excerpt = value.get("content") or value.get("text") or ""
            compact_evidence.append(
                {
                    key: value[key]
                    for key in (
                        "evidence_id",
                        "paper_id",
                        "canonical_id",
                        "source_locator",
                        "bibkey",
                        "citation_key",
                        "page_start",
                        "page_end",
                    )
                    if value.get(key) is not None
                }
                | {
                    "content": str(excerpt)[:1800],
                    "metadata": compact_metadata,
                }
            )

        return {
            "request": self.request,
            "current_introduction": self.current_introduction[:6000],
            "current_hash": self.current_hash,
            "facts": self.facts,
            "confirmed_contributions": self.contributions,
            "claim_plan": self.claim_plan,
            "evidence": compact_evidence,
            "citation_catalog": dict(self.citation_catalog),
            "citation_requirements": self.citation_requirements,
            "citation_bindings": self.citation_bindings,
            "bibliography_changes": self.bibliography_changes,
            "bibliography_base_hash": self.bibliography_base_hash,
            "citation_coverage": dict(self.citation_coverage),
            "skill_text": self.skill_text[:6000],
            "capabilities": self.capabilities,
            "review_report": self.review_report,
            "writer_payload_limits": {
                "evidence_items": len(compact_evidence),
                "excerpt_chars": 1800,
                "source": "bounded_writer_projection",
            },
        }

    def public_mapping(self) -> dict[str, Any]:
        """限制 Writer 可见内容；不暴露 Service、Store 或底层 RAG 对象。"""

        evidence: list[dict[str, Any]] = []
        for pack in self.evidence_packs:
            evidence.extend(dict(value) for value in pack.evidence)
        return {
            "request": self.request,
            "current_introduction": self.current_introduction,
            "current_hash": self.current_hash,
            "facts": self.facts,
            "confirmed_contributions": self.contributions,
            "claim_plan": self.claim_plan,
            "evidence": tuple(evidence),
            "citation_catalog": dict(self.citation_catalog),
            "citation_requirements": self.citation_requirements,
            "citation_bindings": self.citation_bindings,
            "bibliography_changes": self.bibliography_changes,
            "bibliography_base_hash": self.bibliography_base_hash,
            "citation_coverage": dict(self.citation_coverage),
            "skill_text": self.skill_text,
            "capabilities": self.capabilities,
            "review_report": self.review_report,
        }
