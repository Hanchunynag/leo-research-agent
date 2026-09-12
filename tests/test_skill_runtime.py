from __future__ import annotations

from pathlib import Path
from typing import Any

from app.scholar import (
    Contribution,
    EvidencePack,
    ManuscriptFact,
    ScholarProjectStore,
)
from app.scholar.writing import (
    ClaimSupportService,
    SectionDraft,
    ScholarSkillRuntime,
    SkillRegistry,
    SupportClaimRequest,
    SynthesisWritingService,
    TaskRouter,
    WritingRequest,
)
from app.scholar.writing.runtime import SkillExecutionContext
from app.scholar.approval import PatchApprovalRequest
from app.scholar.writing.service import ScholarWritingService
from tests.test_introduction_vertical import FakeCapability, FakeDelegate, FakeWriter, evidence_pack
from tests.test_scholar_foundation import make_project


class FakeResearch:
    workspace_id = "fixture"
    scope_version = 1

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def research(self, request: Any) -> EvidencePack:
        self.calls.append(request)
        claim = request.target_claims[0] if request.target_claims else request.query
        return EvidencePack(
            request_id=request.request_id,
            query=request.query,
            claims=({
                "claim_id": f"{request.request_id}:C1",
                "text": claim,
                "status": "supported",
                "support_type": "support",
                "evidence_ids": ("E1",),
            },),
            evidence=({
                "evidence_id": "E1",
                "paper_id": "P1",
                "section_id": "P1_s1",
                "chunk_id": "P1_c1",
                "text": "Verified local evidence.",
                "metadata": {"bibkey": "Doe2024"},
            },),
            coverage=1.0,
            local_coverage=1.0,
        )


class CounterResearch(FakeResearch):
    def research(self, request: Any) -> EvidencePack:
        self.calls.append(request)
        return EvidencePack(
            request_id=request.request_id,
            query=request.query,
            claims=({"claim_id": "C1", "status": "unresolved", "evidence_ids": ()},),
            counter_evidence=({
                "evidence_id": "E-counter",
                "paper_id": "P2",
                "section_id": "P2_s1",
                "chunk_id": "P2_c1",
                "text": "Counter evidence.",
            },),
        )


def add_synthesis_sections(root: Path) -> None:
    main = root / "main.tex"
    main.write_text(
        main.read_text(encoding="utf-8")
        + "\n\\input{sections/method}\n"
        + "\\input{sections/results}\n"
        + "\\input{sections/conclusion}\n"
        + "\\input{sections/abstract}\n",
        encoding="utf-8",
    )
    (root / "sections" / "method.tex").write_text(
        "We implement the correction method.\n", encoding="utf-8"
    )
    (root / "sections" / "results.tex").write_text(
        "The experiment reports accuracy 91% and error 0.5 m.\n", encoding="utf-8"
    )
    (root / "sections" / "conclusion.tex").write_text(
        "Existing conclusion.\n", encoding="utf-8"
    )
    (root / "sections" / "abstract.tex").write_text(
        "Existing abstract.\n", encoding="utf-8"
    )


class SynthesisWriter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def generate(self, context: SkillExecutionContext) -> SectionDraft:
        self.calls += 1
        contribution_ids = tuple(value.contribution_id for value in context.contributions)
        return SectionDraft(
            target_section=context.target_section or "conclusion",
            base_hash=context.current_hash or "",
            content=self.content,
            contribution_ids=contribution_ids,
            change_summary="Synthesize current manuscript findings.",
            original_content=context.current_section,
        )


def test_registry_profiles_and_router_are_skill_only(tmp_path: Path) -> None:
    registry = SkillRegistry.default(tmp_path)
    intro = registry.get("WRITE_INTRODUCTION")
    support = registry.get("SUPPORT_CLAIM")
    conclusion = registry.get("WRITE_CONCLUSION")
    abstract = registry.get("WRITE_ABSTRACT")

    assert intro.capabilities.allows("CREATE_DRAFT_PATCH")
    assert intro.capabilities.allows("READ_MANUSCRIPT")
    assert support.capabilities.allows("LOCAL_RESEARCH")
    assert not support.capabilities.allows("CREATE_DRAFT_PATCH")
    assert not conclusion.capabilities.allows("LOCAL_RESEARCH")
    assert not abstract.capabilities.allows("RESOLVE_CITATION")
    assert TaskRouter().route(task_type="WRITE_ABSTRACT").confidence == 1.0
    assert TaskRouter().route(instruction="请帮我处理一下论文").needs_user_choice is True


def test_support_claim_returns_verified_result_without_patch_or_manuscript_write(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    research = FakeResearch()
    service = ClaimSupportService(research, project_store=store)  # type: ignore[arg-type]
    runtime = ScholarSkillRuntime(root, project_store=store, support_claim=service)
    original = (root / "sections" / "introduction.tex").read_text(encoding="utf-8")

    result = runtime.execute(
        SupportClaimRequest("SUP-1", store.project_id, "SGP4 ephemeris error affects positioning")
    )

    assert result.task_type == "SUPPORT_CLAIM"
    assert result.value.support_status == "SUPPORTED"
    assert result.value.supporting_evidence[0]["evidence_id"] == "E1"
    assert result.value.creates_draft_patch is False
    assert research.calls and research.calls[0].purpose == "support_claim"
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") == original
    assert store.get_manuscript_state_projection() is None


def test_support_claim_does_not_treat_counter_evidence_as_support(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    store = ScholarProjectStore(root)
    service = ClaimSupportService(CounterResearch(), project_store=store)  # type: ignore[arg-type]
    result = service.support_claim(SupportClaimRequest("SUP-C", store.project_id, "A disputed claim"))
    assert result.support_status == "CONTRADICTED"
    assert result.supporting_evidence == ()
    assert result.counter_evidence[0]["evidence_id"] == "E-counter"


def make_synthesis_runtime(tmp_path: Path) -> tuple[ScholarSkillRuntime, ScholarProjectStore, Path]:
    root = make_project(tmp_path)
    add_synthesis_sections(root)
    store = ScholarProjectStore(root)
    store.put_fact(ManuscriptFact("F1", "accuracy", "91%", confirmed=True))
    store.put_contribution(Contribution("C1", "A confirmed correction approach.", "confirmed", True))
    synthesis = SynthesisWritingService(root, project_store=store)
    return ScholarSkillRuntime(root, project_store=store, synthesis=synthesis), store, root


def test_conclusion_and_abstract_share_synthesis_runtime_and_do_not_use_rag(tmp_path: Path) -> None:
    runtime, store, root = make_synthesis_runtime(tmp_path)
    conclusion_writer = SynthesisWriter("The method addresses the problem and achieves 91% accuracy.")
    conclusion = runtime.execute(
        WritingRequest(
            "CON-1",
            store.project_id,
            "Write the conclusion.",
            target_section="conclusion",
        ),
        writer=conclusion_writer,
    )
    assert conclusion.status == "READY"
    assert conclusion.value.patch is not None
    assert conclusion.value.patch.citation_keys == ()
    assert conclusion.value.patch.used_evidence_ids == ()
    assert (root / "sections" / "conclusion.tex").read_text(encoding="utf-8") == "Existing conclusion.\n"

    abstract_writer = SynthesisWriter("We address the problem with the method and obtain 91% accuracy.")
    abstract = runtime.execute(
        WritingRequest(
            "ABS-1",
            store.project_id,
            "Write the abstract.",
            target_section="abstract",
        ),
        writer=abstract_writer,
    )
    assert abstract.status == "READY"
    assert abstract.value.patch is not None
    assert abstract.value.patch.target_section == "abstract"
    assert abstract.value.patch.citation_keys == ()
    assert conclusion.value.patch.patch_id != abstract.value.patch.patch_id


def test_conclusion_rejects_new_result_and_abstract_requires_results(tmp_path: Path) -> None:
    runtime, store, root = make_synthesis_runtime(tmp_path)
    result = runtime.execute(
        WritingRequest("CON-2", store.project_id, "Write conclusion", target_section="conclusion"),
        writer=SynthesisWriter("The method achieves 99% accuracy."),
    )
    assert result.status == "NEEDS_USER_REVIEW"
    assert any(issue.code == "NEW_MANUSCRIPT_RESULT" and issue.severity == "BLOCKER" for issue in result.value.review_report.issues)

    (root / "sections" / "results.tex").unlink()
    abstract = runtime.execute(
        WritingRequest("ABS-2", store.project_id, "Write abstract", target_section="abstract"),
        writer=SynthesisWriter("No result available."),
    )
    assert abstract.status == "FAILED"
    assert abstract.value.error_codes == ("INSUFFICIENT_MANUSCRIPT_STATE",)


def test_deterministic_dependency_graph_marks_downstream_sections_stale(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    add_synthesis_sections(root)
    store = ScholarProjectStore(root)
    synchronizer = __import__("app.scholar.manuscript", fromlist=["ManuscriptSynchronizer"]).ManuscriptSynchronizer(root)
    first = synchronizer.scan()
    store.save_manuscript_state(first)
    (root / "sections" / "method.tex").write_text("Changed method.\n", encoding="utf-8")
    second = synchronizer.scan(previous=first)
    assert {"method", "conclusion", "abstract"} <= set(second.stale_sections)


def test_cross_skill_project_flow_reuses_runtime_and_patch_approval(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    add_synthesis_sections(root)
    store = ScholarProjectStore(root)
    store.put_contribution(Contribution("C1", "A confirmed correction approach.", "confirmed", True))

    introduction = ScholarWritingService(
        root,
        FakeCapability(),  # type: ignore[arg-type]
        FakeWriter(),
        project_store=store,
        delegate=FakeDelegate((evidence_pack(),)),  # type: ignore[arg-type]
    )
    support_research = FakeResearch()
    support = ClaimSupportService(support_research, project_store=store)  # type: ignore[arg-type]
    synthesis = SynthesisWritingService(root, project_store=store)
    runtime = ScholarSkillRuntime(
        root,
        project_store=store,
        introduction=introduction,
        support_claim=support,
        synthesis=synthesis,
    )

    supported = runtime.execute(SupportClaimRequest("FLOW-S", store.project_id, "Prior work supports correction."))
    assert supported.task_type == "SUPPORT_CLAIM"
    intro = runtime.execute(
        WritingRequest("FLOW-I", store.project_id, "Revise prior work", focus="ephemeris correction"),
        writer=introduction.writer,
    )
    assert intro.value.patch is not None
    applied = introduction.approval_service.approve(
        PatchApprovalRequest(
            intro.value.patch.patch_id,
            store.project_id,
            "ACCEPT",
            intro.value.patch.base_hash,
            "human:flow",
        )
    )
    assert applied.status == "APPLIED"

    conclusion = runtime.execute(
        WritingRequest("FLOW-C", store.project_id, "Write conclusion", target_section="conclusion"),
        writer=SynthesisWriter("The method addresses the problem and achieves 91% accuracy."),
    )
    abstract = runtime.execute(
        WritingRequest("FLOW-A", store.project_id, "Write abstract", target_section="abstract"),
        writer=SynthesisWriter("We address the problem and obtain 91% accuracy."),
    )
    assert conclusion.value.patch is not None
    assert abstract.value.patch is not None
    assert conclusion.value.patch.project_id == abstract.value.patch.project_id == store.project_id
    assert (root / "sections" / "introduction.tex").read_text(encoding="utf-8") != "Original introduction.\n"
