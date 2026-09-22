from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any

import pytest

from app.contracts import CandidateEvidence
from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline
from app.knowledge import UnifiedKnowledgeService
from app.scholar.models import EvidencePack
from app.scholar.research import (
    EvidenceRef,
    EvidenceValidationError,
    InvalidEvidenceLocator,
    PaperCandidate,
    PaperSearchResult,
    ResearchBudget,
    ResearchCapabilityService,
    ResearchRequest,
    SectionCandidate,
    SectionSearchResult,
    SourceNotFound,
)
from app.scholar.writing.models import ResearchNeed
from app.scholar.writing.service import ResearchDelegate
from app.workspaces import WorkspaceService
from tests.test_corpus_workspace import write_fixture


class FakeKnowledge:
    workspace_id = "default"
    scope_version = 1

    def __init__(self, evidence: EvidenceIntelligencePipeline) -> None:
        self.evidence = evidence
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return {
            "retriever": "fixture_hierarchical",
            "candidate_papers": [
                {
                    "paper_id": "P_001",
                    "document_id": "D_001",
                    "work_id": "W_001",
                    "title": "Measurement paper",
                    "authors": ["Author"],
                    "year": 2024,
                    "abstract": "A measured result.",
                    "score": 0.9,
                    "rank": 1,
                }
            ],
            "results": [
                {
                    "paper_id": "P_001",
                    "document_id": "D_001",
                    "work_id": "W_001",
                    "section_id": "D_001_s0001",
                    "chunk_id": "D_001_c1",
                    "score": 0.8,
                    "rank": 1,
                    "content": "canonical text",
                    "block_ids": ["P_001_b1"],
                }
            ],
        }


class FakeOfficialRuntime(FakeKnowledge):
    embedding_provider = object()
    reranker_provider = None
    last_diagnostics = {"retrieval_mode": "fixture"}

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        result = super().retrieve(query, **kwargs)
        return result


def capability(tmp_path: Path) -> ResearchCapabilityService:
    write_fixture(tmp_path, "D_001", "canonical text")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(corpus, workspaces, max_candidates_per_document=20, max_selected_per_document=20)
    return ResearchCapabilityService(FakeKnowledge(evidence), corpus=corpus, evidence=evidence, workspaces=workspaces)


def test_search_papers_and_sections_hide_backend_details(tmp_path: Path) -> None:
    service = capability(tmp_path)

    papers = service.search_papers("measurement", top_k=1)
    sections = service.search_sections("measurement", paper_ids=["P_001"], top_k=1)

    assert len(papers.papers) == 1
    assert papers.papers[0].paper_id == "P_001"
    assert not hasattr(papers.papers[0], "vector")
    assert sections.sections[0].section_id == "D_001_s0001"
    assert sections.sections[0].chunk_ids == ("D_001_c1",)


def test_capability_uses_unified_service_without_exposing_its_backend(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001", "canonical text")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(corpus, workspaces, max_candidates_per_document=20, max_selected_per_document=20)
    unified = UnifiedKnowledgeService(FakeOfficialRuntime(evidence), evidence)
    service = ResearchCapabilityService(unified)

    result = service.search_sections("measurement", top_k=1)

    assert result.sections[0].chunk_ids == ("D_001_c1",)
    assert "embedding_provider" not in result.metadata


def test_local_retrieval_serializes_concurrent_tool_calls(tmp_path: Path) -> None:
    service = capability(tmp_path)
    knowledge = service.knowledge
    active = 0
    maximum = 0
    state_lock = Lock()
    retrieve = knowledge.retrieve

    def tracked_retrieve(query: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            sleep(0.02)
            return retrieve(query, **kwargs)
        finally:
            with state_lock:
                active -= 1

    knowledge.retrieve = tracked_retrieve  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.search_papers, "measurement", top_k=1)
            for _ in range(2)
        ]
        for future in futures:
            assert future.result().papers

    assert maximum == 1


def test_research_delegate_runs_independent_needs_in_parallel() -> None:
    active = 0
    maximum = 0
    state_lock = Lock()
    from threading import Barrier

    barrier = Barrier(3)

    class ParallelCapability:
        workspace_id = "default"
        scope_version = 1

        def research(self, request: object, *, allow_web: bool = False, parallel: bool = False) -> EvidencePack:
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            try:
                barrier.wait(timeout=1)
                return EvidencePack(
                    request_id=str(getattr(request, "request_id")),
                    query=str(getattr(request, "query")),
                    claims=(),
                    evidence=(),
                )
            finally:
                with state_lock:
                    active -= 1

    delegate = ResearchDelegate(ParallelCapability(), max_parallelism=3)  # type: ignore[arg-type]
    needs = tuple(
        ResearchNeed(
            need_id=f"need-{index}",
            rhetorical_move="background",
            query=f"query-{index}",
            target_claims=(f"claim-{index}",),
        )
        for index in range(3)
    )

    packs = delegate.research_needs("REQ-PARALLEL", needs)

    assert maximum == 3
    assert [pack.query for pack in packs] == ["query-0", "query-1", "query-2"]
    assert {pack.metadata["research_need_id"] for pack in packs} == {
        "need-0",
        "need-1",
        "need-2",
    }
    assert {pack.metadata["parallel_worker"] for pack in packs} >= {
        "research-need_0",
        "research-need_1",
    }


def test_introduction_coverage_fans_out_fifteen_papers_with_bounded_parallelism(
    tmp_path: Path,
) -> None:
    service = capability(tmp_path)
    paper_ids = tuple(f"P_{index:03d}" for index in range(1, 16))
    papers = PaperSearchResult(
        "REQ-15",
        "coverage",
        tuple(
            PaperCandidate(
                paper_id=paper_id,
                title=f"Paper {paper_id}",
                rank=index,
            )
            for index, paper_id in enumerate(paper_ids, 1)
        ),
    )
    request = ResearchRequest(
        "REQ-15",
        "coverage",
        budget=ResearchBudget(
            max_papers=15,
            max_sections=15,
            max_evidence_items=15,
        ),
        metadata={"ensure_paper_coverage": True},
    )
    active = 0
    maximum = 0
    calls: list[tuple[str, ...]] = []
    state_lock = Lock()
    from threading import Barrier

    barrier = Barrier(3)

    def fake_search_papers(
        _request: ResearchRequest,
        *,
        top_k: int,
        parallel: bool,
    ) -> PaperSearchResult:
        assert top_k == 15
        assert parallel is True
        return papers

    def fake_search_sections(
        _request: ResearchRequest,
        *,
        paper_ids: tuple[str, ...],
        top_k: int,
        parallel: bool,
    ) -> SectionSearchResult:
        nonlocal active, maximum
        calls.append(tuple(paper_ids))
        if len(paper_ids) != 1:
            return SectionSearchResult("REQ-15", "coverage", ())
        assert top_k == 1
        assert parallel is True
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=1)
            return SectionSearchResult(
                "REQ-15",
                "coverage",
                (
                    SectionCandidate(
                        paper_id=paper_ids[0],
                        section_id=f"{paper_ids[0]}-section",
                        chunk_ids=(f"{paper_ids[0]}-chunk",),
                    ),
                ),
            )
        finally:
            with state_lock:
                active -= 1

    service.search_papers = fake_search_papers  # type: ignore[method-assign]
    service.search_sections = fake_search_sections  # type: ignore[method-assign]
    service.read_evidence = lambda _refs: ()  # type: ignore[method-assign]

    service._local_candidates(request, parallel=True)

    assert maximum == 3
    assert len(calls) == 16
    assert calls[0] == paper_ids
    assert {call[0] for call in calls[1:]} == set(paper_ids)


def test_read_evidence_checks_locator_ownership_and_provenance(tmp_path: Path) -> None:
    service = capability(tmp_path)

    value = service.read_evidence(evidence_refs=(EvidenceRef(paper_id="P_001", section_id="D_001_s0001", chunk_id="D_001_c1"),))

    assert value[0].text == "canonical text"
    assert value[0].paper_id == "P_001"
    assert value[0].page_start == 2
    with pytest.raises(InvalidEvidenceLocator):
        service.read_evidence((EvidenceRef(paper_id="P_OTHER", chunk_id="D_001_c1"),))
    with pytest.raises(SourceNotFound):
        service.read_evidence((EvidenceRef(chunk_id="fabricated"),))


def test_evidence_candidate_validation_rejects_fabricated_source_text(tmp_path: Path) -> None:
    service = capability(tmp_path)
    request = ResearchRequest("REQ-1", "measurement", target_claims=("canonical text",), budget=ResearchBudget(max_papers=1, max_sections=1, max_evidence_items=1))
    candidate = CandidateEvidence(
        "E-1", "REQ-1", "default", 1, "fabricated text", 0.8, "research_capability_local",
        work_id="W_001", document_id="D_001", chunk_id="D_001_c1", section_id="D_001_s0001", block_ids=("P_001_b1",),
    )

    with pytest.raises(EvidenceValidationError):
        service.validate_candidates(request, (candidate,))


def test_evidence_pack_preserves_claim_links_and_deterministic_coverage(tmp_path: Path) -> None:
    service = capability(tmp_path)
    request = ResearchRequest(
        "REQ-1", "measurement", target_claims=("canonical text", "missing claim"),
        budget=ResearchBudget(max_papers=1, max_sections=1, max_evidence_items=1),
    )
    candidate = CandidateEvidence(
        "E-1", "REQ-1", "default", 1, "canonical text", 0.8, "research_capability_local",
        work_id="W_001", document_id="D_001", chunk_id="D_001_c1", section_id="D_001_s0001", block_ids=("P_001_b1",),
        metadata={"paper_id": "P_001", "section_id": "D_001_s0001"},
    )

    pack = service.build_evidence_pack(request, (candidate,))

    assert isinstance(pack, EvidencePack)
    assert pack.coverage == 0.5
    assert pack.claims[0]["evidence_ids"] == ("E-1",)
    assert pack.claims[1]["status"] == "unresolved"
    assert pack.unresolved == ("missing claim",)
    assert pack.evidence[0]["chunk_id"] == "D_001_c1"
