from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline
from app.research.harness import HarnessState, ResearchBudgetPolicy, ResearchRunHarness
from app.scholar import Contribution, ScholarProjectStore
from app.scholar.approval import PatchApprovalRequest
from app.scholar.research import LiteratureSearchRequest, ResearchBudget, ResearchCapabilityService, ResearchRequest
from app.scholar.research.web import WebLiteratureAdapter
from app.scholar.writing import ClaimSupportService, ScholarWritingService, SupportClaimRequest, WritingRequest
from app.scholar.writing.models import SectionDraft, WritingContext
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture
from tests.test_scholar_foundation import make_project


class Knowledge:
    workspace_id = "default"
    scope_version = 1

    def __init__(self, *, local: bool) -> None:
        self.local = local
        self.calls: list[str] = []

    def retrieve(self, query: str, **_: Any) -> Mapping[str, Any]:
        self.calls.append(query)
        if not self.local:
            return {"candidate_papers": [], "results": []}
        return {
            "candidate_papers": [{"paper_id": "P_001", "document_id": "D_001", "title": "Local paper"}],
            "results": [{
                "paper_id": "P_001",
                "document_id": "D_001",
                "section_id": "D_001_s0001",
                "chunk_id": "D_001_c1",
                "content": "canonical text supports the claim",
                "score": 0.9,
                "block_ids": ["P_001_b1"],
            }],
        }


def make_service(tmp_path: Path, *, local: bool, searcher: Any) -> ResearchCapabilityService:
    if local:
        write_fixture(tmp_path, "D_001", "canonical text supports the claim")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(corpus, workspaces, max_candidates_per_document=20, max_selected_per_document=20)
    web = WebLiteratureAdapter(searcher=searcher)
    return ResearchCapabilityService(
        Knowledge(local=local), corpus=corpus, evidence=evidence, workspaces=workspaces, web=web
    )


def request(mode: str, *, claim: str = "canonical text supports the claim") -> ResearchRequest:
    return ResearchRequest(
        request_id="REQ-WEB",
        query=claim,
        target_claims=(claim,),
        freshness_mode=mode,  # type: ignore[arg-type]
        budget=ResearchBudget(max_papers=2, max_sections=2, max_evidence_items=2, max_web_queries=1),
    )


def web_result(_: Any, __: Any) -> Mapping[str, Any]:
    return {
        "results": [{
            "paper_id": "WEB-1",
            "title": "External evidence paper",
            "authors": ["Author"],
            "doi": "10.1000/example",
            "publication_year": 2025,
            "abstract": "canonical text supports the claim",
            "landing_page_url": "https://doi.org/10.1000/example",
            "sources": ["fixture-provider"],
        }],
        "provider_failures": [],
    }


def test_local_first_skips_web_when_verified_local_evidence_is_sufficient(tmp_path: Path) -> None:
    called = False

    def searcher(_: Any, __: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return web_result(_, __)

    service = make_service(tmp_path, local=True, searcher=searcher)
    pack = service.research(request("LOCAL_FIRST"), allow_web=True)

    assert called is False
    assert pack.web_used is False
    assert pack.metadata["freshness_decision"]["reason"] == "LOCAL_EVIDENCE_SUFFICIENT"


def test_fresh_required_checks_web_even_when_local_evidence_exists(tmp_path: Path) -> None:
    service = make_service(tmp_path, local=True, searcher=web_result)
    pack = service.research(
        request("FRESH_REQUIRED", claim="canonical text supports the claim"),
        allow_web=True,
    )

    assert pack.web_used is True
    assert any(value.get("source_type") == "WEB_LITERATURE" for value in pack.evidence)
    assert pack.evidence[0].get("source_locator")
    assert pack.evidence[0].get("locator_type") == "ABSTRACT"


def test_default_local_only_never_calls_web(tmp_path: Path) -> None:
    def fail(_: Any, __: Any) -> Mapping[str, Any]:
        raise AssertionError("Web must not be called")

    service = make_service(tmp_path, local=False, searcher=fail)
    pack = service.research(request("LOCAL_ONLY"), allow_web=True)

    assert pack.web_used is False
    assert pack.unresolved


def test_unverifiable_snippet_is_not_verified_evidence(tmp_path: Path) -> None:
    def snippet(_: Any, __: Any) -> Mapping[str, Any]:
        return {"results": [{"paper_id": "WEB-1", "title": "Snippet only", "snippet": "looks similar"}]}

    service = make_service(tmp_path, local=False, searcher=snippet)
    pack = service.research(request("LOCAL_FIRST"), allow_web=True)

    assert pack.web_used is True
    assert pack.evidence == ()
    assert pack.unresolved


def test_provider_failure_is_not_reported_as_unsupported(tmp_path: Path) -> None:
    def timeout(_: Any, __: Any) -> Mapping[str, Any]:
        raise TimeoutError("provider timeout")

    service = make_service(tmp_path, local=False, searcher=timeout)
    pack = service.research(request("LOCAL_FIRST"), allow_web=True)

    assert pack.unresolved
    assert pack.metadata["error_code"] == "WEB_PROVIDER_UNAVAILABLE"
    assert pack.metadata["web_failure"]["error_code"] == "WEB_PROVIDER_TIMEOUT"


def test_same_doi_reuses_local_evidence_instead_of_creating_web_duplicate(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001", "canonical text supports the claim")
    paper_path = tmp_path / "data" / "canonical" / "P_001" / "paper.json"
    value = json.loads(paper_path.read_text(encoding="utf-8"))
    value["metadata"].update({"doi": "10.1000/example", "authors": ["Author"], "year": 2024})
    paper_path.write_text(json.dumps(value), encoding="utf-8")

    calls = 0

    def searcher(_: Any, __: Any) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "results": [{
                "paper_id": "WEB-SAME",
                "title": "Fixture",
                "authors": ["Author"],
                "doi": "https://doi.org/10.1000/example",
                "publication_year": 2025,
                "abstract": "canonical text supports the claim",
                "landing_page_url": "https://doi.org/10.1000/example",
            }],
        }

    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(corpus, workspaces, max_candidates_per_document=20, max_selected_per_document=20)
    service = ResearchCapabilityService(
        Knowledge(local=True), corpus=corpus, evidence=evidence, workspaces=workspaces,
        web=WebLiteratureAdapter(searcher=searcher),
    )
    pack = service.research(request("FRESH_REQUIRED"), allow_web=True)

    assert calls == 1
    assert len(pack.evidence) == 1
    assert pack.evidence[0]["source_type"] == "LOCAL_CORPUS"
    assert pack.metadata["freshness_decision"]["web_required"] is True


def test_fresh_required_rejects_dated_but_unverifiable_snippet(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        local=False,
        searcher=lambda *_: {
            "results": [{
                "paper_id": "WEB-1",
                "title": "Recent paper",
                "authors": ["Author"],
                "publication_year": 2025,
                "snippet": "looks relevant",
            }],
        },
    )
    pack = service.research(request("FRESH_REQUIRED"), allow_web=True)

    assert pack.evidence == ()
    assert pack.metadata["error_code"] == "FRESHNESS_UNAVAILABLE"


def test_metadata_date_conflict_is_preserved_during_normalization() -> None:
    result = WebLiteratureAdapter._normalize(
        {
            "results": [{
                "paper_id": "WEB-1", "title": "Same work", "authors": ["Author"],
                "doi": "10.1000/example", "publication_year": 2024,
            }, {
                "paper_id": "WEB-2", "title": "Same work", "authors": ["Author"],
                "doi": "10.1000/example", "publication_year": 2025,
            }],
        },
        "REQ-META",
        "same work",
    )

    assert len(result.candidates) == 1
    assert result.metadata_conflicts


def test_discovery_cache_and_harness_trace_are_reused_without_web_loop(tmp_path: Path) -> None:
    from app.scholar.research.cache import LiteratureDiscoveryCache

    calls = 0

    def searcher(_: Any, __: Any) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return web_result(_, __)

    adapter = WebLiteratureAdapter(searcher=searcher, cache=LiteratureDiscoveryCache(tmp_path))
    first = adapter.search(LiteratureSearchRequest("CACHE-1", "ephemeris correction", max_results=2))
    second = adapter.search(LiteratureSearchRequest("CACHE-2", "ephemeris correction", max_results=2))

    assert calls == 1
    assert first.candidates and second.candidates
    assert adapter.last_diagnostics["cache_hit"] is True


def test_research_harness_explains_freshness_and_web_verification(tmp_path: Path) -> None:
    service = make_service(tmp_path, local=False, searcher=web_result)
    harness = ResearchRunHarness(
        "scholar_research",
        ResearchBudgetPolicy(
            max_steps=10,
            max_tool_calls=2,
            max_external_searches=1,
            max_retrieval_rounds=1,
            max_context_tokens=1_000,
            max_total_tokens=2_000,
        ),
    )
    harness.transition(HarnessState.CONTEXT_PREPARING)
    harness.transition(HarnessState.PLANNING)
    harness.transition(HarnessState.EXECUTING)
    service.research(request("LOCAL_FIRST"), allow_web=True, harness=harness)

    steps = [event.name for event in harness.trace if event.kind == "step"]
    assert "FRESHNESS_DECIDE" in steps
    assert "WEB_DISCOVERY" in steps
    assert "EVIDENCE_VALIDATE" in steps
    assert harness.provider_usage


def test_support_claim_web_to_introduction_patch_to_human_apply(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    write_fixture(root, "D_001", "Local corpus does not answer the external claim.")
    store = ScholarProjectStore(root)
    store.put_contribution(Contribution("C1", "A user-confirmed ephemeris correction approach.", "confirmed", True))
    corpus = CanonicalCorpusService(root)
    workspaces = WorkspaceService(root, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(corpus, workspaces, max_candidates_per_document=20, max_selected_per_document=20)

    class EmptyKnowledge(Knowledge):
        def __init__(self) -> None:
            super().__init__(local=False)

    def searcher(literature_request: Any, _: Any) -> Mapping[str, Any]:
        # Query-specific content gives the deterministic coverage proxy enough
        # overlap without introducing a semantic judge in this integration.
        return {
            "results": [{
                "paper_id": f"WEB-{literature_request.request_id}",
                "title": "External ephemeris correction study",
                "authors": ["Author"],
                "doi": "10.1000/ephemeris",
                "publication_year": 2025,
                "abstract": literature_request.query,
                "landing_page_url": "https://doi.org/10.1000/ephemeris",
                "bibkey": "Author2025",
            }],
        }

    research = ResearchCapabilityService(
        EmptyKnowledge(),
        corpus=corpus,
        evidence=evidence,
        workspaces=workspaces,
        web=WebLiteratureAdapter(searcher=searcher),
    )
    support = ClaimSupportService(research, project_store=store)
    supported = support.support_claim(
        SupportClaimRequest(
            "SUP-WEB-1",
            store.project_id,
            "ephemeris correction affects positioning",
            metadata={"freshness_mode": "LOCAL_FIRST", "max_web_queries": 1},
        )
    )
    assert supported.support_status == "SUPPORTED"
    assert supported.metadata["web_used"] is True

    class WebWriter:
        def generate(self, context: WritingContext) -> SectionDraft:
            evidence_ids = tuple(
                item["evidence_id"]
                for pack in context.evidence_packs
                for item in pack.evidence
            )
            claim_ids = tuple(item.claim_id for item in context.claim_plan.claims)
            contribution_ids = tuple(item.contribution_id for item in context.contributions)
            return SectionDraft(
                target_section="introduction",
                base_hash=context.current_hash,
                content="Prior work studies ephemeris correction. Our confirmed direction extends it.\n",
                claim_ids=claim_ids,
                evidence_ids=evidence_ids,
                citation_keys=tuple(context.citation_catalog),
                contribution_ids=contribution_ids,
                original_content=context.current_introduction,
                change_summary="Add externally verified prior work.",
            )

    writing = ScholarWritingService(root, research, WebWriter(), project_store=store)
    result = writing.write_introduction(
        WritingRequest(
            "WRITE-WEB-1",
            store.project_id,
            "Revise the prior work in the introduction.",
            focus="ephemeris correction prior work",
            metadata={"freshness_mode": "LOCAL_FIRST", "max_web_queries": 1},
        )
    )
    assert result.patch is not None
    assert result.patch.project_id == store.project_id
    assert result.patch.used_evidence_ids
    assert result.review_report is not None and result.review_report.valid is True

    applied = writing.approval_service.approve(
        PatchApprovalRequest(
            result.patch.patch_id,
            store.project_id,
            "ACCEPT",
            result.patch.base_hash,
            "human-test",
        )
    )
    assert applied.status == "APPLIED"
    assert "Prior work studies ephemeris correction." in (root / "sections" / "introduction.tex").read_text(encoding="utf-8")
