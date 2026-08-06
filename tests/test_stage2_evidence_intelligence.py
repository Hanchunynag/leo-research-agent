from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.contracts import CandidateEvidence, EvidenceIntelligenceService, EvidenceRequest, SelectedEvidence
from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline, SelectedEvidenceContextBuilder
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture


def request(**options: object) -> EvidenceRequest:
    return EvidenceRequest(
        "REQ-1",
        "default",
        1,
        "How are A and B related?",
        top_k=10,
        retrieval_options=options,
    )


def graph_candidate(
    candidate_id: str,
    *,
    content: str,
    relation_path: tuple[str, ...],
    grade: str = "candidate",
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id,
        "REQ-1",
        "default",
        1,
        content,
        0.8,
        "lightrag_relation",
        work_id="W_001",
        document_id="D_001",
        chunk_id="D_001_c1",
        relation_path=relation_path,
        evidence_grade=grade,  # type: ignore[arg-type]
    )


def setup_pipeline(tmp_path: Path) -> EvidenceIntelligencePipeline:
    write_fixture(tmp_path, "D_001", "A measurement is used together with B observations.")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    return EvidenceIntelligencePipeline(corpus, workspaces)


def test_graph_relation_is_backfilled_and_directness_controls_grade(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    indirect = graph_candidate("E-1", content="A drives B", relation_path=("A", "B"))
    inferred = graph_candidate("E-2", content="C drives D", relation_path=("C", "D"))

    bundle = pipeline.verify(request(), [indirect, inferred])

    assert isinstance(pipeline, EvidenceIntelligenceService)
    assert [value.content for value in bundle.evidence] == [
        "A measurement is used together with B observations.",
        "A measurement is used together with B observations.",
    ]
    assert {value.directness for value in bundle.evidence} == {"indirect", "inferred"}
    inferred_value = next(value for value in bundle.evidence if value.directness == "inferred")
    assert inferred_value.evidence_grade == "graph_inference"
    assert "不能表述为论文已证明" in inferred_value.metadata["graph_inference_disclaimer"]
    assert pipeline.last_diagnostics["graph_backfill_rate"] == 1.0


def test_candidate_and_analogy_keep_different_evidence_grades(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    candidate = graph_candidate("E-1", content="A drives B", relation_path=("candidate",), grade="candidate")
    analogy = graph_candidate("E-2", content="A supports B", relation_path=("analogy",), grade="analogy")
    candidate = replace(candidate, retrieval_source="lightrag_chunk")
    analogy = replace(analogy, retrieval_source="lightrag_chunk")

    bundle = pipeline.verify(request(), [candidate, analogy])

    assert {value.evidence_grade for value in bundle.evidence} == {"candidate", "analogy"}


def test_only_selected_evidence_can_enter_context_builder(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    bundle = pipeline.verify(
        request(),
        [graph_candidate("E-1", content="A drives B", relation_path=("A", "B"))],
    )
    selected = pipeline.select(request(token_budget=100), bundle)
    builder = SelectedEvidenceContextBuilder()

    context = builder.build(request(), selected, token_budget=100)

    assert context.selected_evidence[0].evidence.state == "selected"
    assert context.diagnostics["selected_only"] is True
    unsafe = SelectedEvidence(bundle.evidence[0], 1, "unsafe", 5)
    with pytest.raises(ValueError, match="只有 selected"):
        builder.build(request(), [unsafe], token_budget=100)


def test_scope_filter_rejects_non_member_before_verification(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    outsider = CandidateEvidence(
        "E-X", "REQ-1", "default", 1, "outside", 1.0, "lightrag_chunk", document_id="D_X", chunk_id="D_X_c1"
    )

    bundle = pipeline.verify(request(), [outsider])

    assert bundle.evidence == ()
    assert bundle.rejected_candidate_ids == ("E-X",)
    assert pipeline.last_diagnostics["workspace_scope_pass_count"] == 0


def test_selection_never_exceeds_request_top_k(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    candidates = [
        graph_candidate(
            f"E-{index}",
            content=f"A drives B {index}",
            relation_path=("A", f"B{index}"),
        )
        for index in range(5)
    ]
    limited = replace(request(token_budget=10_000), top_k=2)

    selected = pipeline.select(limited, pipeline.verify(limited, candidates))

    assert len(selected) == 2


def test_selection_reserves_direct_chunk_and_graph_evidence(tmp_path: Path) -> None:
    pipeline = setup_pipeline(tmp_path)
    direct = replace(
        graph_candidate("DIRECT", content="A and B", relation_path=()),
        retrieval_source="lightrag_chunk",
    )
    graph = graph_candidate(
        "RELATION", content="A drives B", relation_path=("A", "B")
    )
    limited = replace(request(token_budget=10_000), top_k=2)

    selected = pipeline.select(limited, pipeline.verify(limited, [graph, direct]))

    assert [value.evidence.metadata["retrieval_source"] for value in selected] == [
        "lightrag_chunk",
        "lightrag_relation",
    ]
