from __future__ import annotations

from typing import Any, Mapping

from app.research import (
    DimensionCoverageAnalyzer,
    QueryFrameBuilder,
    ResearchRuntime,
    WorkflowName,
    WorkflowRequest,
    build_default_gateway,
)


QUERY = "伪距修正星历后，如何通过伪距率、自适应噪声和鲁棒权值提高定位精度？"
EXPECTED = (
    "ephemeris_correction",
    "range_rate_observation",
    "state_observability",
    "adaptive_noise_estimation",
    "robust_weighting",
    "positioning_accuracy",
)


def _evidence(
    evidence_id: str,
    dimensions: list[str],
    *,
    document_id: str,
    directness: str = "direct",
    relation: bool = True,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "evidence_state": "selected",
        "document_id": document_id,
        "chunk_id": f"{document_id}_{evidence_id}",
        "page_start": 1,
        "page_end": 1,
        "block_ids": ["B_1"],
        "content": "Canonical relation evidence.",
        "directness": directness,
        "evidence_grade": "primary",
        "dimension_tags": dimensions,
        "relation_path": ["source", "target"] if relation else [],
    }


def test_query_frame_extracts_required_research_dimensions_without_planner() -> None:
    frame = QueryFrameBuilder().build(QUERY)

    assert frame.research_object == "positioning"
    assert frame.required_dimensions == EXPECTED
    assert "positioning_accuracy" in frame.target_outcome


def test_dimension_report_tracks_direct_source_conflict_and_path_coverage() -> None:
    frame = QueryFrameBuilder().build(QUERY)
    evidence = [
        _evidence("E1", list(EXPECTED[:3]), document_id="D_1"),
        _evidence("E2", list(EXPECTED[3:]), document_id="D_2"),
    ]
    conflicts = [
        {"supporting_evidence_id": "E1", "opposing_evidence_id": "E2"}
    ]

    report = DimensionCoverageAnalyzer().evaluate(frame, evidence, conflicts)

    assert report.required_dimension_coverage == 1.0
    assert report.direct_evidence_coverage == 1.0
    assert report.source_diversity == 2
    assert report.conflict_coverage == 1.0
    assert report.relation_path_coverage == 1.0
    assert report.missing_dimensions == ()
    assert report.overall_sufficient is True


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, context: Any) -> Mapping[str, Any]:
        self.calls += 1
        ids = [value["evidence_id"] for value in context.selected_evidence]
        return {
            "answerable": True,
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Canonical relation evidence supports the mechanism.",
                    "evidence_ids": ids,
                }
            ],
            "usage": {"total_tokens": 10},
        }


def test_relation_workflow_uses_missing_dimensions_for_one_gap_retrieval() -> None:
    calls: list[str] = []
    first = [
        _evidence("E1", list(EXPECTED[:2]), document_id="D_1"),
        _evidence("E2", [EXPECTED[2]], document_id="D_2"),
    ]
    gap = [
        _evidence("E3", list(EXPECTED[3:]), document_id="D_3"),
    ]

    def retrieve(arguments, context):  # type: ignore[no-untyped-def]
        calls.append(str(arguments["query"]))
        return {"results": first if len(calls) == 1 else gap}

    handlers = {
        "workspace.read_scope": lambda arguments, context: {
            "workspace_id": "default",
            "scope_version": 1,
            "constraints": [],
        },
        "knowledge.retrieve": retrieve,
    }
    generator = _Generator()
    runtime = ResearchRuntime(build_default_gateway(handlers), generator)

    result = runtime.run(
        WorkflowRequest(
            query=QUERY,
            workspace_id="default",
            scope_version=1,
            workflow=WorkflowName.RELATION_REASONING,
        )
    )

    assert len(calls) == 2
    assert "adaptive_noise_estimation" in calls[1]
    assert result["coverage"]["required_dimension_coverage"] == 1.0
    assert result["coverage"]["missing_dimensions"] == ()
    assert result["workflow_details"]["gap_retrievals"] == 1
    assert generator.calls == 1


def test_missing_dimension_is_reported_even_with_many_evidence_items() -> None:
    frame = QueryFrameBuilder().build(QUERY)
    evidence = [
        _evidence(f"E{index}", ["ephemeris_correction"], document_id=f"D_{index}")
        for index in range(5)
    ]

    report = DimensionCoverageAnalyzer().evaluate(frame, evidence)

    assert report.overall_sufficient is False
    assert "positioning_accuracy" in report.missing_dimensions

