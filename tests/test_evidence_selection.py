from __future__ import annotations

from typing import Any

from app.agentic.models import (
    CoverageItem,
    CoverageReport,
    EvidenceRequirement,
    PlannedSubquestion,
    QueryPlan,
)
from app.agentic.selection import CoverageAwareEvidenceSelector
from app.agentic.store import stable_json


def _plan() -> QueryPlan:
    return QueryPlan(
        intent="fact_list",
        target_category="measurement",
        excluded_categories=["input", "prior", "state"],
        subquestions=[
            PlannedSubquestion(id="SQ1", question="什么观测量估计星历误差？"),
            PlannedSubquestion(id="SQ2", question="什么观测量估计钟漂？"),
        ],
        retrieval_queries=["星历误差 钟漂 观测量"],
        required_evidence=[
            EvidenceRequirement(subquestion_id="SQ1", requirement="直接星历证据"),
            EvidenceRequirement(subquestion_id="SQ2", requirement="直接钟漂证据"),
        ],
        answer_constraints=["不得将预测星历列为观测量"],
    )


def _coverage(*pairs: tuple[str, list[str]]) -> CoverageReport:
    return CoverageReport(
        overall_sufficient=all(values for _, values in pairs),
        coverage=[
            CoverageItem(
                subquestion_id=subquestion_id,
                status="sufficient" if values else "missing",
                supporting_evidence_ids=values,
            )
            for subquestion_id, values in pairs
        ],
        followup_queries=[],
    )


def _candidate(
    evidence_id: str,
    content: str,
    *,
    rank: int,
    grade: int,
    work_id: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    suffix = evidence_id.removeprefix("E")
    return {
        "evidence_id": evidence_id,
        "chunk_id": f"C{suffix}",
        "work_id": work_id,
        "document_id": document_id or f"D_{work_id}",
        "title": f"Paper {work_id}",
        "section_path": ["METHOD"],
        "content": content,
        "rank": rank,
        "directness_grade": grade,
    }


def test_selector_keeps_one_required_evidence_for_each_subquestion() -> None:
    candidates = [
        _candidate(
            "E001",
            "Introduction background about ephemeris and clock errors.",
            rank=1,
            grade=1,
            work_id="W_background",
        ),
        _candidate(
            "E002",
            "Carrier phase measurements estimate ephemeris errors.",
            rank=7,
            grade=3,
            work_id="W_method",
        ),
        _candidate(
            "E003",
            "Doppler frequency measurements estimate clock drift.",
            rank=8,
            grade=3,
            work_id="W_method",
        ),
    ]
    selector = CoverageAwareEvidenceSelector(max_per_work=1)

    selected = selector.select(
        "哪些观测量估计星历误差和钟漂？",
        _plan(),
        _coverage(("SQ1", ["E002"]), ("SQ2", ["E003"])),
        candidates,
        2,
    )

    assert [item["evidence_id"] for item in selected] == ["E002", "E003"]
    assert all(item["selection_required"] is True for item in selected)
    assert selector.last_diagnostics["covered_subquestions"] == ["SQ1", "SQ2"]
    assert selector.last_diagnostics["coverage_preserved"] is True


def test_selector_filters_background_and_uses_source_diversity() -> None:
    candidates = [
        _candidate(
            "E001",
            "Carrier phase measurements estimate ephemeris error from navigation signals.",
            rank=1,
            grade=3,
            work_id="W_one",
        ),
        _candidate(
            "E002",
            "Carrier phase measurements estimate ephemeris errors using navigation signals.",
            rank=2,
            grade=3,
            work_id="W_one",
        ),
        _candidate(
            "E003",
            "Doppler observations constrain receiver and satellite clock drift.",
            rank=3,
            grade=2,
            work_id="W_two",
        ),
        _candidate(
            "E004",
            "Introduction background mentions ephemeris error as a challenge.",
            rank=4,
            grade=0,
            work_id="W_three",
        ),
    ]
    selector = CoverageAwareEvidenceSelector(max_per_work=2)

    selected = selector.select(
        "星历误差和钟漂观测量",
        _plan(),
        _coverage(("SQ1", ["E001"]), ("SQ2", ["E003"])),
        candidates,
        3,
    )

    assert [item["evidence_id"] for item in selected[:2]] == ["E001", "E003"]
    assert "E004" not in {item["evidence_id"] for item in selected}
    assert {item["work_id"] for item in selected} == {"W_one", "W_two"}
    assert selector.last_diagnostics["dropped_low_directness_count"] == 1
    assert selector.last_diagnostics["dropped_redundant_count"] == 1


def test_selector_required_evidence_bypasses_per_work_cap() -> None:
    candidates = [
        _candidate(
            "E001",
            "Carrier phase measurements estimate ephemeris error.",
            rank=1,
            grade=3,
            work_id="W_same",
        ),
        _candidate(
            "E002",
            "Doppler frequency measurements estimate clock drift.",
            rank=2,
            grade=3,
            work_id="W_same",
        ),
    ]
    selector = CoverageAwareEvidenceSelector(max_per_work=1)

    selected = selector.select(
        "观测量",
        _plan(),
        _coverage(("SQ1", ["E001"]), ("SQ2", ["E002"])),
        candidates,
        2,
    )

    assert [item["evidence_id"] for item in selected] == ["E001", "E002"]


def test_selector_is_deterministic_and_reports_budget_drops() -> None:
    contents = [
        "Carrier phase estimates orbital ephemeris error.",
        "Doppler frequency constrains relative clock drift.",
        "Pseudorange observations support absolute positioning.",
        "Signal strength metrics describe receiver tracking quality.",
    ]
    candidates = [
        _candidate(
            f"E00{index}",
            contents[index - 1],
            rank=index,
            grade=2,
            work_id=f"W{index}",
        )
        for index in range(1, 5)
    ]
    coverage = _coverage(("SQ1", ["E001"]), ("SQ2", ["E002"]))
    first_selector = CoverageAwareEvidenceSelector()
    second_selector = CoverageAwareEvidenceSelector()

    first = first_selector.select("观测量", _plan(), coverage, candidates, 3)
    second = second_selector.select("观测量", _plan(), coverage, candidates, 3)

    assert stable_json(first) == stable_json(second)
    assert stable_json(first_selector.last_diagnostics) == stable_json(
        second_selector.last_diagnostics
    )
    assert first_selector.last_diagnostics["dropped_budget_count"] == 1


def test_selector_reports_when_final_budget_cannot_preserve_coverage() -> None:
    candidates = [
        _candidate(
            "E001",
            "Carrier phase measurements estimate ephemeris error.",
            rank=1,
            grade=3,
            work_id="W_one",
        ),
        _candidate(
            "E002",
            "Doppler measurements estimate clock drift.",
            rank=2,
            grade=3,
            work_id="W_two",
        ),
    ]
    selector = CoverageAwareEvidenceSelector()

    selected = selector.select(
        "观测量",
        _plan(),
        _coverage(("SQ1", ["E001"]), ("SQ2", ["E002"])),
        candidates,
        1,
    )

    assert [item["evidence_id"] for item in selected] == ["E001"]
    assert selector.last_diagnostics["coverage_preserved"] is False
    assert selector.last_diagnostics["uncovered_subquestions"] == ["SQ2"]
