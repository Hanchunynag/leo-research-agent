"""Deterministic research-task planning for the LangGraph Agent.

The planner owns task intent and evidence organisation only.  Existing Knowledge
Service retrieval and Evidence verification remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Literal


ResearchTaskType = Literal[
    "direct_qa",
    "paper_summary",
    "multi_paper_summary",
    "author_analysis",
    "timeline",
    "compare",
    "literature_search",
]

COMPARISON_DIMENSIONS = (
    "method",
    "data_and_experiment",
    "contribution",
    "limitation",
)


def classify_research_task(query: str) -> ResearchTaskType:
    """Classify the bounded workflow shapes supported by this Agent."""

    normalized = query.casefold()
    if any(
        marker in normalized
        for marker in (
            "时间线", "演进", "演化", "发展路线", "研究路线", "技术路线",
            "timeline", "evolution", "progression", "roadmap",
        )
    ):
        return "timeline"
    if any(
        marker in normalized
        for marker in ("比较", "对比", "compare", " versus ", " vs. ", " vs ")
    ):
        return "compare"
    if any(marker in normalized for marker in ("作者", "author")) and any(
        marker in normalized
        for marker in ("方向", "研究", "分析", "profile", "direction", "research")
    ):
        return "author_analysis"
    if any(
        marker in normalized
        for marker in ("多篇", "多论文", "文献综述", "综述", "literature review", "survey")
    ):
        return "multi_paper_summary"
    if any(marker in normalized for marker in ("找", "搜索", "英文论文", "find", "search")):
        return "literature_search"
    if any(marker in normalized for marker in ("讲什么", "总结", "summary", "summarize")):
        return "paper_summary"
    return "direct_qa"


def _planner_fields(query: str) -> dict[str, Any]:
    years = [int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", query)]
    constraints: dict[str, Any] = {}
    if len(years) >= 2:
        constraints = {"year_from": min(years), "year_to": max(years)}
    elif len(years) == 1:
        constraints = {"year_from": years[0], "year_to": years[0]}
    keywords = [
        value
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", query)
        if value.casefold() not in {"paper", "papers", "论文", "哪些", "什么", "如何"}
    ]
    return {
        "research_intent": "",
        "keywords": list(dict.fromkeys(keywords))[:20],
        "time_constraints": constraints,
        "expected_evidence": ["direct passage from full text"],
        "evidence_type_required": "full_text",
        "paper_candidate_limit": 10,
        "retrieval_budget": {"paper_rounds": 1, "evidence_rounds": 1},
    }


def build_research_plan(task_type: ResearchTaskType, query: str = "") -> dict[str, Any]:
    """Return a small, serializable, auditable plan for graph nodes."""

    plan: dict[str, Any] = {
        "task_type": task_type,
        "semantic_retrieval": True,
        "requires_paper_resolution": task_type
        in {"timeline", "compare", "multi_paper_summary", "author_analysis"},
        "requires_metadata": task_type in {"timeline", "compare", "author_analysis"},
        "metadata_fields": [],
        "output_structure": "grounded_answer",
    }
    plan.update(_planner_fields(query))
    plan["research_intent"] = task_type
    if task_type == "timeline":
        plan.update(
            {
                "metadata_fields": ["title", "year", "authors", "venue", "doi"],
                "order_by": "publication_year_ascending",
                "output_structure": "chronological_stages",
                "evidence_focus": ["method", "contribution", "limitation"],
                "per_paper_evidence": True,
            }
        )
    elif task_type == "compare":
        plan.update(
            {
                "metadata_fields": ["title", "year", "authors", "venue", "doi"],
                "comparison_dimensions": list(COMPARISON_DIMENSIONS),
                "output_structure": "comparison_matrix",
                "per_paper_evidence": True,
            }
        )
    elif task_type == "author_analysis":
        plan.update(
            {
                "metadata_fields": ["title", "year", "authors", "venue", "keywords"],
                "output_structure": "author_research_profile",
            }
        )
    return plan


def sort_timeline(metadata: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort papers by resolved year, keeping unknown dates visible at the end."""

    values = [dict(value) for value in metadata]

    def key(value: Mapping[str, Any]) -> tuple[bool, int, str]:
        year = value.get("publication_year")
        if isinstance(year, int) and not isinstance(year, bool):
            missing_year = False
            year_key = year
        else:
            missing_year = True
            year_key = 10_000
        return (
            missing_year,
            year_key,
            str(value.get("matched_title") or value.get("title") or "").casefold(),
        )

    return sorted(values, key=key)


def group_evidence_by_paper(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Build the paper-to-evidence index used by compare/timeline diagnostics."""

    grouped: dict[str, list[str]] = {}
    for item in evidence:
        document_id = str(item.get("document_id") or "")
        evidence_id = str(item.get("evidence_id") or item.get("chunk_id") or "")
        if document_id and evidence_id:
            grouped.setdefault(document_id, []).append(evidence_id)
    return grouped
