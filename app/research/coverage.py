"""Relation 问题的规则化 QueryFrame 与研究维度 Coverage。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


_DIMENSION_TERMS: dict[str, tuple[str, ...]] = {
    "ephemeris_correction": (
        "星历修正",
        "修正星历",
        "ephemeris correction",
        "corrected ephemeris",
        "orbit correction",
    ),
    "range_rate_observation": (
        "伪距率",
        "pseudorange rate",
        "range rate",
        "doppler observation",
        "多普勒观测",
    ),
    "state_observability": (
        "状态可观",
        "state observability",
        "observable state",
        "可观测性",
    ),
    "adaptive_noise_estimation": (
        "自适应噪声",
        "adaptive noise",
        "variance component estimation",
        "方差分量估计",
        "vce",
    ),
    "robust_weighting": (
        "鲁棒权",
        "鲁棒估计",
        "robust weight",
        "robust estimation",
    ),
    "positioning_accuracy": (
        "定位精度",
        "positioning accuracy",
        "position error",
        "位置误差",
        "rmse",
    ),
    "error_propagation": (
        "误差传播",
        "error propagation",
        "误差来源",
        "error source",
    ),
    "fixed_weighting": ("固定权", "fixed weight", "fixed weighting"),
    "method_result_relation": (
        "方法到结果",
        "method to result",
        "method outcome",
        "实验结果",
    ),
    "cross_paper_support": (
        "多篇论文",
        "across papers",
        "supporting study",
        "反例",
        "counterexample",
    ),
}


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


@dataclass(frozen=True, slots=True)
class QueryFrame:
    research_object: str
    current_method: tuple[str, ...]
    target_outcome: tuple[str, ...]
    required_dimensions: tuple[str, ...]
    optional_dimensions: tuple[str, ...] = ()
    excluded_dimensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DimensionCoverageReport:
    required_dimension_coverage: float
    direct_evidence_coverage: float
    source_diversity: int
    conflict_coverage: float
    relation_path_coverage: float
    covered_dimensions: tuple[str, ...]
    direct_dimensions: tuple[str, ...]
    conflict_dimensions: tuple[str, ...]
    relation_path_dimensions: tuple[str, ...]
    missing_dimensions: tuple[str, ...]
    overall_sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sufficient": self.overall_sufficient}


class QueryFrameBuilder:
    def build(self, query: str) -> QueryFrame:
        normalized = _normalize(query)
        dimensions = [
            dimension
            for dimension, terms in _DIMENSION_TERMS.items()
            if any(_normalize(term) in normalized for term in terms)
        ]
        # 伪距率用于提高定位精度时，状态可观性是关系链中的必需中间维度。
        if (
            "range_rate_observation" in dimensions
            and "positioning_accuracy" in dimensions
            and "state_observability" not in dimensions
        ):
            position = dimensions.index("range_rate_observation") + 1
            dimensions.insert(position, "state_observability")
        if not dimensions:
            dimensions = ["relation_mechanism"]
        methods = tuple(
            value
            for value in dimensions
            if value
            not in {
                "positioning_accuracy",
                "method_result_relation",
                "cross_paper_support",
            }
        )
        outcomes = tuple(
            value
            for value in dimensions
            if value in {"positioning_accuracy", "method_result_relation"}
        )
        object_name = "positioning" if "positioning_accuracy" in dimensions else "research_relation"
        return QueryFrame(
            research_object=object_name,
            current_method=methods,
            target_outcome=outcomes,
            required_dimensions=tuple(dimensions),
        )


class DimensionCoverageAnalyzer:
    @staticmethod
    def _evidence_dimensions(
        evidence: Mapping[str, Any], required: Sequence[str]
    ) -> set[str]:
        metadata = evidence.get("metadata")
        metadata_value = metadata if isinstance(metadata, Mapping) else {}
        declared: set[str] = set()
        for key in ("dimension_tags", "entities", "relation_labels"):
            raw = evidence.get(key, metadata_value.get(key))
            if isinstance(raw, (list, tuple, set)):
                declared.update(str(value) for value in raw)
        text = _normalize(
            " ".join(
                (
                    str(evidence.get("content") or ""),
                    " ".join(str(value) for value in evidence.get("relation_path", [])),
                    " ".join(declared),
                )
            )
        )
        matched = {
            dimension
            for dimension in required
            if dimension in declared
            or any(
                _normalize(term) in text
                for term in _DIMENSION_TERMS.get(dimension, ())
            )
        }
        if "relation_mechanism" in required and evidence.get("directness") in {
            "direct",
            "indirect",
        }:
            matched.add("relation_mechanism")
        return matched

    def evaluate(
        self,
        frame: QueryFrame,
        evidence: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]] = (),
    ) -> DimensionCoverageReport:
        required = tuple(
            value
            for value in frame.required_dimensions
            if value not in frame.excluded_dimensions
        )
        evidence_dimensions: dict[str, set[str]] = {}
        covered: set[str] = set()
        direct: set[str] = set()
        paths: set[str] = set()
        documents: set[str] = set()
        for value in evidence:
            evidence_id = str(value.get("evidence_id") or "")
            dimensions = self._evidence_dimensions(value, required)
            evidence_dimensions[evidence_id] = dimensions
            covered.update(dimensions)
            if value.get("directness") == "direct":
                direct.update(dimensions)
            relation_path = value.get("relation_path")
            if isinstance(relation_path, (list, tuple)) and len(relation_path) >= 2:
                paths.update(dimensions)
            document_id = value.get("document_id")
            if document_id:
                documents.add(str(document_id))
        conflict_dimensions: set[str] = set()
        for conflict in conflicts:
            ids = {
                str(conflict.get("supporting_evidence_id") or ""),
                str(conflict.get("opposing_evidence_id") or ""),
            }
            for evidence_id in ids:
                conflict_dimensions.update(evidence_dimensions.get(evidence_id, set()))
        denominator = max(1, len(required))
        missing = tuple(value for value in required if value not in covered)
        required_coverage = len(covered & set(required)) / denominator
        direct_coverage = len(direct & set(required)) / denominator
        path_coverage = len(paths & set(required)) / denominator
        conflict_required = bool(conflicts)
        conflict_coverage = (
            len(conflict_dimensions & set(required)) / denominator
            if conflict_required
            else 1.0
        )
        return DimensionCoverageReport(
            required_dimension_coverage=required_coverage,
            direct_evidence_coverage=direct_coverage,
            source_diversity=len(documents),
            conflict_coverage=conflict_coverage,
            relation_path_coverage=path_coverage,
            covered_dimensions=tuple(value for value in required if value in covered),
            direct_dimensions=tuple(value for value in required if value in direct),
            conflict_dimensions=tuple(
                value for value in required if value in conflict_dimensions
            ),
            relation_path_dimensions=tuple(value for value in required if value in paths),
            missing_dimensions=missing,
            overall_sufficient=(
                not missing
                and bool(direct & set(required))
                and (not conflict_required or conflict_coverage > 0)
            ),
        )

