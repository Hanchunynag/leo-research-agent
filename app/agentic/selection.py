"""Coverage-aware 最终证据选择：子问题覆盖、去冗余与来源多样性。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from app.agentic.models import CoverageReport, QueryPlan
from app.indexing.tokenization import tokenize


MAX_REDUNDANCY = 0.72


def _identity(candidate: dict[str, Any]) -> str:
    """返回优先使用稳定 Evidence ID 的候选标识。"""

    return str(candidate.get("evidence_id") or candidate.get("chunk_id") or "")


def _rank(candidate: dict[str, Any]) -> int:
    value = candidate.get("rank")
    return value if isinstance(value, int) and not isinstance(value, bool) else 10**9


def _directness(candidate: dict[str, Any]) -> int:
    value = candidate.get("directness_grade")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, min(3, value))
    return 0


def _features(candidate: dict[str, Any]) -> frozenset[str]:
    """使用英文词与中文字/双字特征，避免为 MMR 再做 Embedding。"""

    text = " ".join(
        (
            str(candidate.get("title") or ""),
            " ".join(str(value) for value in candidate.get("section_path") or []),
            str(candidate.get("content") or ""),
        )
    )
    return frozenset(tokenize(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class CoverageAwareEvidenceSelector:
    """在有限 Context 预算内先保覆盖，再用确定性 MMR 补足证据。"""

    def __init__(
        self,
        *,
        mmr_lambda: float = 0.75,
        max_per_work: int = 2,
        min_directness_grade: int = 1,
    ) -> None:
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda 必须在 0 到 1 之间。")
        if max_per_work < 1:
            raise ValueError("max_per_work 必须大于 0。")
        if min_directness_grade not in {0, 1, 2, 3}:
            raise ValueError("min_directness_grade 必须在 0 到 3 之间。")
        self.mmr_lambda = mmr_lambda
        self.max_per_work = max_per_work
        self.min_directness_grade = min_directness_grade
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _quality_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
        return (-_directness(candidate), _rank(candidate), _identity(candidate))

    def select(
        self,
        query: str,
        plan: QueryPlan,
        coverage: CoverageReport,
        candidates: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """返回最多 ``top_k`` 条证据，Coverage 必选项优先于多样性限制。"""

        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")
        by_id: dict[str, dict[str, Any]] = {}
        for value in candidates:
            candidate_id = _identity(value)
            if not candidate_id:
                continue
            current = by_id.get(candidate_id)
            if current is None or self._quality_key(value) < self._quality_key(current):
                by_id[candidate_id] = dict(value)

        ordered_candidates = sorted(by_id.values(), key=self._quality_key)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        selected_reasons: dict[str, list[str]] = {}
        selected_document_by_work: dict[str, str] = {}
        covered_subquestions: list[str] = []
        uncovered_subquestions: list[str] = []

        # Coverage 列表是 Planner 子问题的结构化结果；每项只占用一条最佳直接证据。
        coverage_bonus_ids: set[str] = set()
        for coverage_item in coverage.coverage:
            available = [
                by_id[evidence_id]
                for evidence_id in coverage_item.supporting_evidence_ids
                if evidence_id in by_id
            ]
            coverage_bonus_ids.update(
                evidence_id
                for evidence_id in coverage_item.supporting_evidence_ids
                if evidence_id in by_id
            )
            if not available:
                uncovered_subquestions.append(coverage_item.subquestion_id)
                continue
            duplicate_selected = next(
                (
                    chosen
                    for chosen in selected
                    if any(
                        str(candidate.get("content") or "").strip()
                        == str(chosen.get("content") or "").strip()
                        for candidate in available
                    )
                ),
                None,
            )
            compatible = [
                candidate
                for candidate in available
                if selected_document_by_work.get(
                    str(candidate.get("work_id") or "")
                )
                in {None, str(candidate.get("document_id") or "")}
            ]
            if duplicate_selected is None and not compatible:
                uncovered_subquestions.append(coverage_item.subquestion_id)
                continue
            best = duplicate_selected or min(compatible, key=self._quality_key)
            best_id = _identity(best)
            if best_id not in selected_ids:
                if len(selected) >= top_k:
                    uncovered_subquestions.append(coverage_item.subquestion_id)
                    continue
                selected.append(dict(best))
                selected_ids.add(best_id)
                selected_reasons[best_id] = []
                selected_document_by_work.setdefault(
                    str(best.get("work_id") or ""),
                    str(best.get("document_id") or ""),
                )
            selected_reasons[best_id].append(
                f"required_for_{coverage_item.subquestion_id}"
            )
            covered_subquestions.append(coverage_item.subquestion_id)

        feature_cache = {
            candidate_id: _features(candidate)
            for candidate_id, candidate in by_id.items()
        }
        query_features = frozenset(tokenize(query))
        work_counts = Counter(str(item.get("work_id") or "") for item in selected)
        selected_documents = {
            str(item.get("document_id") or "") for item in selected
        }

        while len(selected) < top_k:
            eligible = [
                item
                for item in ordered_candidates
                if _identity(item) not in selected_ids
                and _directness(item) >= self.min_directness_grade
                and work_counts[str(item.get("work_id") or "")] < self.max_per_work
                and selected_document_by_work.get(str(item.get("work_id") or ""))
                in {None, str(item.get("document_id") or "")}
                and max(
                    (
                        _jaccard(
                            feature_cache[_identity(item)],
                            feature_cache[_identity(chosen)],
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                < MAX_REDUNDANCY
            ]
            if not eligible:
                break

            scored: list[tuple[float, tuple[int, int, str], dict[str, Any]]] = []
            for item in eligible:
                candidate_id = _identity(item)
                features = feature_cache[candidate_id]
                relevance = (
                    0.55 * (_directness(item) / 3.0)
                    + 0.25 * (1.0 / max(1, _rank(item)))
                    + 0.10 * _jaccard(query_features, features)
                    + (0.10 if candidate_id in coverage_bonus_ids else 0.0)
                )
                redundancy = max(
                    (
                        _jaccard(features, feature_cache[_identity(chosen)])
                        for chosen in selected
                    ),
                    default=0.0,
                )
                work_id = str(item.get("work_id") or "")
                document_id = str(item.get("document_id") or "")
                source_bonus = (
                    (0.08 if work_counts[work_id] == 0 else 0.0)
                    + (0.04 if document_id not in selected_documents else 0.0)
                )
                score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                    + source_bonus
                )
                scored.append((score, self._quality_key(item), item))
            _, _, best = min(scored, key=lambda value: (-value[0], value[1]))
            best_id = _identity(best)
            selected.append(dict(best))
            selected_ids.add(best_id)
            selected_reasons[best_id] = ["mmr_fill"]
            work_counts[str(best.get("work_id") or "")] += 1
            selected_documents.add(str(best.get("document_id") or ""))
            selected_document_by_work.setdefault(
                str(best.get("work_id") or ""),
                str(best.get("document_id") or ""),
            )

        dropped: Counter[str] = Counter()
        for candidate in ordered_candidates:
            candidate_id = _identity(candidate)
            if candidate_id in selected_ids:
                continue
            if _directness(candidate) < self.min_directness_grade:
                dropped["low_directness"] += 1
            elif max(
                (
                    _jaccard(
                        feature_cache[candidate_id],
                        feature_cache[_identity(chosen)],
                    )
                    for chosen in selected
                ),
                default=0.0,
            ) >= MAX_REDUNDANCY:
                dropped["redundant"] += 1
            elif selected_document_by_work.get(
                str(candidate.get("work_id") or "")
            ) not in {None, str(candidate.get("document_id") or "")}:
                dropped["document_version_conflict"] += 1
            elif (
                work_counts[str(candidate.get("work_id") or "")]
                >= self.max_per_work
            ):
                dropped["per_work_limit"] += 1
            else:
                dropped["budget"] += 1

        required_ids = {
            candidate_id
            for candidate_id, reasons in selected_reasons.items()
            if any(reason.startswith("required_for_") for reason in reasons)
        }
        for index, candidate in enumerate(selected, 1):
            candidate_id = _identity(candidate)
            candidate["source_id"] = f"S{index}"
            candidate["selection_required"] = candidate_id in required_ids
            candidate["selection_reason"] = "+".join(
                selected_reasons[candidate_id]
            )

        planned_ids = [item.id for item in plan.subquestions]
        self.last_diagnostics = {
            "strategy": "coverage_mmr",
            "candidate_count": len(ordered_candidates),
            "selected_count": len(selected),
            "top_k_budget": top_k,
            "required_evidence_count": len(required_ids),
            "covered_subquestions": [
                value for value in planned_ids if value in covered_subquestions
            ],
            "uncovered_subquestions": [
                value for value in planned_ids if value in uncovered_subquestions
            ],
            "coverage_preserved": not uncovered_subquestions,
            "dropped_redundant_count": dropped["redundant"],
            "dropped_low_directness_count": dropped["low_directness"],
            "dropped_per_work_limit_count": dropped["per_work_limit"],
            "dropped_document_version_conflict_count": dropped[
                "document_version_conflict"
            ],
            "dropped_budget_count": dropped["budget"],
            "mmr_lambda": self.mmr_lambda,
            "max_per_work": self.max_per_work,
            "min_directness_grade": self.min_directness_grade,
            "selected": [
                {
                    "source_id": item["source_id"],
                    "evidence_id": item.get("evidence_id"),
                    "chunk_id": item.get("chunk_id"),
                    "work_id": item.get("work_id"),
                    "directness_grade": _directness(item),
                    "reason": item["selection_reason"],
                }
                for item in selected
            ],
        }
        return selected
