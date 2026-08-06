"""集中式 LightRAG official cutover 门槛与失败分类。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CutoverAcceptanceConfig:
    minimum_retrieval_questions: int
    minimum_confirmed_relation_questions: int
    workspace_leakage_max: int
    excluded_evidence_leakage_max: int
    source_backfill_rate_min: float
    direct_qa_ndcg_ratio_min: float
    relation_path_precision_min: float
    relation_coverage_ratio_min: float
    unrelated_reprocessed_documents_max: int
    delete_residual_max: int
    require_legacy_rollback_smoke: bool
    require_failure_diagnostics: bool

    @classmethod
    def load(cls, path: Path) -> CutoverAcceptanceConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema_version") != "1.0":
            raise ValueError("Cutover Acceptance Schema 不受支持。")
        thresholds = value.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("Cutover Acceptance 缺少 thresholds。")
        return cls(**{field: thresholds[field] for field in cls.__dataclass_fields__})


def evaluate_cutover(
    config: CutoverAcceptanceConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    def section(name: str) -> Mapping[str, Any]:
        value = report.get(name)
        return value if isinstance(value, Mapping) else {}

    dataset = section("dataset")
    quality = section("quality")
    safety = section("safety")
    incremental = section("incremental")
    deletion = section("deletion")
    rollback = section("rollback")
    diagnostics = section("diagnostics")

    legacy_ndcg = quality.get("legacy_direct_ndcg_at_10")
    shadow_ndcg = quality.get("lightrag_direct_ndcg_at_10")
    legacy_relation = quality.get("legacy_relation_coverage")
    shadow_relation = quality.get("lightrag_relation_coverage")

    checks = {
        "retrieval_dataset_complete": int(dataset.get("retrieval_question_count") or 0)
        >= config.minimum_retrieval_questions,
        "relation_dataset_confirmed": int(dataset.get("confirmed_relation_question_count") or 0)
        >= config.minimum_confirmed_relation_questions,
        "workspace_leakage": int(safety.get("workspace_leakage_count") or 0)
        <= config.workspace_leakage_max,
        "excluded_evidence_leakage": int(safety.get("excluded_evidence_leakage_count") or 0)
        <= config.excluded_evidence_leakage_max,
        "source_backfill": isinstance(safety.get("source_backfill_rate"), (int, float))
        and float(safety["source_backfill_rate"]) >= config.source_backfill_rate_min,
        "direct_qa_ndcg": isinstance(legacy_ndcg, (int, float))
        and isinstance(shadow_ndcg, (int, float))
        and float(shadow_ndcg) >= float(legacy_ndcg) * config.direct_qa_ndcg_ratio_min,
        "relation_path_precision": isinstance(quality.get("relation_path_precision"), (int, float))
        and float(quality["relation_path_precision"]) >= config.relation_path_precision_min,
        "relation_coverage": isinstance(legacy_relation, (int, float))
        and isinstance(shadow_relation, (int, float))
        and float(shadow_relation) >= float(legacy_relation) * config.relation_coverage_ratio_min,
        "incremental_no_unrelated_reprocess": incremental.get("full_rebuild") is False
        and int(incremental.get("unrelated_reprocessed_document_count") or 0)
        <= config.unrelated_reprocessed_documents_max,
        "delete_no_residual": int(deletion.get("residual_result_count", -1))
        == config.delete_residual_max,
        "legacy_rollback_smoke": (not config.require_legacy_rollback_smoke)
        or rollback.get("legacy_top_k_restored") is True,
        "failure_diagnostics": (not config.require_failure_diagnostics)
        or diagnostics.get("all_failures_classified") is True,
    }
    categories = {
        "retrieval_dataset_complete": "dataset",
        "relation_dataset_confirmed": "dataset",
        "workspace_leakage": "scope",
        "excluded_evidence_leakage": "scope",
        "source_backfill": "mapping",
        "direct_qa_ndcg": "reranking",
        "relation_path_precision": "relation",
        "relation_coverage": "relation",
        "incremental_no_unrelated_reprocess": "indexing",
        "delete_no_residual": "deletion",
        "legacy_rollback_smoke": "rollback",
        "failure_diagnostics": "diagnostics",
    }
    failures = [
        {"check": name, "category": categories[name]}
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "official_cutover_approved": not failures,
    }
