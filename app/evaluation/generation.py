"""科研 RAG 生成质量评测与可选 RAGAS 适配。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.1"
RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)


def _ids(values: Any, key: str | None = None) -> set[str]:
    if not isinstance(values, list):
        return set()
    result: set[str] = set()
    for value in values:
        raw = value.get(key) if key and isinstance(value, Mapping) else value
        if isinstance(raw, list):
            result.update(item for item in raw if isinstance(item, str) and item)
        elif isinstance(raw, str) and raw:
            result.add(raw)
    return result


def _claim_units(values: Any) -> list[dict[str, Any]]:
    """Return structurally usable Claim Units for scope analysis."""

    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            continue
        claim_id = str(value.get("claim_id") or f"C{index}").strip()
        text = value.get("text")
        # Keep compatibility with the original compact evaluation fixture,
        # which represented a single claim only by its evidence IDs. Full
        # production records still carry claim text and explicit claim IDs.
        compact_claim = isinstance(value.get("evidence_ids"), list)
        if claim_id and (
            (isinstance(text, str) and text.strip()) or compact_claim
        ):
            result.append(dict(value, claim_id=claim_id))
    return result


def _citation_scope_analysis(
    record: Mapping[str, Any],
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Analyze Claim Unit -> Supporting Evidence -> Citation Binding.

    Refusals and generation failures with no Claim Unit are not citation-scope
    samples. Claim-bearing answers remain strict: every emitted Citation must
    bind to its declared claim and Selected Evidence, and every claim must have
    a valid binding.
    """

    selected_ids = _ids(record.get("selected_evidence"), "evidence_id") | _ids(
        record.get("selected_evidence"), "chunk_id"
    )
    claims = _claim_units(record.get("claims"))
    raw_citations = record.get("citations")
    citations = (
        [value for value in raw_citations if isinstance(value, Mapping)]
        if isinstance(raw_citations, list)
        else []
    )
    claims_by_id = {str(value["claim_id"]): value for value in claims}
    bindings: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for citation in citations:
        claim_id = str(citation.get("claim_id") or "").strip()
        if not claim_id and len(claims) == 1:
            # Legacy one-claim records had no explicit binding key. Do not
            # apply this fallback when multiple claims could be ambiguous.
            claim_id = str(claims[0]["claim_id"])
        evidence_id = str(
            citation.get("evidence_id") or citation.get("chunk_id") or ""
        ).strip()
        claim = claims_by_id.get(claim_id)
        declared_ids = _ids(claim.get("evidence_ids")) if claim else set()
        reasons: list[str] = []
        if claim is None:
            reasons.append("unknown_claim_id")
        if not evidence_id:
            reasons.append("missing_evidence_id")
        elif evidence_id not in selected_ids:
            reasons.append("evidence_outside_selected_scope")
        elif (
            claim is not None
            and declared_ids
            and evidence_id not in declared_ids
        ):
            reasons.append("evidence_not_declared_by_claim")
        binding = {
            "claim_id": claim_id,
            "evidence_id": evidence_id,
            "valid": not reasons,
            "issues": reasons,
        }
        bindings.append(binding)
        if reasons:
            issues.append({"kind": "citation_binding", **binding})

    bound_claim_ids = {
        str(value["claim_id"])
        for value in bindings
        if value["valid"] and value["claim_id"] in claims_by_id
    }
    unbound_claims = [
        {
            "claim_id": str(claim["claim_id"]),
            "supporting_evidence_ids": sorted(_ids(claim.get("evidence_ids"))),
            "reason": "claim_has_no_valid_citation_binding",
        }
        for claim in claims
        if str(claim["claim_id"]) not in bound_claim_ids
    ]
    issues.extend({"kind": "claim_binding", **value} for value in unbound_claims)

    if not claims:
        outcome = record.get("outcome")
        outcome_code = outcome.get("code") if isinstance(outcome, Mapping) else None
        analysis = {
            "status": "not_applicable",
            "reason": str(outcome_code or "no_claim_units"),
            "claim_unit_count": 0,
            "binding_count": len(bindings),
            "valid_binding_count": sum(value["valid"] for value in bindings),
            "bound_claim_count": 0,
            "unbound_claims": [],
            "bindings": bindings,
            "issues": issues,
        }
        return None, None, analysis

    valid_binding_count = sum(value["valid"] for value in bindings)
    scope_precision = valid_binding_count / max(1, len(bindings))
    binding_coverage = len(bound_claim_ids) / max(1, len(claims))
    analysis = {
        "status": "pass" if not issues else "fail",
        "claim_unit_count": len(claims),
        "binding_count": len(bindings),
        "valid_binding_count": valid_binding_count,
        "bound_claim_count": len(bound_claim_ids),
        "unbound_claims": unbound_claims,
        "bindings": bindings,
        "issues": issues,
    }
    return scope_precision, binding_coverage, analysis


def evaluate_grounded_generation(
    records: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """评估无需 Judge LLM 的引用边界与证据覆盖指标。

    这组指标不冒充 Faithfulness 或 Answer Correctness：后两者应使用
    RAGAS/人工 Judge。这里专门保证论文 RAG 的 citation contract 可回归。
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        selected = record.get("selected_evidence")
        selected_ids = _ids(selected, "evidence_id") | _ids(selected, "chunk_id")
        citations = record.get("citations")
        citation_ids = _ids(citations, "evidence_id") | _ids(citations, "chunk_id")
        gold_ids = _ids(record.get("reference_evidence_ids"))
        cited_gold = citation_ids & gold_ids
        scope_precision, binding_coverage, scope_analysis = _citation_scope_analysis(record)
        row: dict[str, Any] = {
            "question_id": str(record.get("question_id") or ""),
            "citation_scope_precision": scope_precision,
            "citation_binding_coverage": binding_coverage,
            "citation_precision": (
                len(cited_gold) / max(1, len(citation_ids))
                if gold_ids and citation_ids
                else None
            ),
            "citation_recall": (
                len(cited_gold) / max(1, len(gold_ids))
                if gold_ids and citation_ids
                else None
            ),
            "context_recall_proxy": (
                len(selected_ids & gold_ids) / max(1, len(gold_ids)) if gold_ids else None
            ),
            "citation_metadata_complete": float(
                all(
                    isinstance(value, Mapping)
                    and value.get("page_start") is not None
                    and value.get("page_end") is not None
                    and value.get("evidence_id") in selected_ids
                    for value in citations or []
                )
            ),
            "citation_scope_analysis": scope_analysis,
        }
        expected_answerable = record.get("expected_answerable")
        if isinstance(expected_answerable, bool):
            row["answerable_accuracy"] = float(
                bool(record.get("answerable")) == expected_answerable
            )
        rows.append(row)

    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
        | {
            "citation_scope_precision",
            "citation_binding_coverage",
            "citation_precision",
            "citation_recall",
            "context_recall_proxy",
        }
    )
    metric_values = {
        name: [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
        ]
        for name in metric_names
    }
    report: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluator": "grounded_generation_contract",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "metrics": {
            name: round(sum(values) / len(values), 6) if values else None
            for name, values in metric_values.items()
        },
        "per_question": rows,
        "citation_scope": {
            "evaluated_question_count": sum(
                row["citation_scope_analysis"]["status"] != "not_applicable"
                for row in rows
            ),
            "not_applicable_question_count": sum(
                row["citation_scope_analysis"]["status"] == "not_applicable"
                for row in rows
            ),
            "failure_cases": [
                {
                    "question_id": row["question_id"],
                    "analysis": row["citation_scope_analysis"],
                }
                for row in rows
                if row["citation_scope_analysis"]["status"] == "fail"
            ],
            "known_limitations": [
                "拒答或生成失败且没有 Claim Unit 的样本不参与 Citation Scope Precision 均值，但会保留 outcome 和 N/A 计数。",
                "该离线评测检查结构化 Claim/Citation 绑定与 Selected Evidence 边界，不替代语义 Judge 的 entailment 或人工判断。",
            ],
        },
        "ragas_metrics_available": list(RAGAS_METRICS),
    }
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        report["output_path"] = str(resolved)
        write_json_atomic(resolved, report)
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} 每行必须是 JSON 对象。")
        values.append(value)
    if not values:
        raise ValueError("生成评测集不能为空。")
    return values


def evaluate_generation_files(
    predictions_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    return evaluate_grounded_generation(load_jsonl(predictions_path), output_path=output_path)


def run_ragas(
    samples: list[dict[str, Any]],
    *,
    llm: Any,
    embeddings: Any,
) -> dict[str, Any]:
    """可选 Judge LLM 评测；运行时不强制外部服务可用。"""

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
        ],
        llm=llm,
        embeddings=embeddings,
    )
    return {
        "framework": "ragas",
        "metrics": list(RAGAS_METRICS),
        "rows": cast(Any, result).to_pandas().to_dict(orient="records"),
    }
