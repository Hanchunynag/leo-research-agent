"""科研 RAG 生成质量评测与可选 RAGAS 适配。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.0"
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
        claims = record.get("claims")
        claim_ids = _ids(claims, "evidence_ids")
        citations = record.get("citations")
        citation_ids = _ids(citations, "evidence_id") | claim_ids
        gold_ids = _ids(record.get("reference_evidence_ids"))
        cited_selected = citation_ids & selected_ids
        cited_gold = citation_ids & gold_ids
        row: dict[str, Any] = {
            "question_id": str(record.get("question_id") or ""),
            "citation_scope_precision": len(cited_selected) / max(1, len(citation_ids)),
            "citation_precision": (
                len(cited_gold) / max(1, len(citation_ids)) if gold_ids else None
            ),
            "citation_recall": (
                len(cited_gold) / max(1, len(gold_ids)) if gold_ids else None
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
        }
        expected_answerable = record.get("expected_answerable")
        if isinstance(expected_answerable, bool):
            row["answerable_accuracy"] = float(
                bool(record.get("answerable")) == expected_answerable
            )
        rows.append(row)

    metric_names = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float))
    })
    report: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluator": "grounded_generation_contract",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "metrics": {
            name: round(
                sum(float(row.get(name) or 0.0) for row in rows) / max(1, len(rows)),
                6,
            )
            for name in metric_names
        },
        "per_question": rows,
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
