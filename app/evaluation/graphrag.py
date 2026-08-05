"""GraphRAG-specific, per-question-type evaluation and optional RAGAS adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ABLATIONS = (
    "legacy_single_query", "graphrag_single_query", "graphrag_multi_query",
    "graphrag_multi_query_without_drift_guard", "graphrag_without_graph_path",
    "graphrag_without_community_retrieval", "graphrag_without_reranker",
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall(predicted: set[str], reference: set[str]) -> dict[str, float]:
    overlap = len(predicted & reference)
    return {"precision": _ratio(overlap, len(predicted)),
            "recall": _ratio(overlap, len(reference))}


def reciprocal_rank(values: list[str], references: set[str]) -> float:
    return next((1.0 / rank for rank, value in enumerate(values, 1)
                 if value in references), 0.0)


def evaluate_records(references: Iterable[dict[str, Any]],
                     predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reference_map = {str(value["question_id"]): value for value in references}
    prediction_map = {str(value["question_id"]): value for value in predictions}
    rows: list[dict[str, Any]] = []
    for question_id, reference in reference_map.items():
        prediction = prediction_map.get(question_id, {})
        expected_entities = set(map(str, reference.get("reference_entity_ids") or []))
        expected_claims = set(map(str, reference.get("reference_claim_ids") or []))
        predicted_entities = set(map(str, prediction.get("entity_ids") or []))
        predicted_claims = list(map(str, prediction.get("claim_ids") or []))
        entity = precision_recall(predicted_entities, expected_entities)
        relation_recall = _ratio(len(set(predicted_claims) & expected_claims), len(expected_claims))
        expected_paths = {tuple(map(str, path)) for path in reference.get("reference_paths") or []}
        predicted_paths = {tuple(map(str, path)) for path in prediction.get("paths") or []}
        path_recall = _ratio(len(expected_paths & predicted_paths), len(expected_paths))
        edge_claims = prediction.get("path_edge_claim_ids") or []
        path_completeness = _ratio(sum(bool(edge) for path in edge_claims for edge in path),
                                   sum(len(path) for path in edge_claims))
        answerable = bool(reference.get("answerable"))
        refused = not bool(prediction.get("answerable", False))
        row = {
            "question_id": question_id, "question_type": reference.get("question_type"),
            "entity_linking_accuracy": entity["recall"],
            "entity_resolution_precision": entity["precision"],
            "entity_resolution_recall": entity["recall"],
            "relation_extraction_recall": relation_recall,
            "direct_relation_recall_at_k": relation_recall,
            "graph_path_recall_at_k": path_recall,
            "path_evidence_completeness": path_completeness,
            "no_relation_refusal_accuracy": float(answerable != refused),
            "contradiction_coverage": float(bool(prediction.get("contradictions")))
                if reference.get("question_type") == "contradiction" else 1.0,
            "community_retrieval_recall": reciprocal_rank(
                list(map(str, prediction.get("community_ids") or [])),
                set(map(str, reference.get("reference_community_ids") or []))),
            "query_drift_rejection_accuracy": float(bool(prediction.get("drift_rejected")))
                if reference.get("question_type") == "query_drift_trap" else 1.0,
            "cross_query_candidate_oracle_recall": _ratio(len(
                set(map(str, prediction.get("candidate_chunk_ids") or [])) &
                set(map(str, reference.get("reference_chunk_ids") or []))),
                len(set(map(str, reference.get("reference_chunk_ids") or [])))),
        }
        rows.append(row)
    metric_names = [key for key in rows[0] if key not in {"question_id", "question_type"}] if rows else []
    overall = {metric: sum(float(row[metric]) for row in rows) / len(rows)
               for metric in metric_names} if rows else {}
    by_type: dict[str, dict[str, float]] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(row)
    for question_type, values in grouped.items():
        by_type[question_type] = {metric: sum(float(row[metric]) for row in values) / len(values)
                                  for metric in metric_names}
    return {"question_count": len(rows), "overall": overall,
            "by_question_type": by_type, "questions": rows,
            "required_ablations": list(ABLATIONS)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evaluate_files(references_path: Path, predictions_path: Path,
                   output_path: Path | None = None) -> dict[str, Any]:
    report = evaluate_records(load_jsonl(references_path), load_jsonl(predictions_path))
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_ragas(samples: list[dict[str, Any]], *, llm: Any, embeddings: Any) -> dict[str, Any]:
    """Run the five required RAGAS metrics; imports stay isolated from runtime RAG."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
    )
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=[ContextPrecision(), ContextRecall(),
        Faithfulness(), AnswerRelevancy(), AnswerCorrectness()], llm=llm,
        embeddings=embeddings)
    return result.to_pandas().to_dict(orient="records")
