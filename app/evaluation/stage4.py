"""阶段四真实检索/关系评估 Schema 与纯确定性指标。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.evaluation.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True, slots=True)
class RelationQuestion:
    question_id: str
    workspace_id: str
    query: str
    question_type: str
    required_dimensions: tuple[str, ...]
    correct_relation_paths: tuple[tuple[str, ...], ...]
    allowed_indirect_paths: tuple[tuple[str, ...], ...]
    forbidden_relations: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    opposing_evidence_ids: tuple[str, ...]
    annotation_status: str
    annotator: str | None = None
    confirmed_at: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RelationQuestion:
        def text(name: str) -> str:
            raw = value.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{name} 必须是非空字符串。")
            return raw.strip()

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, [])
            if not isinstance(raw, list) or not all(
                isinstance(item, str) and item.strip() for item in raw
            ):
                raise ValueError(f"{name} 必须是字符串数组。")
            return tuple(dict.fromkeys(item.strip() for item in raw))

        def paths(name: str) -> tuple[tuple[str, ...], ...]:
            raw = value.get(name, [])
            if not isinstance(raw, list):
                raise ValueError(f"{name} 必须是路径数组。")
            output: list[tuple[str, ...]] = []
            for item in raw:
                if not isinstance(item, list) or len(item) < 2 or not all(
                    isinstance(part, str) and part.strip() for part in item
                ):
                    raise ValueError(f"{name} 中每条路径至少包含两个非空节点。")
                output.append(tuple(part.strip() for part in item))
            return tuple(output)

        status = text("annotation_status")
        if status not in {"confirmed", "pending_human_confirmation"}:
            raise ValueError("annotation_status 不受支持。")
        question = cls(
            question_id=text("question_id"),
            workspace_id=text("workspace_id"),
            query=text("query"),
            question_type=text("question_type"),
            required_dimensions=strings("required_dimensions"),
            correct_relation_paths=paths("correct_relation_paths"),
            allowed_indirect_paths=paths("allowed_indirect_paths"),
            forbidden_relations=strings("forbidden_relations"),
            source_chunk_ids=strings("source_chunk_ids"),
            supporting_evidence_ids=strings("supporting_evidence_ids"),
            opposing_evidence_ids=strings("opposing_evidence_ids"),
            annotation_status=status,
            annotator=(str(value["annotator"]).strip() if value.get("annotator") else None),
            confirmed_at=(str(value["confirmed_at"]).strip() if value.get("confirmed_at") else None),
            notes=str(value.get("notes") or "").strip(),
        )
        if question.annotation_status == "confirmed":
            if not question.annotator or not question.confirmed_at:
                raise ValueError("confirmed qrel 必须记录 annotator 和 confirmed_at。")
            if not (
                question.required_dimensions
                and question.correct_relation_paths
                and question.source_chunk_ids
                and question.supporting_evidence_ids
            ):
                raise ValueError("confirmed qrel 缺少人工确认的维度、路径、原文或支持证据。")
        return question


def load_relation_questions(path: Path) -> tuple[RelationQuestion, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[RelationQuestion] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
        try:
            value = RelationQuestion.from_mapping(raw)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if value.question_id in seen:
            raise ValueError(f"{path}:{line_number}: question_id 重复。")
        seen.add(value.question_id)
        values.append(value)
    if not values:
        raise ValueError("关系评估集不能为空。")
    return tuple(values)


def retrieval_quality(
    ranked_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(ranked_chunk_ids, relevant_chunk_ids, 5),
        "recall_at_10": recall_at_k(ranked_chunk_ids, relevant_chunk_ids, 10),
        "mrr": reciprocal_rank(ranked_chunk_ids, relevant_chunk_ids),
        "ndcg_at_10": ndcg_at_k(ranked_chunk_ids, relevant_chunk_ids, 10),
    }


def relation_quality(
    question: RelationQuestion,
    *,
    predicted_paths: Sequence[Sequence[str]],
    selected_evidence_ids: Sequence[str],
) -> dict[str, Any]:
    if question.annotation_status != "confirmed":
        return {
            "ground_truth_status": question.annotation_status,
            "relation_path_precision": None,
            "relation_path_coverage": None,
            "support_coverage": None,
            "opposition_coverage": None,
        }
    expected = set(question.correct_relation_paths) | set(question.allowed_indirect_paths)
    predicted = {tuple(value) for value in predicted_paths if len(value) >= 2}
    forbidden = {
        path for path in predicted if any(label in question.forbidden_relations for label in path)
    }
    correct = (predicted & expected) - forbidden
    selected = set(selected_evidence_ids)
    return {
        "ground_truth_status": "confirmed",
        "relation_path_precision": len(correct) / max(1, len(predicted)),
        "relation_path_coverage": len(correct) / max(1, len(question.correct_relation_paths)),
        "forbidden_relation_count": len(forbidden),
        "support_coverage": len(selected & set(question.supporting_evidence_ids))
        / max(1, len(question.supporting_evidence_ids)),
        "opposition_coverage": len(selected & set(question.opposing_evidence_ids))
        / max(1, len(question.opposing_evidence_ids)),
    }


def selection_distribution(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    documents = [str(value.get("document_id") or "") for value in values]
    counts = Counter(value for value in documents if value)
    total = sum(counts.values())
    return {
        "document_count": len(counts),
        "single_document_evidence_share": max(counts.values(), default=0) / max(1, total),
        "document_distribution": dict(sorted(counts.items())),
        "selected_evidence_tokens": sum(int(value.get("token_count") or 0) for value in values),
    }
