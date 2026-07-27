"""基于稳定论文身份和 block 标注的检索评测。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

from app.embeddings.base import EmbeddingProvider
from app.reranking.base import RerankerProvider
from app.retrieval.search import load_chunks, search_evidence
from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RetrievalQuestion:
    question_id: str
    question: str
    relevant_work_ids: list[str]
    relevant_document_ids: list[str]
    relevant_block_ids: list[str]
    question_type: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RetrievalQuestion:
        def required_text(field: str) -> str:
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field} 必须是非空字符串。")
            return item.strip()

        def string_list(field: str) -> list[str]:
            item = value.get(field, [])
            if not isinstance(item, list) or not all(
                isinstance(entry, str) and entry for entry in item
            ):
                raise ValueError(f"{field} 必须是非空字符串数组。")
            return list(dict.fromkeys(item))

        question = cls(
            question_id=required_text("question_id"),
            question=required_text("question"),
            relevant_work_ids=string_list("relevant_work_ids"),
            relevant_document_ids=string_list("relevant_document_ids"),
            relevant_block_ids=string_list("relevant_block_ids"),
            question_type=required_text("question_type"),
        )
        if not (
            question.relevant_block_ids
            or question.relevant_document_ids
            or question.relevant_work_ids
        ):
            raise ValueError("至少需要一种 relevant identity。")
        return question

    @property
    def target_level(self) -> str:
        if self.relevant_block_ids:
            return "block"
        if self.relevant_document_ids:
            return "document"
        return "work"

    @property
    def target_ids(self) -> set[str]:
        if self.relevant_block_ids:
            return set(self.relevant_block_ids)
        if self.relevant_document_ids:
            return set(self.relevant_document_ids)
        return set(self.relevant_work_ids)


def load_retrieval_questions(path: Path) -> list[RetrievalQuestion]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    questions: list[RetrievalQuestion] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number} 必须是 JSON 对象。")
        try:
            question = RetrievalQuestion.from_dict(value)
        except ValueError as error:
            raise ValueError(f"{resolved}:{line_number}: {error}") from error
        if question.question_id in seen_ids:
            raise ValueError(f"{resolved}:{line_number}: question_id 重复。")
        seen_ids.add(question.question_id)
        questions.append(question)
    if not questions:
        raise ValueError("检索评测集不能为空。")
    return questions


def chunk_evidence_ids(chunk: dict[str, Any], level: str) -> set[str]:
    if level == "work":
        work_id = chunk.get("work_id")
        return {work_id} if isinstance(work_id, str) else set()
    if level == "document":
        document_id = chunk.get("document_id")
        return {document_id} if isinstance(document_id, str) else set()
    if level != "block":
        raise ValueError(f"不支持的相关性层级：{level}")
    block_ids = {
        value for value in chunk.get("block_ids", []) if isinstance(value, str)
    }
    parent_contexts = chunk.get("parent_contexts")
    if isinstance(parent_contexts, list):
        for context in parent_contexts:
            if isinstance(context, dict):
                block_ids.update(
                    value
                    for value in context.get("block_ids", [])
                    if isinstance(value, str)
                )
    overlap_context = chunk.get("overlap_context")
    if isinstance(overlap_context, dict):
        block_ids.update(
            value
            for value in overlap_context.get("block_ids", [])
            if isinstance(value, str)
        )
    return block_ids


def relevant_chunk_ids(
    question: RetrievalQuestion,
    chunks: Sequence[dict[str, Any]],
) -> set[str]:
    relevant: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        if chunk_evidence_ids(chunk, question.target_level) & question.target_ids:
            relevant.add(chunk_id)
    return relevant


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, 1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], 1)
        if chunk_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


RankedRetriever = Callable[[str, int], list[dict[str, Any]]]


def evaluate_ranked_retriever(
    questions: Sequence[RetrievalQuestion],
    chunks: Sequence[dict[str, Any]],
    retrieve: RankedRetriever,
    retriever_name: str,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    normalized_k = sorted(set(k_values))
    if not normalized_k or normalized_k[0] < 1 or normalized_k[-1] > 100:
        raise ValueError("k_values 必须在 1 到 100 之间。")
    limit = normalized_k[-1]
    per_question: list[dict[str, Any]] = []
    metric_values: defaultdict[str, list[float]] = defaultdict(list)
    type_values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for question in questions:
        qrels = relevant_chunk_ids(question, chunks)
        if not qrels:
            raise ValueError(
                f"{question.question_id} 的标注在当前 chunks 中没有对应证据。"
            )
        results = retrieve(question.question, limit)
        ranked_ids = [
            chunk_id
            for result in results
            if isinstance((chunk_id := result.get("chunk_id")), str)
        ]
        metrics = {
            f"recall@{k}": recall_at_k(ranked_ids, qrels, k) for k in normalized_k
        }
        metrics["mrr"] = reciprocal_rank(ranked_ids, qrels)
        metrics[f"ndcg@{limit}"] = ndcg_at_k(ranked_ids, qrels, limit)
        for name, value in metrics.items():
            metric_values[name].append(value)
            type_values[question.question_type][name].append(value)
        first_relevant_rank = next(
            (rank for rank, chunk_id in enumerate(ranked_ids, 1) if chunk_id in qrels),
            None,
        )
        per_question.append(
            {
                **asdict(question),
                "target_level": question.target_level,
                "relevant_chunk_ids": sorted(qrels),
                "retrieved_chunk_ids": ranked_ids,
                "first_relevant_rank": first_relevant_rank,
                "metrics": metrics,
            }
        )

    def averages(values: dict[str, list[float]]) -> dict[str, float]:
        return {
            name: round(sum(items) / len(items), 6)
            for name, items in sorted(values.items())
        }

    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "retriever": retriever_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "k_values": normalized_k,
        "metrics": averages(metric_values),
        "metrics_by_question_type": {
            question_type: averages(values)
            for question_type, values in sorted(type_values.items())
        },
        "per_question": per_question,
    }


def evaluate_bm25(
    project_root: Path,
    questions_path: Path,
    output_path: Path | None = None,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)

    def retrieve(question: str, limit: int) -> list[dict[str, Any]]:
        result = search_evidence(
            project_root=root,
            query=question,
            limit=limit,
            max_chunks_per_work=20,
        )
        raw_results = result.get("results")
        return (
            [value for value in raw_results if isinstance(value, dict)]
            if isinstance(raw_results, list)
            else []
        )

    report = evaluate_ranked_retriever(
        questions=questions,
        chunks=chunks,
        retrieve=retrieve,
        retriever_name="bm25",
        k_values=k_values,
    )
    report["questions_path"] = str(questions_path.expanduser().resolve())
    if output_path is not None:
        resolved_output = output_path.expanduser().resolve()
        report["output_path"] = str(resolved_output)
        write_json_atomic(resolved_output, report)
    return report


def evaluate_dense(
    project_root: Path,
    questions_path: Path,
    provider: EmbeddingProvider,
    output_path: Path | None = None,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    from app.retrieval.dense import search_dense_evidence

    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)

    def retrieve(question: str, limit: int) -> list[dict[str, Any]]:
        result = search_dense_evidence(
            project_root=root,
            provider=provider,
            query=question,
            limit=limit,
            max_chunks_per_work=20,
        )
        raw_results = result.get("results")
        return (
            [value for value in raw_results if isinstance(value, dict)]
            if isinstance(raw_results, list)
            else []
        )

    report = evaluate_ranked_retriever(
        questions=questions,
        chunks=chunks,
        retrieve=retrieve,
        retriever_name="dense",
        k_values=k_values,
    )
    report["model_name"] = getattr(provider, "model_name", None)
    report["model_revision"] = getattr(provider, "revision", None)
    report["questions_path"] = str(questions_path.expanduser().resolve())
    if output_path is not None:
        resolved_output = output_path.expanduser().resolve()
        report["output_path"] = str(resolved_output)
        write_json_atomic(resolved_output, report)
    return report


def evaluate_hybrid_rrf(
    project_root: Path,
    questions_path: Path,
    provider: EmbeddingProvider,
    output_path: Path | None = None,
    k_values: Sequence[int] = (1, 5, 10),
    candidate_limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    from app.retrieval.hybrid import search_hybrid_evidence

    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)

    def retrieve(question: str, limit: int) -> list[dict[str, Any]]:
        result = search_hybrid_evidence(
            project_root=root,
            provider=provider,
            query=question,
            limit=limit,
            max_chunks_per_work=20,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
        )
        raw_results = result.get("results")
        return (
            [value for value in raw_results if isinstance(value, dict)]
            if isinstance(raw_results, list)
            else []
        )

    report = evaluate_ranked_retriever(
        questions=questions,
        chunks=chunks,
        retrieve=retrieve,
        retriever_name="hybrid_rrf",
        k_values=k_values,
    )
    report["model_name"] = getattr(provider, "model_name", None)
    report["model_revision"] = getattr(provider, "revision", None)
    report["candidate_limit_per_source"] = candidate_limit
    report["rrf_k"] = rrf_k
    report["questions_path"] = str(questions_path.expanduser().resolve())
    if output_path is not None:
        resolved_output = output_path.expanduser().resolve()
        report["output_path"] = str(resolved_output)
        write_json_atomic(resolved_output, report)
    return report


def evaluate_candidate_pool_oracle(
    project_root: Path,
    questions_path: Path,
    provider: EmbeddingProvider,
    output_path: Path | None = None,
    candidate_limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """测量 BM25∪Dense 候选池及其 RRF Top-N 的理论召回上限。"""

    from app.retrieval.dense import search_dense_evidence
    from app.retrieval.hybrid import reciprocal_rank_fusion

    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)
    per_question: list[dict[str, Any]] = []
    union_recalls: list[float] = []
    rrf_recalls: list[float] = []
    union_sizes: list[int] = []
    for question in questions:
        qrels = relevant_chunk_ids(question, chunks)
        if not qrels:
            raise ValueError(
                f"{question.question_id} 的标注在当前 chunks 中没有对应证据。"
            )
        bm25 = search_evidence(
            project_root=root,
            query=question.question,
            limit=candidate_limit,
            max_chunks_per_work=20,
        )
        dense = search_dense_evidence(
            project_root=root,
            provider=provider,
            query=question.question,
            limit=candidate_limit,
            max_chunks_per_work=20,
        )
        bm25_results = [
            value for value in bm25.get("results", []) if isinstance(value, dict)
        ]
        dense_results = [
            value for value in dense.get("results", []) if isinstance(value, dict)
        ]
        union_ids = {
            chunk_id
            for result in [*bm25_results, *dense_results]
            if isinstance((chunk_id := result.get("chunk_id")), str)
        }
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_results, "dense": dense_results},
            rrf_k=rrf_k,
            limit=candidate_limit,
        )
        rrf_ids = {
            chunk_id
            for result in fused
            if isinstance((chunk_id := result.get("chunk_id")), str)
        }
        union_hits = qrels & union_ids
        rrf_hits = qrels & rrf_ids
        union_recall = len(union_hits) / len(qrels)
        rrf_recall = len(rrf_hits) / len(qrels)
        union_recalls.append(union_recall)
        rrf_recalls.append(rrf_recall)
        union_sizes.append(len(union_ids))
        per_question.append(
            {
                **asdict(question),
                "target_level": question.target_level,
                "relevant_chunk_ids": sorted(qrels),
                "bm25_candidate_ids": [
                    result.get("chunk_id") for result in bm25_results
                ],
                "dense_candidate_ids": [
                    result.get("chunk_id") for result in dense_results
                ],
                "union_candidate_count": len(union_ids),
                "union_relevant_ids": sorted(union_hits),
                "union_oracle_recall": union_recall,
                "rrf_candidate_ids": [result.get("chunk_id") for result in fused],
                "rrf_relevant_ids": sorted(rrf_hits),
                "rrf_pool_oracle_recall": rrf_recall,
            }
        )

    def pool_summary(values: list[float]) -> dict[str, float]:
        count = len(values)
        return {
            "mean_recall": round(sum(values) / count, 6),
            "any_hit_rate": round(sum(value > 0 for value in values) / count, 6),
            "full_recall_rate": round(sum(value == 1 for value in values) / count, 6),
        }

    report: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "retriever": "candidate_pool_oracle",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "candidate_limit_per_source": candidate_limit,
        "rrf_k": rrf_k,
        "model_name": getattr(provider, "model_name", None),
        "model_revision": getattr(provider, "revision", None),
        "union_top_n_each": {
            **pool_summary(union_recalls),
            "mean_candidate_count": round(sum(union_sizes) / len(union_sizes), 3),
        },
        "rrf_top_n": pool_summary(rrf_recalls),
        "questions_path": str(questions_path.expanduser().resolve()),
        "per_question": per_question,
    }
    if output_path is not None:
        resolved_output = output_path.expanduser().resolve()
        report["output_path"] = str(resolved_output)
        write_json_atomic(resolved_output, report)
    return report


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(median, 3),
        "p95": round(ordered[p95_index], 3),
    }


def evaluate_reranked(
    project_root: Path,
    questions_path: Path,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
    output_path: Path | None = None,
    k_values: Sequence[int] = (1, 5, 10),
    candidate_limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """在同一 qrels 上评测 RRF Top-N 的 Cross-Encoder 精排。"""

    from app.retrieval.reranked import search_reranked_evidence

    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)
    embedding_warmup_started = perf_counter()
    embedding_warmup_vector = embedding_provider.embed_query(
        "warmup retrieval query"
    )
    embedding_warmup_ms = (perf_counter() - embedding_warmup_started) * 1000
    if not embedding_warmup_vector:
        raise RuntimeError("Embedding warmup 返回了空向量。")
    reranker_warmup_started = perf_counter()
    warmup_scores = reranker_provider.score(
        "warmup relevance query",
        ["warmup candidate document"],
    )
    reranker_warmup_ms = (perf_counter() - reranker_warmup_started) * 1000
    if len(warmup_scores) != 1:
        raise RuntimeError("Reranker warmup 返回的分数数量不正确。")

    diagnostics: dict[str, dict[str, Any]] = {}

    def retrieve(question: str, limit: int) -> list[dict[str, Any]]:
        result = search_reranked_evidence(
            project_root=root,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
            query=question,
            limit=candidate_limit,
            max_chunks_per_work=20,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
        )
        raw_results = result.get("results")
        values = (
            [value for value in raw_results if isinstance(value, dict)]
            if isinstance(raw_results, list)
            else []
        )
        diagnostics[question] = {
            "candidate_count": result.get("candidate_count"),
            "timing": result.get("timing"),
            "results": values,
        }
        return values[:limit]

    report = evaluate_ranked_retriever(
        questions=questions,
        chunks=chunks,
        retrieve=retrieve,
        retriever_name="hybrid_rrf_reranked",
        k_values=k_values,
    )
    candidate_ms: list[float] = []
    reranking_ms: list[float] = []
    total_ms: list[float] = []
    total_pairs = 0
    for question_report in report["per_question"]:
        question_text = str(question_report["question"])
        diagnostic = diagnostics[question_text]
        results = diagnostic["results"]
        qrels = set(question_report["relevant_chunk_ids"])
        rrf_rank = min(
            (
                int(result["rrf_rank"])
                for result in results
                if result.get("chunk_id") in qrels
                and isinstance(result.get("rrf_rank"), int)
            ),
            default=None,
        )
        reranked_rank = next(
            (
                rank
                for rank, result in enumerate(results, 1)
                if result.get("chunk_id") in qrels
            ),
            None,
        )
        rank_delta = (
            reranked_rank - rrf_rank
            if reranked_rank is not None and rrf_rank is not None
            else None
        )
        timing = diagnostic.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        question_report["reranking_diagnostics"] = {
            "rrf_candidate_first_relevant_rank": rrf_rank,
            "reranked_candidate_first_relevant_rank": reranked_rank,
            "rank_delta": rank_delta,
            "candidate_count": diagnostic["candidate_count"],
            "timing": timing,
        }
        candidate_ms.append(float(timing.get("candidate_retrieval_ms", 0.0)))
        reranking_ms.append(float(timing.get("reranking_ms", 0.0)))
        total_ms.append(float(timing.get("total_ms", 0.0)))
        total_pairs += int(diagnostic["candidate_count"] or 0)

    reranking_seconds = sum(reranking_ms) / 1000
    total_seconds = sum(total_ms) / 1000
    report.update(
        {
            "embedding_model_name": getattr(embedding_provider, "model_name", None),
            "embedding_model_revision": getattr(
                embedding_provider, "revision", None
            ),
            "reranker_model_name": getattr(reranker_provider, "model_name", None),
            "reranker_model_revision": getattr(
                reranker_provider, "revision", None
            ),
            "reranker_max_length": getattr(reranker_provider, "max_length", None),
            "candidate_limit": candidate_limit,
            "rrf_k": rrf_k,
            "performance": {
                "embedding_warmup_ms": round(embedding_warmup_ms, 3),
                "reranker_warmup_ms": round(reranker_warmup_ms, 3),
                "candidate_retrieval_ms": _latency_summary(candidate_ms),
                "reranking_ms": _latency_summary(reranking_ms),
                "total_ms": _latency_summary(total_ms),
                "total_pairs": total_pairs,
                "pairs_per_second": round(
                    total_pairs / reranking_seconds if reranking_seconds else 0.0,
                    3,
                ),
                "queries_per_second": round(
                    len(questions) / total_seconds if total_seconds else 0.0,
                    3,
                ),
            },
            "questions_path": str(questions_path.expanduser().resolve()),
        }
    )
    if output_path is not None:
        resolved_output = output_path.expanduser().resolve()
        report["output_path"] = str(resolved_output)
        write_json_atomic(resolved_output, report)
    return report
