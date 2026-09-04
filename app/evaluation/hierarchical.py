"""Hierarchical RAG 的 Paper→Evidence 级联评测。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from app.embeddings.base import EmbeddingProvider
from app.evaluation.retrieval import (
    RetrievalQuestion,
    load_retrieval_questions,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_ids,
)
from app.indexing.paper import load_paper_records
from app.retrieval.hierarchical import search_hierarchical_evidence
from app.retrieval.search import load_chunks
from app.reranking.base import RerankerProvider
from app.storage import write_json_atomic


EVALUATION_SCHEMA_VERSION = "1.0"


def _paper_ids_for_question(
    question: RetrievalQuestion,
    chunks: Sequence[dict[str, Any]],
) -> set[str]:
    relevant_chunks = relevant_chunk_ids(question, chunks)
    papers: set[str] = set()
    for chunk in chunks:
        if chunk.get("chunk_id") not in relevant_chunks:
            continue
        paper_id = chunk.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            papers.add(paper_id)
    return papers


def _ids(values: Sequence[dict[str, Any]], key: str) -> list[str]:
    return [
        str(value[key])
        for value in values
        if isinstance(value.get(key), str) and value[key]
    ]


def _ranking_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    *,
    prefix: str,
    k_values: Sequence[int],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in k_values:
        values[f"{prefix}_recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        values[f"{prefix}_precision@{k}"] = (
            len(set(ranked_ids[:k]) & relevant_ids) / max(1, min(k, len(ranked_ids)))
        )
    values[f"{prefix}_mrr"] = reciprocal_rank(ranked_ids, relevant_ids)
    values[f"{prefix}_ndcg@{max(k_values)}"] = ndcg_at_k(
        ranked_ids, relevant_ids, max(k_values)
    )
    return values


def _average(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> dict[str, float]:
    return {
        name: round(
            sum(float(row.get(name) or 0.0) for row in rows) / max(1, len(rows)),
            6,
        )
        for name in names
    }


def evaluate_hierarchical(
    project_root: Path,
    questions_path: Path,
    embedding_provider: EmbeddingProvider,
    *,
    reranker_provider: RerankerProvider | None = None,
    output_path: Path | None = None,
    k_values: Sequence[int] = (1, 5, 10),
    limit: int = 10,
    paper_limit: int = 10,
    paper_candidate_limit: int = 30,
    chunk_candidate_limit: int = 40,
    max_chunks_per_work: int = 2,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """在同一 qrels 上评估两层检索及其级联损失。"""

    normalized_k = sorted(set(int(value) for value in k_values))
    if not normalized_k or normalized_k[0] < 1 or normalized_k[-1] > 100:
        raise ValueError("k_values 必须在 1 到 100 之间。")
    if limit < 1 or paper_limit < 1:
        raise ValueError("limit/paper_limit 必须大于 0。")

    root = project_root.expanduser().resolve()
    questions = load_retrieval_questions(questions_path)
    chunks = load_chunks(root)
    papers = load_paper_records(root)
    paper_ids_from_documents = {
        str(value.get("document_id")): str(value.get("paper_id"))
        for value in papers
        if value.get("document_id") and value.get("paper_id")
    }
    rows: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    metric_names: list[str] = []
    for question in questions:
        gold_chunks = relevant_chunk_ids(question, chunks)
        if not gold_chunks:
            raise ValueError(f"{question.question_id} 在当前 chunks 中没有有效 qrel。")
        gold_papers = _paper_ids_for_question(question, chunks)
        if not gold_papers:
            gold_papers = {
                paper_ids_from_documents[value]
                for value in question.relevant_document_ids
                if value in paper_ids_from_documents
            }
        started = perf_counter()
        result = search_hierarchical_evidence(
            root,
            embedding_provider,
            question.question,
            reranker_provider=reranker_provider,
            limit=limit,
            paper_limit=paper_limit,
            paper_candidate_limit=paper_candidate_limit,
            chunk_candidate_limit=chunk_candidate_limit,
            max_chunks_per_work=max_chunks_per_work,
            rrf_k=rrf_k,
        )
        elapsed = (perf_counter() - started) * 1000
        latency_ms.append(elapsed)

        paper_stage = result.get("paper_retrieval")
        paper_stage = paper_stage if isinstance(paper_stage, dict) else {}
        paper_ranked = [
            value for value in paper_stage.get("results", [])
            if isinstance(value, dict)
        ]
        paper_ids = _ids(paper_ranked, "paper_id")
        evidence = [
            value for value in result.get("results", [])
            if isinstance(value, dict)
        ]
        evidence_ids = _ids(evidence, "chunk_id")
        candidate_paper_ids = {
            str(value)
            for value in result.get("candidate_paper_ids", [])
            if isinstance(value, str) and value
        }
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "gold_paper_ids": sorted(gold_papers),
            "gold_chunk_ids": sorted(gold_chunks),
            "paper_ranked_ids": paper_ids,
            "evidence_ranked_ids": evidence_ids,
            "candidate_paper_count": len(candidate_paper_ids),
            "evidence_count": len(evidence),
            "scope_leakage_count": sum(
                1
                for value in evidence
                if str(value.get("paper_id") or "") not in candidate_paper_ids
            ),
            "paper_stage_recall": float(bool(candidate_paper_ids & gold_papers)),
            "cascade_recall": len(set(evidence_ids) & gold_chunks) / max(1, len(gold_chunks)),
            "latency_ms": round(elapsed, 3),
            "no_hit": not bool(evidence),
        }
        row.update(_ranking_metrics(paper_ids, gold_papers, prefix="paper", k_values=normalized_k))
        row.update(_ranking_metrics(evidence_ids, gold_chunks, prefix="evidence", k_values=normalized_k))

        branch_results = paper_stage.get("branch_results")
        if isinstance(branch_results, dict):
            for branch_name in ("bm25", "dense"):
                branch = [
                    value for value in branch_results.get(branch_name, [])
                    if isinstance(value, dict)
                ]
                branch_ids = _ids(branch, "paper_id")
                row.update(_ranking_metrics(
                    branch_ids,
                    gold_papers,
                    prefix=f"paper_{branch_name}",
                    k_values=normalized_k,
                ))
        metric_names = [
            key for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"latency_ms"}
        ]
        rows.append(row)

    metric_names = sorted(set(metric_names))
    scope_leakage = sum(int(row["scope_leakage_count"]) for row in rows)
    report: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluator": "hierarchical_paper_to_evidence",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "k_values": normalized_k,
        "paper_limit": paper_limit,
        "paper_candidate_limit_per_source": paper_candidate_limit,
        "chunk_candidate_limit_per_source": chunk_candidate_limit,
        "rrf_k": rrf_k,
        "embedding_model_name": getattr(embedding_provider, "model_name", None),
        "embedding_model_revision": getattr(embedding_provider, "revision", None),
        "reranker_model_name": getattr(reranker_provider, "model_name", None),
        "reranker_model_revision": getattr(reranker_provider, "revision", None),
        "metrics": {
            **_average(rows, metric_names),
            "scope_leakage_count": scope_leakage,
            "scope_leakage_rate": round(scope_leakage / max(1, sum(row["evidence_count"] for row in rows)), 6),
            "paper_survival_rate": round(sum(row["paper_stage_recall"] for row in rows) / max(1, len(rows)), 6),
            "mean_latency_ms": round(sum(latency_ms) / max(1, len(latency_ms)), 3),
            "p95_latency_ms": round(sorted(latency_ms)[max(0, int(len(latency_ms) * 0.95) - 1)], 3),
        },
        "metrics_by_question_type": {
            question_type: _average(
                [row for row in rows if row["question_type"] == question_type],
                metric_names,
            )
            for question_type in sorted({str(row["question_type"]) for row in rows})
        },
        "per_question": rows,
    }
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        report["output_path"] = str(resolved)
        write_json_atomic(resolved, report)
    return report
