"""BM25 证据检索、元数据过滤和 work_id 去重。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.indexing.bm25 import (
    bm25_index_path,
    chunks_digest,
    searchable_chunk_text,
)
from app.indexing.tokenization import normalize_search_text, tokenize


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return payload


def load_chunks(project_root: Path) -> list[dict[str, Any]]:
    path = project_root.expanduser().resolve() / "data" / "knowledge" / "chunks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    chunks: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
        chunks.append(value)
    return chunks


def _validate_limit(value: int, field: str, maximum: int = 100) -> int:
    if isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{field} 必须在 1 到 {maximum} 之间。")
    return value


def search_evidence(
    project_root: Path,
    query: str,
    limit: int = 10,
    work_id: str | None = None,
    document_id: str | None = None,
    max_chunks_per_work: int = 2,
) -> dict[str, Any]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空。")
    validated_limit = _validate_limit(limit, "limit")
    per_work = _validate_limit(max_chunks_per_work, "max_chunks_per_work", 20)
    root = project_root.expanduser().resolve()
    chunks = load_chunks(root)
    index = load_json_object(bm25_index_path(root))
    if index.get("chunks_digest") != chunks_digest(chunks):
        raise RuntimeError("BM25 索引与 chunks.jsonl 不一致，请重新构建 knowledge。")
    documents = index.get("documents")
    postings = index.get("postings")
    if not isinstance(documents, list) or not isinstance(postings, dict):
        raise ValueError("BM25 索引结构无效。")

    query_tokens = list(dict.fromkeys(tokenize(cleaned_query)))
    scores: defaultdict[int, float] = defaultdict(float)
    document_count = int(index.get("document_count", 0))
    average_length = float(index.get("average_document_length", 0.0)) or 1.0
    k1 = 1.5
    b = 0.75
    for term in query_tokens:
        raw_postings = postings.get(term)
        if not isinstance(raw_postings, list) or not raw_postings:
            continue
        document_frequency = len(raw_postings)
        inverse_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for raw_posting in raw_postings:
            if (
                not isinstance(raw_posting, list)
                or len(raw_posting) != 2
                or not all(isinstance(value, int) for value in raw_posting)
            ):
                continue
            index_value, frequency = raw_posting
            if index_value >= len(documents):
                continue
            document = documents[index_value]
            if not isinstance(document, dict):
                continue
            if work_id and document.get("work_id") != work_id:
                continue
            if document_id and document.get("document_id") != document_id:
                continue
            length = int(document.get("length", 0))
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            scores[index_value] += inverse_frequency * (
                frequency * (k1 + 1) / denominator
            )

    normalized_query = normalize_search_text(cleaned_query)
    ranked: list[tuple[int, float]] = []
    for index_value, score in scores.items():
        if index_value >= len(chunks):
            continue
        chunk = chunks[index_value]
        haystack = normalize_search_text(searchable_chunk_text(chunk))
        if normalized_query in haystack:
            score *= 1.2
        ranked.append((index_value, score))
    ranked.sort(key=lambda value: (-value[1], str(chunks[value[0]].get("chunk_id"))))

    work_counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []
    for index_value, score in ranked:
        chunk = chunks[index_value]
        result_work_id = str(chunk.get("work_id") or "")
        if work_counts[result_work_id] >= per_work:
            continue
        work_counts[result_work_id] += 1
        results.append(
            {
                "rank": len(results) + 1,
                "score": round(score, 6),
                "chunk_id": chunk.get("chunk_id"),
                "work_id": chunk.get("work_id"),
                "document_id": chunk.get("document_id"),
                "paper_id": chunk.get("paper_id"),
                "title": chunk.get("title"),
                "authors": chunk.get("authors"),
                "year": chunk.get("year"),
                "doi": chunk.get("doi"),
                "section_path": chunk.get("section_path"),
                "content_zone": chunk.get("content_zone"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "block_ids": chunk.get("block_ids"),
                "content_types": chunk.get("content_types"),
                "parent_contexts": chunk.get("parent_contexts") or [],
                "overlap_context": chunk.get("overlap_context"),
                "content": chunk.get("content"),
                "citation": (
                    f"{chunk.get('document_id')} pp. "
                    f"{chunk.get('page_start')}-{chunk.get('page_end')}"
                ),
            }
        )
        if len(results) >= validated_limit:
            break
    return {
        "query": cleaned_query,
        "query_tokens": query_tokens,
        "result_count": len(results),
        "work_id_filter": work_id,
        "document_id_filter": document_id,
        "max_chunks_per_work": per_work,
        "results": results,
    }
