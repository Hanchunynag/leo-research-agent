"""Epoch-filtered FTS5 BM25 retrieval with metadata filters."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.index_registry.store import IndexRegistryStore
from app.indexing.lexical_fts import connect_lexical, lexical_index_path
from app.indexing.tokenization import normalize_search_text, tokenize


def _match_expression(query: str) -> str:
    normalized = normalize_search_text(query)
    terms = list(dict.fromkeys(tokenize(normalized)))
    if not terms:
        terms = re.findall(r"[\w\u3400-\u9fff]+", normalized)
    escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms if term]
    return " OR ".join(escaped)


def search_lexical_evidence(
    project_root: Path, query: str, limit: int = 10, *, epoch: int | None = None,
    work_id: str | None = None, document_id: str | None = None,
    content_zone: str | None = None, max_chunks_per_work: int = 2,
    path: Path | None = None,
) -> dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query cannot be empty")
    if not 1 <= limit <= 100 or not 1 <= max_chunks_per_work <= 20:
        raise ValueError("limit/max_chunks_per_work out of range")
    active_epoch = epoch or IndexRegistryStore(project_root).active_epoch()
    if active_epoch is None:
        raise RuntimeError("no active index epoch")
    expression = _match_expression(cleaned)
    if not expression:
        return {"query": cleaned, "result_count": 0, "results": []}
    sql = """
    SELECT m.chunk_json, bm25(chunk_fts, 3.0, 2.2, 1.2, 0.7, 1.5) AS raw_score
    FROM chunk_fts JOIN chunk_meta m ON m.rowid=chunk_fts.rowid
    WHERE chunk_fts MATCH ? AND m.valid_from_epoch <= ?
      AND (m.valid_to_epoch IS NULL OR ? < m.valid_to_epoch)
    """
    params: list[Any] = [expression, active_epoch, active_epoch]
    for column, value in (("work_id", work_id), ("document_id", document_id),
                          ("content_zone", content_zone)):
        if value:
            sql += f" AND m.{column}=?"
            params.append(value)
    sql += " ORDER BY raw_score LIMIT ?"
    params.append(max(limit * 8, 40))
    connection = connect_lexical(path or lexical_index_path(project_root))
    try:
        rows = connection.execute(sql, params).fetchall()
    finally:
        connection.close()
    phrase = normalize_search_text(cleaned)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        value = json.loads(row["chunk_json"])
        score = -float(row["raw_score"])
        searchable = normalize_search_text(" ".join((
            str(value.get("title") or ""), " ".join(value.get("section_path") or []),
            str(value.get("content") or ""))))
        if phrase and phrase in searchable:
            score *= 1.25
        ranked.append((score, value))
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("chunk_key"))))
    counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []
    for score, chunk in ranked:
        key = str(chunk.get("work_id") or chunk.get("document_id") or "")
        if counts[key] >= max_chunks_per_work:
            continue
        counts[key] += 1
        result = dict(chunk)
        result.update({"rank": len(results) + 1, "score": round(score, 8),
                       "retrieval_source": "lexical_fts",
                       "citation": f"{chunk.get('document_id')} pp. {chunk.get('page_start')}-{chunk.get('page_end')}"})
        results.append(result)
        if len(results) >= limit:
            break
    return {"query": cleaned, "retriever": "lexical_fts", "active_epoch": active_epoch,
            "result_count": len(results), "results": results}
