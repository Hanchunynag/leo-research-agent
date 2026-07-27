"""无需外部服务的确定性 BM25 索引。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.indexing.tokenization import tokenize
from app.storage import write_json_atomic


BM25_SCHEMA_VERSION = "2.0"


def chunks_digest(chunks: list[dict[str, Any]]) -> str:
    payload = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "title": chunk.get("title"),
            "section_path": chunk.get("section_path"),
            "content": chunk.get("content"),
            "parent_contexts": chunk.get("parent_contexts"),
            "overlap_context": chunk.get("overlap_context"),
        }
        for chunk in chunks
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def searchable_chunk_text(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title") or "")
    section_path = chunk.get("section_path")
    section = (
        " ".join(value for value in section_path if isinstance(value, str))
        if isinstance(section_path, list)
        else ""
    )
    content = str(chunk.get("content") or "")
    parent_contexts = chunk.get("parent_contexts")
    parent_parts: list[str] = []
    if isinstance(parent_contexts, list):
        for context in parent_contexts:
            if not isinstance(context, dict):
                continue
            path = context.get("section_path")
            if isinstance(path, list):
                parent_parts.append(
                    " ".join(value for value in path if isinstance(value, str))
                )
            parent_parts.append(str(context.get("content") or ""))
    overlap_context = chunk.get("overlap_context")
    overlap_content = (
        str(overlap_context.get("content") or "")
        if isinstance(overlap_context, dict)
        else ""
    )
    # 标题和章节名称重复一次，给予论文/章节级术语适度权重。
    return "\n".join(
        (title, title, section, section, *parent_parts, overlap_content, content)
    )


def build_bm25_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    postings: dict[str, list[list[int]]] = {}
    documents: list[dict[str, Any]] = []
    total_length = 0
    for index, chunk in enumerate(chunks):
        tokens = tokenize(searchable_chunk_text(chunk))
        frequencies = Counter(tokens)
        total_length += len(tokens)
        documents.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "work_id": chunk.get("work_id"),
                "document_id": chunk.get("document_id"),
                "content_zone": chunk.get("content_zone"),
                "length": len(tokens),
            }
        )
        for term, frequency in frequencies.items():
            postings.setdefault(term, []).append([index, frequency])
    return {
        "bm25_schema_version": BM25_SCHEMA_VERSION,
        "chunks_digest": chunks_digest(chunks),
        "document_count": len(documents),
        "average_document_length": (
            total_length / len(documents) if documents else 0.0
        ),
        "documents": documents,
        "postings": postings,
    }


def bm25_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "index" / "bm25.json"


def write_bm25_index(project_root: Path, index: dict[str, Any]) -> Path:
    output = bm25_index_path(project_root)
    write_json_atomic(output, index)
    return output
