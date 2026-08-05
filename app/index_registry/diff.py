"""Stable chunk identity and incremental change classification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from app.index_registry.models import ChunkDiff
from app.indexing.dense import dense_chunk_text


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_block_ids(chunk: Mapping[str, Any]) -> list[str]:
    values = chunk.get("block_ids")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def stable_chunk_key(chunk: Mapping[str, Any]) -> str:
    """Hash structural identity, not sequence position or display chunk_id."""

    document_id = str(chunk.get("document_id") or "")
    section_id = str(chunk.get("section_id") or "")
    policy = str(chunk.get("chunk_policy_version") or "")
    if not document_id or not section_id or not policy:
        raise ValueError("stable chunk_key requires document_id, section_id and policy")
    payload = "\x1f".join(
        (document_id, section_id, "\x1e".join(_ordered_block_ids(chunk)), policy)
    )
    return sha256_text(payload)


def graph_extraction_text(chunk: Mapping[str, Any]) -> str:
    section_path = chunk.get("section_path")
    section = " > ".join(map(str, section_path)) if isinstance(section_path, list) else ""
    return "\n".join(
        part
        for part in (
            f"Title: {chunk.get('title') or ''}",
            f"Section: {section}",
            f"Content: {chunk.get('content') or ''}",
        )
        if part.strip()
    )


def versioned_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(chunk)
    value["chunk_key"] = stable_chunk_key(value)
    value["content_hash"] = sha256_text(str(value.get("content") or ""))
    value["dense_text_hash"] = sha256_text(dense_chunk_text(value))
    value["graph_text_hash"] = sha256_text(graph_extraction_text(value))
    return value


def diff_chunks(
    current_chunks: Iterable[Mapping[str, Any]],
    previous_by_key: Mapping[str, Mapping[str, Any]],
) -> list[ChunkDiff]:
    current = {value["chunk_key"]: value for value in map(versioned_chunk, current_chunks)}
    diffs: list[ChunkDiff] = []
    for key in sorted(current):
        value = current[key]
        old = previous_by_key.get(key)
        if old is None:
            diffs.append(ChunkDiff("added", key, value, None, True, True))
            continue
        dense_changed = value["dense_text_hash"] != old.get("dense_text_hash")
        graph_changed = value["graph_text_hash"] != old.get("graph_text_hash")
        content_changed = value["content_hash"] != old.get("content_hash")
        if dense_changed and graph_changed:
            kind = "changed"
        elif dense_changed:
            kind = "dense_changed"
        elif graph_changed or content_changed:
            kind = "graph_changed"
        else:
            kind = "unchanged"
        diffs.append(ChunkDiff(kind, key, value, dict(old), dense_changed, graph_changed))
    for key in sorted(set(previous_by_key) - set(current)):
        diffs.append(ChunkDiff("deleted", key, None, dict(previous_by_key[key])))
    return diffs


def chunk_json(chunk: Mapping[str, Any]) -> str:
    return json.dumps(dict(chunk), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
