"""基于章节边界生成确定性、可追溯的结构化 Chunk。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.indexing.tokenization import token_count
from app.storage import write_json_atomic


CHUNK_SCHEMA_VERSION = "2.0"
CHUNK_POLICY_VERSION = "2.1"
DEFAULT_MAX_TOKENS = 700
DEFAULT_MIN_CHUNK_TOKENS = 80
DEFAULT_OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class _Unit:
    block_id: str
    page_number: int
    block_type: str
    content: str
    token_count: int
    related_text_block_ids: list[str]


def _split_sentences(value: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+", value.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_oversized_text(value: str, maximum_tokens: int) -> list[str]:
    if token_count(value) <= maximum_tokens:
        return [value]
    sentences = _split_sentences(value)
    if len(sentences) > 1:
        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            count = token_count(sentence)
            if current and current_tokens + count > maximum_tokens:
                parts.append(" ".join(current))
                current = []
                current_tokens = 0
            if count > maximum_tokens:
                parts.extend(_split_oversized_text(sentence, maximum_tokens))
            else:
                current.append(sentence)
                current_tokens += count
        if current:
            parts.append(" ".join(current))
        return parts

    words = value.split()
    if len(words) <= 1:
        character_window = max(maximum_tokens * 3, 200)
        return [
            value[index : index + character_window]
            for index in range(0, len(value), character_window)
        ]
    parts = []
    current_words: list[str] = []
    current_tokens = 0
    for word in words:
        count = max(token_count(word), 1)
        if current_words and current_tokens + count > maximum_tokens:
            parts.append(" ".join(current_words))
            current_words = []
            current_tokens = 0
        current_words.append(word)
        current_tokens += count
    if current_words:
        parts.append(" ".join(current_words))
    return parts


def _block_units(block: dict[str, Any], maximum_tokens: int) -> list[_Unit]:
    block_id = block.get("block_id")
    content = block.get("content")
    if not isinstance(block_id, str) or not isinstance(content, str) or not content:
        return []
    related = block.get("related_text_block_ids")
    related_ids = (
        [value for value in related if isinstance(value, str)]
        if isinstance(related, list)
        else []
    )
    return [
        _Unit(
            block_id=block_id,
            page_number=int(block.get("page_number", 0)),
            block_type=str(block.get("type", "paragraph")),
            content=part,
            token_count=token_count(part),
            related_text_block_ids=related_ids,
        )
        for part in _split_oversized_text(content, maximum_tokens)
        if part
    ]


def chunk_input_fingerprint(
    structure: dict[str, Any],
    maximum_tokens: int,
    minimum_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> str:
    payload = {
        "policy": CHUNK_POLICY_VERSION,
        "maximum_tokens": maximum_tokens,
        "minimum_chunk_tokens": minimum_chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "structure_fingerprint": structure.get("input_fingerprint"),
        "blocks": structure.get("blocks"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_parent_path(parent: list[str], child: list[str]) -> bool:
    return bool(parent) and len(child) > len(parent) and child[: len(parent)] == parent


def _context_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_type": "parent_section",
        "work_id": chunk.get("work_id"),
        "document_id": chunk.get("document_id"),
        "paper_id": chunk.get("paper_id"),
        "section_id": chunk.get("section_id"),
        "section_path": chunk.get("section_path") or [],
        "content_zone": chunk.get("content_zone"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "block_ids": chunk.get("block_ids") or [],
        "related_text_block_ids": chunk.get("related_text_block_ids") or [],
        "content_types": chunk.get("content_types") or [],
        "content": chunk.get("content") or "",
        "token_count": int(chunk.get("token_count", 0)),
    }


def _can_absorb_parent_chunk(
    current: dict[str, Any],
    following: dict[str, Any],
    minimum_chunk_tokens: int,
) -> bool:
    if minimum_chunk_tokens <= 0:
        return False
    content_types = current.get("content_types")
    if not isinstance(content_types, list) or not set(content_types) <= {
        "paragraph",
        "list",
    }:
        return False
    parent_path = current.get("section_path")
    child_path = following.get("section_path")
    return (
        int(current.get("token_count", 0)) < minimum_chunk_tokens
        and current.get("section_id") is not None
        and current.get("content_zone") == following.get("content_zone")
        and isinstance(parent_path, list)
        and isinstance(child_path, list)
        and all(isinstance(value, str) for value in parent_path)
        and all(isinstance(value, str) for value in child_path)
        and _is_parent_path(parent_path, child_path)
    )


def _absorb_small_parent_chunks(
    chunks: list[dict[str, Any]],
    chunk_units: list[list[_Unit]],
    minimum_chunk_tokens: int,
) -> tuple[list[dict[str, Any]], list[list[_Unit]], int]:
    remaining_chunks = list(chunks)
    remaining_units = list(chunk_units)
    absorbed_count = 0
    index = 0
    while index + 1 < len(remaining_chunks):
        current = remaining_chunks[index]
        following = remaining_chunks[index + 1]
        if not _can_absorb_parent_chunk(
            current,
            following,
            minimum_chunk_tokens,
        ):
            index += 1
            continue
        inherited = current.get("parent_contexts")
        parent_contexts = (
            [value for value in inherited if isinstance(value, dict)]
            if isinstance(inherited, list)
            else []
        )
        following["parent_contexts"] = [
            *parent_contexts,
            _context_from_chunk(current),
        ]
        remaining_chunks.pop(index)
        remaining_units.pop(index)
        absorbed_count += 1
        if index:
            index -= 1
    return remaining_chunks, remaining_units, absorbed_count


def _tail_text_by_tokens(value: str, maximum_tokens: int) -> str:
    if maximum_tokens <= 0:
        return ""
    if token_count(value) <= maximum_tokens:
        return value
    sentences = _split_sentences(value)
    selected_sentences: list[str] = []
    selected_tokens = 0
    for sentence in reversed(sentences):
        count = token_count(sentence)
        if selected_sentences and selected_tokens + count > maximum_tokens:
            break
        if count <= maximum_tokens - selected_tokens:
            selected_sentences.append(sentence)
            selected_tokens += count
        elif not selected_sentences:
            break
    if selected_sentences:
        return " ".join(reversed(selected_sentences))

    selected_words: list[str] = []
    selected_tokens = 0
    for word in reversed(value.split()):
        count = max(token_count(word), 1)
        if selected_words and selected_tokens + count > maximum_tokens:
            break
        if count > maximum_tokens:
            continue
        selected_words.append(word)
        selected_tokens += count
    if selected_words:
        return " ".join(reversed(selected_words))

    for start in range(max(len(value) - maximum_tokens * 4, 0), len(value)):
        candidate = value[start:]
        if token_count(candidate) <= maximum_tokens:
            return candidate
    return ""


def _overlap_context(
    units: list[_Unit],
    overlap_tokens: int,
) -> dict[str, Any] | None:
    if overlap_tokens <= 0:
        return None
    selected: list[_Unit] = []
    remaining = overlap_tokens
    for unit in reversed(units):
        if unit.token_count <= remaining:
            selected.append(unit)
            remaining -= unit.token_count
            if remaining == 0:
                break
            continue
        excerpt = _tail_text_by_tokens(unit.content, remaining)
        if excerpt:
            selected.append(
                _Unit(
                    block_id=unit.block_id,
                    page_number=unit.page_number,
                    block_type=unit.block_type,
                    content=excerpt,
                    token_count=token_count(excerpt),
                    related_text_block_ids=unit.related_text_block_ids,
                )
            )
        break
    if not selected:
        return None
    selected.reverse()
    content = "\n\n".join(unit.content for unit in selected)
    pages = [unit.page_number for unit in selected]
    return {
        "context_type": "same_section_overlap",
        "page_start": min(pages),
        "page_end": max(pages),
        "block_ids": list(dict.fromkeys(unit.block_id for unit in selected)),
        "content": content,
        "token_count": token_count(content),
    }


def _attach_overlap_contexts(
    chunks: list[dict[str, Any]],
    chunk_units: list[list[_Unit]],
    overlap_tokens: int,
) -> int:
    overlap_count = 0
    for index, chunk in enumerate(chunks):
        chunk["overlap_context"] = None
        if index == 0:
            continue
        previous = chunks[index - 1]
        if (
            chunk.get("section_id") is None
            or chunk.get("section_id") != previous.get("section_id")
            or chunk.get("content_zone") != previous.get("content_zone")
        ):
            continue
        context = _overlap_context(chunk_units[index - 1], overlap_tokens)
        if context is None:
            continue
        context["section_id"] = previous.get("section_id")
        context["section_path"] = previous.get("section_path") or []
        context["work_id"] = previous.get("work_id")
        context["document_id"] = previous.get("document_id")
        context["paper_id"] = previous.get("paper_id")
        context["source_chunk_id"] = previous.get("chunk_id")
        chunk["overlap_context"] = context
        overlap_count += 1
    return overlap_count


def build_chunks(
    structure: dict[str, Any],
    maximum_tokens: int = DEFAULT_MAX_TOKENS,
    minimum_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> dict[str, Any]:
    if maximum_tokens < 50:
        raise ValueError("maximum_tokens 不能小于 50。")
    if minimum_chunk_tokens < 0 or minimum_chunk_tokens >= maximum_tokens:
        raise ValueError("minimum_chunk_tokens 必须大于等于 0 且小于 maximum_tokens。")
    if overlap_tokens < 0 or overlap_tokens >= maximum_tokens:
        raise ValueError("overlap_tokens 必须大于等于 0 且小于 maximum_tokens。")
    document_id = structure.get("document_id")
    work_id = structure.get("work_id")
    paper_id = structure.get("paper_id")
    title = structure.get("title")
    if not all(
        isinstance(value, str) and value
        for value in (document_id, work_id, paper_id, title)
    ):
        raise ValueError("structure 缺少 work/document/paper identity 或 title。")
    blocks = structure.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("structure.blocks 必须是数组。")

    chunks: list[dict[str, Any]] = []
    chunk_units: list[list[_Unit]] = []
    current_units: list[_Unit] = []
    current_section_id: str | None = None
    current_section_path: list[str] = []
    current_zone: str | None = None
    current_tokens = 0

    def flush() -> None:
        nonlocal current_units, current_tokens
        if not current_units:
            return
        block_ids = list(dict.fromkeys(unit.block_id for unit in current_units))
        related_ids = list(
            dict.fromkeys(
                related_id
                for unit in current_units
                for related_id in unit.related_text_block_ids
            )
        )
        pages = [unit.page_number for unit in current_units]
        content_types = list(dict.fromkeys(unit.block_type for unit in current_units))
        content = "\n\n".join(unit.content for unit in current_units)
        chunks.append(
            {
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "chunk_policy_version": CHUNK_POLICY_VERSION,
                "chunk_id": None,
                "work_id": work_id,
                "document_id": document_id,
                "paper_id": paper_id,
                "title": title,
                "authors": structure.get("authors") or [],
                "year": structure.get("year"),
                "doi": structure.get("doi"),
                "section_id": current_section_id,
                "section_path": current_section_path,
                "content_zone": current_zone,
                "page_start": min(pages),
                "page_end": max(pages),
                "block_ids": block_ids,
                "related_text_block_ids": related_ids,
                "content_types": content_types,
                "content": content,
                "token_count": token_count(content),
                "parent_contexts": [],
                "overlap_context": None,
            }
        )
        chunk_units.append(list(current_units))
        current_units = []
        current_tokens = 0

    for value in blocks:
        if not isinstance(value, dict) or value.get("searchable") is not True:
            continue
        section_id = value.get("section_id")
        section_path = value.get("section_path")
        zone = value.get("content_zone")
        normalized_path = (
            [item for item in section_path if isinstance(item, str)]
            if isinstance(section_path, list)
            else []
        )
        if current_units and (section_id != current_section_id or zone != current_zone):
            flush()
        if not current_units:
            current_section_id = section_id if isinstance(section_id, str) else None
            current_section_path = normalized_path
            current_zone = zone if isinstance(zone, str) else None
        for unit in _block_units(value, maximum_tokens):
            if current_units and current_tokens + unit.token_count > maximum_tokens:
                flush()
                current_section_id = section_id if isinstance(section_id, str) else None
                current_section_path = normalized_path
                current_zone = zone if isinstance(zone, str) else None
            current_units.append(unit)
            current_tokens += unit.token_count
    flush()

    chunks, chunk_units, absorbed_count = _absorb_small_parent_chunks(
        chunks,
        chunk_units,
        minimum_chunk_tokens,
    )
    for chunk_index, chunk in enumerate(chunks, 1):
        chunk["chunk_id"] = f"{document_id}_cp02_c{chunk_index:06d}"
    overlap_context_count = _attach_overlap_contexts(
        chunks,
        chunk_units,
        overlap_tokens,
    )

    return {
        "chunk_collection_schema_version": "2.0",
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "input_fingerprint": chunk_input_fingerprint(
            structure,
            maximum_tokens,
            minimum_chunk_tokens,
            overlap_tokens,
        ),
        "structure_fingerprint": structure.get("input_fingerprint"),
        "paper_id": paper_id,
        "document_id": document_id,
        "work_id": work_id,
        "title": title,
        "maximum_tokens": maximum_tokens,
        "minimum_chunk_tokens": minimum_chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "absorbed_parent_chunk_count": absorbed_count,
        "overlap_context_count": overlap_context_count,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def document_chunks_path(project_root: Path, document_id: str) -> Path:
    return (
        project_root.expanduser().resolve()
        / "data"
        / "knowledge"
        / "chunks"
        / f"{document_id}.chunks.json"
    )


def write_document_chunks(project_root: Path, payload: dict[str, Any]) -> Path:
    document_id = payload.get("document_id")
    if not isinstance(document_id, str):
        raise ValueError("chunk collection document_id 不能为空。")
    output = document_chunks_path(project_root, document_id)
    write_json_atomic(output, payload)
    return output
