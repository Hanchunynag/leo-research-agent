"""从 MinerU canonical blocks 恢复内容区域、章节路径和资产关系。"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from app.storage import write_json_atomic


STRUCTURE_SCHEMA_VERSION = "1.0"
STRUCTURE_POLICY_VERSION = "1.2"
SEARCHABLE_ZONES = {"abstract", "main_body", "appendix"}
ASSET_TYPES = {"equation", "figure", "table", "algorithm"}
TEXT_TYPES = {"paragraph", "list", "algorithm"}


class _Section(TypedDict):
    section_id: str
    title: str
    level: int
    section_path: list[str]
    content_zone: str
    page_start: int
    page_end: int
    heading_block_id: str | None
    block_ids: list[str]


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", without_tags).strip()


def normalized_heading(value: str) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"^(?:[IVXLCDM]+|\d+(?:\.\d+)*|[A-Z])[.)]\s*", "", text)
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def content_zone_for_heading(value: str, current_zone: str) -> str:
    heading = normalized_heading(value)
    if heading in {"TABLE OF CONTENTS", "CONTENTS"}:
        return "table_of_contents"
    if heading.startswith("ABSTRACT"):
        return "abstract"
    if heading.startswith("BIOGRAPH") or heading.startswith("ABOUT THE AUTHOR"):
        return "biography"
    if heading.startswith("ACKNOWLEDG"):
        return "acknowledgments"
    if heading.startswith("REFERENCE") or heading.startswith("BIBLIOGRAPHY"):
        return "references"
    if heading.startswith("APPENDIX"):
        return "appendix"
    if "INTRODUCTION" in heading:
        return "main_body"
    if current_zone in {"front_matter", "table_of_contents", "biography"}:
        return current_zone
    return current_zone


def is_false_heading(value: str) -> bool:
    heading = clean_text(value)
    return bool(
        re.match(
            r"^(?:fig(?:ure)?\.?|table|algorithm|equation)\s*\(?"
            r"(?:\d+|[IVXLCDM]+|[A-Z])(?:\b|[.():-])",
            heading,
            flags=re.IGNORECASE,
        )
    )


def infer_heading_level(value: str) -> int:
    heading = clean_text(value)
    roman = re.match(r"^[IVXLCDM]+[.)]\s+", heading, flags=re.IGNORECASE)
    if roman:
        return 1
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", heading)
    if numbered:
        return min(numbered.group(1).count(".") + 1, 3)
    if re.match(r"^[A-Z][.)]\s+", heading):
        return 2

    normalized = normalized_heading(heading)
    major_exact = {
        "ABSTRACT",
        "INTRODUCTION",
        "PROBLEM DESCRIPTION",
        "MODEL DESCRIPTION",
        "FRAMEWORK FORMULATION",
        "RESULTS",
        "SIMULATION RESULTS",
        "EXPERIMENTAL RESULTS",
        "CONCLUSION",
        "CONCLUSIONS",
        "APPENDIX",
        "ACKNOWLEDGMENT",
        "ACKNOWLEDGMENTS",
        "REFERENCES",
        "BIOGRAPHY",
        "BIOGRAPHIES",
        "TABLE OF CONTENTS",
    }
    if normalized in major_exact:
        return 1
    if any(
        normalized.startswith(prefix)
        for prefix in (
            "LEO NNPON FRAMEWORK",
            "OPPORTUNISTIC NAVIGATION WITH",
        )
    ):
        return 1
    return 2


def render_table_text(value: str) -> str:
    text = re.sub(r"</(?:td|th|tr|p)>", " ", value, flags=re.IGNORECASE)
    return clean_text(text)


def render_block_content(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    text = clean_text(block.get("text"))
    caption = clean_text(block.get("caption"))
    if block_type == "equation":
        latex = clean_text(block.get("latex") or block.get("latex_raw"))
        return "\n".join(part for part in ("[Equation]", latex, caption) if part)
    if block_type == "figure":
        return "\n".join(part for part in ("[Figure]", caption or text) if part)
    if block_type == "table":
        table_text = render_table_text(
            str(block.get("table_html") or block.get("table_html_raw") or "")
        )
        return "\n".join(
            part for part in ("[Table]", caption or text, table_text) if part
        )
    if block_type == "algorithm":
        return "\n".join(part for part in ("[Algorithm]", caption, text) if part)
    return text


def structure_input_fingerprint(document: dict[str, Any]) -> str:
    blocks = document.get("blocks")
    relevant_blocks: list[dict[str, Any]] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            relevant_blocks.append(
                {
                    key: block.get(key)
                    for key in (
                        "block_id",
                        "page_number",
                        "reading_order",
                        "type",
                        "text",
                        "caption",
                        "title_level_raw",
                        "latex",
                        "latex_raw",
                        "table_html",
                        "table_html_raw",
                        "image_path",
                        "quality",
                    )
                }
            )
    payload = {
        "policy": STRUCTURE_POLICY_VERSION,
        "identity": document.get("identity"),
        "metadata": document.get("metadata"),
        "source_sha256": (
            document.get("source", {}).get("sha256")
            if isinstance(document.get("source"), dict)
            else None
        ),
        "blocks": relevant_blocks,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", value.casefold())


def _matches_metadata_abstract(block_text: str, metadata_abstract: str) -> bool:
    candidate = _compact_match_text(block_text)
    abstract = _compact_match_text(metadata_abstract)
    if len(candidate) < 80:
        return False
    if candidate in abstract:
        return True
    candidate_tokens = re.findall(r"[a-z0-9]+", block_text.casefold())
    abstract_tokens = Counter(re.findall(r"[a-z0-9]+", metadata_abstract.casefold()))
    if len(candidate_tokens) < 15:
        return False
    candidate_counts = Counter(candidate_tokens)
    overlap = sum(
        min(count, abstract_tokens[token]) for token, count in candidate_counts.items()
    )
    return overlap / len(candidate_tokens) >= 0.82


def _is_explicit_abstract_start(block: dict[str, Any]) -> bool:
    if block.get("type") != "paragraph":
        return False
    return bool(
        re.match(
            r"^ABSTRACT\s*[—:-]",
            clean_text(block.get("text")),
            flags=re.IGNORECASE,
        )
    )


def _block_specific_zone(
    block: dict[str, Any],
    current_zone: str,
    metadata_abstract: str,
) -> str:
    if current_zone != "front_matter" or block.get("type") != "paragraph":
        return current_zone
    text = clean_text(block.get("text"))
    if _is_explicit_abstract_start(block) or _matches_metadata_abstract(
        text,
        metadata_abstract,
    ):
        return "abstract"
    return current_zone


def build_structure(document: dict[str, Any]) -> dict[str, Any]:
    identity = document.get("identity")
    metadata = document.get("metadata")
    if not isinstance(identity, dict) or not isinstance(metadata, dict):
        raise ValueError("canonical 文档缺少 identity 或 metadata。")
    paper_id = document.get("paper_id")
    document_id = identity.get("document_id")
    work_id = identity.get("work_id")
    title = metadata.get("title")
    metadata_abstract = metadata.get("abstract")
    if not isinstance(metadata_abstract, str):
        metadata_abstract = ""
    if not all(
        isinstance(value, str) and value for value in (paper_id, document_id, title)
    ):
        raise ValueError("canonical 文档缺少 paper_id、document_id 或 title。")

    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("canonical blocks 必须是数组。")
    ordered_blocks = sorted(
        (block for block in raw_blocks if isinstance(block, dict)),
        key=lambda block: (
            int(block.get("page_number", 0)),
            int(block.get("reading_order", 0)),
        ),
    )

    current_zone = "front_matter"
    section_stack: list[_Section] = []
    sections: list[_Section] = []
    enriched: list[dict[str, Any]] = []
    document_title_seen = False

    for raw in ordered_blocks:
        block_type = raw.get("type")
        block_id = raw.get("block_id")
        page = int(raw.get("page_number", 0))
        text = clean_text(raw.get("text"))
        is_heading = block_type == "title"
        false_heading = is_heading and is_false_heading(text)
        heading_level: int | None = None
        if is_heading and not document_title_seen and raw.get("title_level_raw") == 1:
            document_title_seen = True
        elif is_heading and not false_heading and text:
            heading_level = infer_heading_level(text)
            current_zone = content_zone_for_heading(text, current_zone)
            while section_stack and int(section_stack[-1]["level"]) >= heading_level:
                section_stack.pop()
            section_id = f"{document_id}_s{len(sections) + 1:04d}"
            section: _Section = {
                "section_id": section_id,
                "title": text,
                "level": heading_level,
                "section_path": [
                    *[str(item["title"]) for item in section_stack],
                    text,
                ],
                "content_zone": current_zone,
                "page_start": page,
                "page_end": page,
                "heading_block_id": block_id if isinstance(block_id, str) else None,
                "block_ids": [],
            }
            sections.append(section)
            section_stack.append(section)

        block_zone = _block_specific_zone(raw, current_zone, metadata_abstract)
        if block_zone != current_zone and _is_explicit_abstract_start(raw):
            current_zone = block_zone
        current_section = section_stack[-1] if section_stack else None
        section_path = list(current_section["section_path"]) if current_section else []
        content = render_block_content(raw)
        quality = raw.get("quality")
        retrieval_enabled = (
            quality.get("retrieval_enabled") if isinstance(quality, dict) else True
        )
        exclusion_reason: str | None = None
        if block_type == "page_metadata":
            exclusion_reason = "page_metadata"
        elif is_heading:
            exclusion_reason = "heading"
        elif not content:
            exclusion_reason = "empty"
        elif retrieval_enabled is False:
            exclusion_reason = "quality_excluded"
        elif block_zone not in SEARCHABLE_ZONES:
            exclusion_reason = f"zone:{block_zone}"
        searchable = exclusion_reason is None
        item = {
            "block_id": block_id,
            "paper_id": paper_id,
            "document_id": document_id,
            "work_id": work_id,
            "page_number": page,
            "reading_order": int(raw.get("reading_order", 0)),
            "type": block_type,
            "content": content,
            "content_zone": block_zone,
            "section_id": (
                current_section.get("section_id") if current_section else None
            ),
            "section_path": section_path,
            "heading_level": heading_level,
            "searchable": searchable,
            "exclusion_reason": exclusion_reason,
            "image_path": raw.get("image_path"),
            "related_text_block_ids": [],
        }
        enriched.append(item)
        if current_section is not None and not is_heading:
            if isinstance(block_id, str):
                current_section["block_ids"].append(block_id)
            current_section["page_end"] = max(int(current_section["page_end"]), page)

    for index, block in enumerate(enriched):
        if block["type"] not in ASSET_TYPES or not block["searchable"]:
            continue
        related: list[str] = []
        for direction in (-1, 1):
            cursor = index + direction
            while 0 <= cursor < len(enriched):
                candidate = enriched[cursor]
                if candidate["section_id"] != block["section_id"]:
                    break
                if candidate["searchable"] and candidate["type"] in TEXT_TYPES:
                    candidate_id = candidate["block_id"]
                    if isinstance(candidate_id, str):
                        related.append(candidate_id)
                    break
                cursor += direction
        block["related_text_block_ids"] = related

    return {
        "structure_schema_version": STRUCTURE_SCHEMA_VERSION,
        "structure_policy_version": STRUCTURE_POLICY_VERSION,
        "input_fingerprint": structure_input_fingerprint(document),
        "paper_id": paper_id,
        "document_id": document_id,
        "work_id": work_id,
        "title": title,
        "authors": metadata.get("authors") or [],
        "year": metadata.get("year"),
        "doi": metadata.get("doi"),
        "canonical_path": None,
        "section_count": len(sections),
        "searchable_block_count": sum(bool(block["searchable"]) for block in enriched),
        "excluded_block_count": sum(
            not bool(block["searchable"]) for block in enriched
        ),
        "sections": sections,
        "blocks": enriched,
    }


def structure_output_path(project_root: Path, document_id: str) -> Path:
    return (
        project_root.expanduser().resolve()
        / "data"
        / "knowledge"
        / "structures"
        / f"{document_id}.structure.json"
    )


def write_structure(project_root: Path, structure: dict[str, Any]) -> Path:
    document_id = structure.get("document_id")
    if not isinstance(document_id, str):
        raise ValueError("structure.document_id 不能为空。")
    output = structure_output_path(project_root, document_id)
    write_json_atomic(output, structure)
    return output
