"""MinerU 输出到 Canonical JSON 的适配器。

本模块只负责无损标准化，不负责：

- 文本纠错；
- 公式纠错；
- 表格重建；
- 标题层级修正；
- Chunk 切分。

原始 MinerU 数据会完整保存在 raw_block 中，
确保后续可以重新处理。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_TYPE_MAPPING = {
    "title": "title",
    "paragraph": "paragraph",
    "list": "list",
    "equation_interline": "equation",
    "image": "figure",
    "chart": "figure",
    "table": "table",
    "algorithm": "algorithm",
    "page_number": "page_metadata",
    "page_header": "page_metadata",
    "page_footer": "page_metadata",
    "page_aside_text": "page_metadata",
    "page_footnote": "page_metadata",
}

NATIVE_TEXT_FALLBACK_POLICY_VERSION = "1.0"
NATIVE_TEXT_FALLBACK_MIN_CHARS = 1200
NATIVE_TEXT_FALLBACK_PAGE_MIN_CHARS = 120
NATIVE_TEXT_FALLBACK_COVERAGE_RATIO = 1.5
OCR_RETRY_POLICY_VERSION = "1.0"


def _visible_text_length(value: Any) -> int:
    """Count visible characters without MinerU's inline markup."""

    if not isinstance(value, str):
        return 0
    without_tags = re.sub(r"<[^>]+>", "", value)
    return len(re.sub(r"\s+", "", without_tags))


def _cjk_character_count(value: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def _retrieval_text(document: dict[str, Any]) -> str:
    blocks = document.get("blocks")
    if not isinstance(blocks, list):
        return ""
    values: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        quality = block.get("quality")
        if isinstance(quality, dict) and quality.get("retrieval_enabled") is False:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            values.append(text)
    return "\n".join(values)


def filename_title_fallback(original_filename: str) -> str | None:
    """Extract a conservative human title from a user-provided PDF name.

    Chinese paper files commonly use ``标题_作者.pdf``.  Keep the title part
    when it is unambiguously CJK so a bad OCR title cannot poison Paper-level
    retrieval.  English filenames are left intact because underscores may be
    meaningful in their citation/export names.
    """

    stem = Path(original_filename).stem.strip()
    if not stem:
        return None
    parts = [part.strip() for part in re.split(r"_+", stem) if part.strip()]
    if len(parts) > 1 and _cjk_character_count(parts[0]) >= 2:
        stem = parts[0]
    stem = re.sub(r"\s+", " ", stem.replace("_", " ")).strip(" ._-（()[]【】")
    return stem or None


def should_retry_with_ocr(
    document: dict[str, Any],
    original_filename: str,
    *,
    pdf_type: str | None = None,
) -> dict[str, Any]:
    """Detect a misleading OCR/text-layer result that needs a second pass.

    A scanned Chinese PDF can contain an OCR text layer that makes PyMuPDF
    report it as ``native_text`` while MinerU has actually extracted mostly
    Latin fragments and ``<sub>/<sup>`` artifacts.  Those documents look
    successful to a page-count check but are not searchable.  Return a
    structured diagnostic so the parse pipeline can run MinerU's explicit OCR
    method and retain the decision in ``paper.json``.
    """

    text = _retrieval_text(document)
    title = str((document.get("metadata") or {}).get("title") or "")
    filename_cjk = _cjk_character_count(original_filename)
    text_cjk = _cjk_character_count(text)
    visible_chars = _visible_text_length(text)
    markup_count = len(re.findall(r"</?(?:sub|sup|span)\b", text, re.IGNORECASE))
    explicit_scanned = pdf_type in {"scanned_image", "scanned_with_ocr"}
    cjk_filename_with_weak_text = (
        filename_cjk >= 2
        and visible_chars >= 200
        and text_cjk < max(12, int(visible_chars * 0.03))
    )
    malformed_title = bool(re.search(r"<[^>]+>", title))
    retry = explicit_scanned or cjk_filename_with_weak_text or (
        filename_cjk >= 2 and malformed_title and text_cjk < max(20, int(visible_chars * 0.08))
    )
    if explicit_scanned:
        reason = "pdf_precheck_requires_ocr"
    elif cjk_filename_with_weak_text:
        reason = "cjk_filename_but_weak_cjk_text_layer"
    elif retry:
        reason = "malformed_title_and_weak_cjk_text_layer"
    else:
        reason = "text_layer_quality_sufficient"
    return {
        "policy_version": OCR_RETRY_POLICY_VERSION,
        "retry": retry,
        "reason": reason,
        "pdf_type": pdf_type,
        "filename_cjk_count": filename_cjk,
        "text_cjk_count": text_cjk,
        "visible_char_count": visible_chars,
        "markup_count": markup_count,
        "parser_title": title,
    }


def load_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def render_inline_items(
    items: Any,
) -> str:
    """把 MinerU 的文本和行内公式序列渲染为字符串。

    普通文本直接保留；
    行内公式使用 $...$ 包裹。
    """

    if not isinstance(items, list):
        return ""

    fragments: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        content = item.get("content")

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        item_type = item.get("type")

        if item_type == "equation_inline":
            fragments.append(f"${content}$")
        else:
            fragments.append(content)

    rendered = " ".join(fragments).strip()
    rendered = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", rendered)
    rendered = re.sub(r"([\(\[])\s+", r"\1", rendered)
    return rendered


def render_list_items(
    content: dict[str, Any],
) -> str:
    """渲染列表块。"""

    list_items = content.get("list_items")

    if not isinstance(list_items, list):
        return ""

    rendered_items: list[str] = []

    for list_item in list_items:
        if not isinstance(list_item, dict):
            continue

        item_text = render_inline_items(
            list_item.get("item_content")
        )

        if item_text:
            rendered_items.append(item_text)

    return "\n".join(rendered_items)


def extract_caption(
    content: dict[str, Any],
    source_type: str,
) -> str | None:
    """从图、表或图表块提取标题。"""

    caption_field_mapping = {
        "image": "image_caption",
        "chart": "chart_caption",
        "table": "table_caption",
    }

    field_name = caption_field_mapping.get(source_type)

    if field_name is None:
        return None

    caption = render_inline_items(
        content.get(field_name)
    )

    return caption or None


def extract_image_path(
    content: dict[str, Any],
    mineru_output_dir: Path,
) -> str | None:
    """提取图片路径，并转换成项目中的相对路径。"""

    image_source = content.get("image_source")

    if not isinstance(image_source, dict):
        return None

    relative_path = image_source.get("path")

    if not isinstance(relative_path, str):
        return None

    relative_path = relative_path.strip()

    if not relative_path:
        return None

    return (
        mineru_output_dir
        / relative_path
    ).as_posix()


def extract_text(
    source_type: str,
    content: dict[str, Any],
) -> str:
    """根据块类型提取适合检索的文本表示。"""

    if source_type == "title":
        return render_inline_items(
            content.get("title_content")
        )

    if source_type == "paragraph":
        return render_inline_items(
            content.get("paragraph_content")
        )

    if source_type == "list":
        return render_list_items(content)

    if source_type == "equation_interline":
        latex = content.get("math_content")

        if isinstance(latex, str) and latex.strip():
            return f"$${latex.strip()}$$"

        return ""

    if source_type == "image":
        return render_inline_items(
            content.get("image_caption")
        )

    if source_type == "chart":
        return render_inline_items(
            content.get("chart_caption")
        )

    if source_type == "table":
        caption = render_inline_items(
            content.get("table_caption")
        )

        html = content.get("html")

        if isinstance(html, str) and html.strip():
            if caption:
                return f"{caption}\n\n{html.strip()}"

            return html.strip()

        return caption

    if source_type == "algorithm":
        return render_inline_items(
            content.get("algorithm_content")
        )

    if source_type == "page_number":
        return render_inline_items(
            content.get("page_number_content")
        )

    if source_type == "page_header":
        return render_inline_items(
            content.get("page_header_content")
        )

    if source_type == "page_footer":
        return render_inline_items(
            content.get("page_footer_content")
        )

    if source_type == "page_aside_text":
        return render_inline_items(
            content.get("page_aside_text_content")
        )

    if source_type == "page_footnote":
        return render_inline_items(
            content.get("page_footnote_content")
        )

    raw_content = content.get("content")

    if isinstance(raw_content, str):
        return raw_content.strip()

    return ""


def build_quality_state(
    source_type: str,
    text: str,
    latex: str | None,
    table_html: str | None,
    image_path: str | None,
) -> dict[str, Any]:
    """生成第一版质量状态。

    这里只根据结构判断，不判断内容语义是否正确。
    """

    issues: list[str] = []

    retrieval_enabled = True
    status = "usable"

    if source_type in {
        "page_number",
        "page_header",
        "page_footer",
        "page_aside_text",
        "page_footnote",
    }:
        retrieval_enabled = False
        status = "excluded"

    elif source_type == "equation_interline":
        if not latex:
            issues.append("missing_latex")

        if not image_path:
            issues.append("missing_equation_image")

        if issues:
            status = "degraded"

    elif source_type == "table":
        if not table_html:
            issues.append("table_html_empty")
            status = "image_only"

        if not image_path:
            issues.append("missing_table_image")

    elif not text and not image_path:
        retrieval_enabled = False
        status = "empty"
        issues.append("empty_block")

    return {
        "status": status,
        "issues": issues,
        "retrieval_enabled": retrieval_enabled,
    }


def convert_block(
    paper_id: str,
    page_number: int,
    block_order: int,
    raw_block: dict[str, Any],
    mineru_output_dir: Path,
) -> dict[str, Any]:
    """把单个 MinerU 块转换成 Canonical 块。"""

    source_type = str(
        raw_block.get("type", "unknown")
    )

    canonical_type = SOURCE_TYPE_MAPPING.get(
        source_type,
        "unknown",
    )

    raw_content = raw_block.get("content")

    if isinstance(raw_content, dict):
        content = raw_content
    else:
        content = {}

    text = extract_text(
        source_type=source_type,
        content=content,
    )

    caption = extract_caption(
        content=content,
        source_type=source_type,
    )

    image_path = extract_image_path(
        content=content,
        mineru_output_dir=mineru_output_dir,
    )

    latex_value = content.get("math_content")

    if isinstance(latex_value, str):
        latex = latex_value.strip() or None
    else:
        latex = None

    html_value = content.get("html")

    if isinstance(html_value, str):
        table_html = html_value.strip() or None
    else:
        table_html = None

    title_level_value = content.get("level")

    if isinstance(title_level_value, int):
        title_level = title_level_value
    else:
        title_level = None

    block_id = (
        f"{paper_id}"
        f"_p{page_number:03d}"
        f"_b{block_order:03d}"
    )

    quality = build_quality_state(
        source_type=source_type,
        text=text,
        latex=latex,
        table_html=table_html,
        image_path=image_path,
    )

    return {
        "block_id": block_id,
        "paper_id": paper_id,
        "page_number": page_number,
        "reading_order": block_order,

        "type": canonical_type,
        "source_type": source_type,

        "bbox": raw_block.get("bbox"),
        "bbox_source": "mineru_content_list_v2",

        "text": text,
        "caption": caption,

        "title_level_raw": title_level,

        "latex": latex,
        "latex_raw": latex,

        "table_html": table_html,
        "table_html_raw": table_html,

        "image_path": image_path,

        "quality": quality,

        "raw_block": raw_block,
    }


def build_page_metadata(
    middle_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """从 middle.json 提取页级元数据。"""

    pdf_info = middle_data.get("pdf_info")

    if not isinstance(pdf_info, list):
        return []

    pages: list[dict[str, Any]] = []

    for position, page_info in enumerate(pdf_info):
        if not isinstance(page_info, dict):
            continue

        page_index = page_info.get(
            "page_idx",
            position,
        )

        pages.append(
            {
                "page_number": position + 1,
                "mineru_page_index": page_index,
                "page_size": page_info.get("page_size"),
                "para_block_count": len(
                    page_info.get("para_blocks", [])
                ),
                "discarded_block_count": len(
                    page_info.get("discarded_blocks", [])
                ),
            }
        )

    return pages


def build_canonical_document(
    paper_id: str,
    content_list_v2_path: Path,
    middle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构建 Canonical 文档和转换报告。"""

    content_pages = load_json(
        content_list_v2_path
    )

    middle_data = load_json(
        middle_path
    )

    if not isinstance(content_pages, list):
        raise TypeError(
            "content_list_v2.json 根对象必须是列表。"
        )

    if not isinstance(middle_data, dict):
        raise TypeError(
            "middle.json 根对象必须是字典。"
        )

    mineru_output_dir = content_list_v2_path.parent

    blocks: list[dict[str, Any]] = []
    type_counter: Counter[str] = Counter()

    for page_index, raw_page_blocks in enumerate(
        content_pages
    ):
        page_number = page_index + 1

        if not isinstance(raw_page_blocks, list):
            continue

        for block_order, raw_block in enumerate(
            raw_page_blocks
        ):
            if not isinstance(raw_block, dict):
                continue

            canonical_block = convert_block(
                paper_id=paper_id,
                page_number=page_number,
                block_order=block_order,
                raw_block=raw_block,
                mineru_output_dir=mineru_output_dir,
            )

            blocks.append(canonical_block)

            type_counter[
                canonical_block["type"]
            ] += 1

    title = next(
        (
            block["text"]
            for block in blocks
            if block["type"] == "title"
            and block["title_level_raw"] == 1
            and block["text"]
        ),
        None,
    )

    pages = build_page_metadata(middle_data)

    canonical_document = {
        "schema_version": "1.0",
        "paper_id": paper_id,

        "metadata": {
            "title": title,
            "authors": [],
            "abstract": None,
        },

        "parser": {
            "name": "mineru",
            "backend": middle_data.get("_backend"),
            "version": middle_data.get(
                "_version_name"
            ),
            "mode": "txt",
            "formula_enabled": True,
            "table_enabled": False,
        },

        "source_files": {
            "content_list_v2": (
                content_list_v2_path.as_posix()
            ),
            "middle_json": middle_path.as_posix(),
        },

        "page_count": len(content_pages),
        "pages": pages,
        "blocks": blocks,
    }

    issue_counter: Counter[str] = Counter()

    for block in blocks:
        for issue in block["quality"]["issues"]:
            issue_counter[issue] += 1

    missing_assets = [
        block["block_id"]
        for block in blocks
        if block["image_path"]
        and not Path(block["image_path"]).exists()
    ]

    report = {
        "paper_id": paper_id,
        "page_count": len(content_pages),
        "block_count": len(blocks),
        "block_type_counts": dict(type_counter),
        "quality_issue_counts": dict(issue_counter),
        "missing_asset_count": len(missing_assets),
        "missing_asset_block_ids": missing_assets,
    }

    return canonical_document, report


def augment_native_pdf_text(
    document: dict[str, Any],
    pdf_path: Path,
) -> dict[str, Any]:
    """补回 MinerU 漏掉的原生 PDF 文本。

    少数期刊版式会被 MinerU 的版面模型识别成大块图片：公式和图仍能
    输出，但正文文本覆盖率很低。此时 PDF 本身通常仍保留可复制文本，
    PyMuPDF 可以提供一个可检索的保底投影。只有在全篇和单页覆盖率都明显
    不足时才追加，避免正常 MinerU 结果出现重复内容。
    """

    report = {
        "policy_version": NATIVE_TEXT_FALLBACK_POLICY_VERSION,
        "status": "not_needed",
        "native_char_count": 0,
        "canonical_char_count": 0,
        "fallback_page_count": 0,
        "fallback_block_ids": [],
    }
    try:
        import pymupdf

        pdf = pymupdf.open(pdf_path)
    except (OSError, RuntimeError, ValueError) as error:
        report["status"] = "unavailable"
        report["error"] = f"{type(error).__name__}: {error}"
        return report

    try:
        blocks = document.get("blocks")
        if not isinstance(blocks, list):
            report["status"] = "invalid_document"
            return report

        def visible_length(value: Any) -> int:
            if not isinstance(value, str):
                return 0
            without_tags = re.sub(r"<[^>]+>", "", value)
            return len(re.sub(r"\s+", "", without_tags))

        canonical_by_page: dict[int, int] = {}
        maximum_order_by_page: dict[int, int] = {}
        for block in blocks:
            if not isinstance(block, dict):
                continue
            page = block.get("page_number")
            if not isinstance(page, int) or page < 1:
                continue
            maximum_order_by_page[page] = max(
                maximum_order_by_page.get(page, -1),
                int(block.get("reading_order", 0)),
            )
            if block.get("type") in {"title", "paragraph", "list", "algorithm"}:
                canonical_by_page[page] = canonical_by_page.get(page, 0) + visible_length(
                    block.get("text")
                )

        native_pages: list[tuple[int, str, float, float]] = []
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            native_pages.append(
                (page_number, text, float(page.rect.width), float(page.rect.height))
            )

        native_char_count = sum(visible_length(text) for _, text, _, _ in native_pages)
        canonical_char_count = sum(canonical_by_page.values())
        report["native_char_count"] = native_char_count
        report["canonical_char_count"] = canonical_char_count
        if native_char_count < NATIVE_TEXT_FALLBACK_MIN_CHARS or native_char_count <= max(
            canonical_char_count * NATIVE_TEXT_FALLBACK_COVERAGE_RATIO,
            NATIVE_TEXT_FALLBACK_MIN_CHARS,
        ):
            return report

        next_order = dict(maximum_order_by_page)
        for page_number, text, width, height in native_pages:
            native_length = visible_length(text)
            existing_length = canonical_by_page.get(page_number, 0)
            if native_length < NATIVE_TEXT_FALLBACK_PAGE_MIN_CHARS:
                continue
            if native_length <= max(
                existing_length * NATIVE_TEXT_FALLBACK_COVERAGE_RATIO,
                NATIVE_TEXT_FALLBACK_PAGE_MIN_CHARS,
            ):
                continue
            order = next_order.get(page_number, -1) + 1
            next_order[page_number] = order
            block_id = f"{document.get('paper_id')}_p{page_number:03d}_b{order:03d}"
            blocks.append(
                {
                    "block_id": block_id,
                    "paper_id": document.get("paper_id"),
                    "page_number": page_number,
                    "reading_order": order,
                    "type": "paragraph",
                    "source_type": "native_pdf_text",
                    "bbox": [0, 0, width, height],
                    "bbox_source": "pymupdf_native_text",
                    "text": text,
                    "caption": None,
                    "title_level_raw": None,
                    "latex": None,
                    "latex_raw": None,
                    "table_html": None,
                    "table_html_raw": None,
                    "image_path": None,
                    "quality": {
                        "status": "fallback",
                        "issues": ["mineru_text_coverage_low"],
                        "retrieval_enabled": True,
                        "validation_passed": True,
                    },
                    "raw_block": {
                        "type": "native_pdf_text",
                        "text": text,
                        "page_number": page_number,
                    },
                }
            )
            report["fallback_block_ids"].append(block_id)

        report["fallback_page_count"] = len(report["fallback_block_ids"])
        report["status"] = (
            "applied" if report["fallback_block_ids"] else "not_needed"
        )
        return report
    finally:
        pdf.close()
