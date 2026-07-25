"""PDF 预检查模块。

该模块不负责完整解析论文，而是在运行 MinerU 或 Docling 前，
快速判断 PDF 的基本结构和解析难度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf


PDFType = Literal[
    "native_text",
    "scanned_image",
    "scanned_with_ocr",
    "mixed",
]


@dataclass(frozen=True)
class PDFPrecheckResult:
    """PDF预检查结果。

    pdf_type:
        native_text:
            主要由原生文字对象构成。

        scanned_image:
            大部分页面是整页图片，并且几乎没有文字层。

        scanned_with_ocr:
            大部分页面是整页图片，但同时存在OCR文字层。

        mixed:
            部分页面有文字层，部分页面主要是图片。

    likely_two_column_pages:
        通过文本块位置粗略判断出的双栏页面编号。
        这里只是候选结果，不代表最终版面解析结论。
    """

    source_path: str
    page_count: int
    file_size_bytes: int

    pages_with_text: list[int]
    pages_without_text: list[int]

    full_page_image_pages: list[int]
    likely_two_column_pages: list[int]

    total_text_characters: int
    total_embedded_images: int

    text_page_ratio: float
    full_page_image_ratio: float

    pdf_type: PDFType

def page_contains_meaningful_text(
    page: pymupdf.Page,
    minimum_characters: int = 50,
) -> bool:
    """判断页面是否包含有意义的文字层。"""

    text = page.get_text("text").strip()

    return len(text) >= minimum_characters

def page_contains_full_page_image(
    page: pymupdf.Page,
    minimum_coverage: float = 0.75,
) -> bool:
    """判断页面是否包含覆盖大部分页面的图片。

    扫描PDF通常会把整页内容保存成一张大图片。
    如果某张图片覆盖页面面积的75%以上，则视为整页扫描图片候选。
    """

    page_area = page.rect.get_area()

    if page_area <= 0:
        return False

    for image_info in page.get_images(full=True):
        xref = image_info[0]

        try:
            rectangles = page.get_image_rects(xref)
        except RuntimeError:
            continue

        for rectangle in rectangles:
            coverage = rectangle.get_area() / page_area

            if coverage >= minimum_coverage:
                return True

    return False

def page_is_likely_two_column(page: pymupdf.Page) -> bool:
    """粗略判断页面是否包含左右双栏正文。

    判断依据：
    1. 忽略页眉、页脚；
    2. 忽略过宽的跨栏元素；
    3. 分别累计页面左侧和右侧的正文字符数；
    4. 左右两侧都具有足够正文时，认为是双栏页面。

    这种方式比单纯统计文本块数量更稳定，因为不同 PDF
    可能把一整栏合并为一个文本块。
    """

    page_width = page.rect.width
    page_height = page.rect.height

    if page_width <= 0 or page_height <= 0:
        return False

    page_middle = page_width / 2
    center_margin = page_width * 0.02

    body_top = page_height * 0.06
    body_bottom = page_height * 0.95

    left_characters = 0
    right_characters = 0

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]

        clean_text = str(text).strip()

        if len(clean_text) < 10:
            continue

        # 忽略页眉和页脚。
        if y1 < body_top or y0 > body_bottom:
            continue

        block_width = x1 - x0

        # 忽略标题、跨栏公式、跨栏图注等宽元素。
        if block_width >= page_width * 0.78:
            continue

        block_center = (x0 + x1) / 2

        if block_center < page_middle - center_margin:
            left_characters += len(clean_text)

        elif block_center > page_middle + center_margin:
            right_characters += len(clean_text)

    minimum_characters_per_column = 150

    return (
        left_characters >= minimum_characters_per_column
        and right_characters >= minimum_characters_per_column
    )

def classify_pdf_type(
    text_page_ratio: float,
    full_page_image_ratio: float,
) -> PDFType:
    """根据文字层和整页图片比例判断 PDF 类型。"""

    if full_page_image_ratio >= 0.8 and text_page_ratio <= 0.2:
        return "scanned_image"

    if full_page_image_ratio >= 0.8 and text_page_ratio >= 0.8:
        return "scanned_with_ocr"

    if text_page_ratio >= 0.8 and full_page_image_ratio < 0.5:
        return "native_text"

    return "mixed"

def precheck_pdf(pdf_path: Path) -> PDFPrecheckResult:
    """执行PDF预检查。

    Args:
        pdf_path:
            已入库PDF的本地路径。

    Returns:
        PDF的页数、文字层、图片和双栏候选信息。
    """

    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages_with_text: list[int] = []
    pages_without_text: list[int] = []

    full_page_image_pages: list[int] = []
    likely_two_column_pages: list[int] = []

    total_text_characters = 0
    total_embedded_images = 0

    with pymupdf.open(pdf_path) as document:
        page_count = document.page_count

        if page_count == 0:
            raise ValueError("PDF不包含任何页面。")

        for page_index, page in enumerate(document):
            page_number = page_index + 1

            page_text = page.get_text("text").strip()
            total_text_characters += len(page_text)

            if page_contains_meaningful_text(page):
                pages_with_text.append(page_number)
            else:
                pages_without_text.append(page_number)

            page_images = page.get_images(full=True)
            total_embedded_images += len(page_images)

            if page_contains_full_page_image(page):
                full_page_image_pages.append(page_number)

            if page_is_likely_two_column(page):
                likely_two_column_pages.append(page_number)

    text_page_ratio = len(pages_with_text) / page_count
    full_page_image_ratio = len(full_page_image_pages) / page_count

    pdf_type = classify_pdf_type(
        text_page_ratio=text_page_ratio,
        full_page_image_ratio=full_page_image_ratio,
    )

    return PDFPrecheckResult(
        source_path=str(pdf_path),
        page_count=page_count,
        file_size_bytes=pdf_path.stat().st_size,
        pages_with_text=pages_with_text,
        pages_without_text=pages_without_text,
        full_page_image_pages=full_page_image_pages,
        likely_two_column_pages=likely_two_column_pages,
        total_text_characters=total_text_characters,
        total_embedded_images=total_embedded_images,
        text_page_ratio=round(text_page_ratio, 4),
        full_page_image_ratio=round(full_page_image_ratio, 4),
        pdf_type=pdf_type,
    )