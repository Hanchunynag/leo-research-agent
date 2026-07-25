from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class IngestResult:
    """论文入库结果。

    paper_id:
        系统内部使用的稳定唯一标识，由 PDF 内容哈希生成。

    sha256:
        PDF 文件完整的 SHA-256 哈希值，用于内容去重。

    original_filename:
        用户上传时的原始文件名。

    stored_filename:
        清理特殊字符后，实际存储在本地的文件名。

    source_path:
        论文在本地知识库中的实际路径。
    """

    paper_id: str
    sha256: str
    original_filename: str
    stored_filename: str
    source_path: Path

def sanitize_pdf_filename(filename: str) -> str:
    """生成适合本地存储的 PDF 文件名。

    处理规则：
    1. 保留中文、英文、数字和下划线；
    2. 空格和特殊字符统一替换为下划线；
    3. 连续下划线合并；
    4. 限制文件名长度；
    5. 强制使用 .pdf 后缀。

    Args:
        filename:
            用户上传时的原始文件名。

    Returns:
        清理后的安全 PDF 文件名。
    """

    original_path = Path(filename)
    stem = original_path.stem.strip()

    # Python 的 \w 在 Unicode 模式下可以保留中文、英文和数字。
    safe_stem = re.sub(r"[^\w\-().]+", "_", stem, flags=re.UNICODE)

    # 合并连续下划线。
    safe_stem = re.sub(r"_+", "_", safe_stem)

    # 去除首尾无意义字符。
    safe_stem = safe_stem.strip("._-")

    if not safe_stem:
        safe_stem = "untitled_paper"

    # 防止文件名过长。
    safe_stem = safe_stem[:150]

    return f"{safe_stem}.pdf"

def calculate_sha256(file_path: Path) -> str:
    """
    计算文件的SHA256哈希值，什么是 sha256？
    SHA256是一种加密哈希函数，它将任意长度的数据映射为固定长度的256位（32字节）哈希值。
    它常用于数据完整性验证和数字签名等场景。
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def ingest_paper(input_path: Path, raw_dir: Path) -> IngestResult:
    """摄入论文 PDF，完成校验、去重和规范化存储。"""

    input_path = input_path.expanduser().resolve()
    raw_dir = raw_dir.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not input_path.is_file():
        raise ValueError(f"输入路径不是文件：{input_path}")

    if input_path.suffix.lower() != ".pdf":
        raise ValueError("当前只支持 PDF 文件。")

    with input_path.open("rb") as file:
        signature = file.read(5)

    if signature != b"%PDF-":
        raise ValueError("文件扩展名为 PDF，但文件内容不是有效 PDF。")

    sha256 = calculate_sha256(input_path)
    paper_id = f"P_{sha256[:12]}"

    paper_dir = raw_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    existing_pdfs = sorted(paper_dir.glob("*.pdf"))

    if existing_pdfs:
        source_path = existing_pdfs[0]
    else:
        stored_filename = sanitize_pdf_filename(input_path.name)
        source_path = paper_dir / stored_filename
        shutil.copy2(input_path, source_path)

    return IngestResult(
        paper_id=paper_id,
        sha256=sha256,
        original_filename=input_path.name,
        stored_filename=source_path.name,
        source_path=source_path,
    )
