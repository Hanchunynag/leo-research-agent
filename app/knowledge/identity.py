"""论文 work/document 身份与真实标题文件名规范。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ingestion.ingest import calculate_sha256


WORK_ID_PATTERN = re.compile(r"^W_[0-9a-f]{12}$")
DOCUMENT_ID_PATTERN = re.compile(r"^D_[0-9a-f]{12}$")
RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
FILENAME_POLICY = "verified_title_v1"
MAX_FILENAME_BYTES = 220


@dataclass(frozen=True)
class RawPDFRename:
    old_path: Path
    new_path: Path
    changed: bool


def normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.strip() or None


def normalize_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def document_id_from_sha256(sha256: str) -> str:
    return f"D_{sha256[:12]}"


def work_identity_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    verification = metadata.get("verification")
    status = verification.get("status") if isinstance(verification, dict) else None
    if status not in {"verified", "selected"}:
        return None

    doi = normalize_doi(metadata.get("doi"))
    if doi:
        return f"doi:{doi}", "doi"

    external_ids = metadata.get("external_ids")
    arxiv_id = external_ids.get("arxiv") if isinstance(external_ids, dict) else None
    if isinstance(arxiv_id, str) and arxiv_id.strip():
        versionless = re.sub(r"v\d+$", "", arxiv_id.strip().casefold())
        return f"arxiv:{versionless}", "arxiv"

    title = metadata.get("title")
    authors = metadata.get("authors")
    year = metadata.get("year")
    if (
        isinstance(title, str)
        and title.strip()
        and isinstance(authors, list)
        and authors
        and isinstance(authors[0], str)
        and isinstance(year, int)
        and not isinstance(year, bool)
    ):
        key = ":".join(
            (
                "bibliographic",
                normalize_identity_text(title),
                normalize_identity_text(authors[0]),
                str(year),
            )
        )
        return key, "bibliographic"
    return None


def build_identity(
    paper_id: str,
    sha256: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    document_id = document_id_from_sha256(sha256)
    work_key = work_identity_key(metadata)
    if work_key is None:
        return {
            "document_id": document_id,
            "work_id": None,
            "work_key": None,
            "work_id_method": None,
            "status": "unresolved",
            "legacy_paper_id": paper_id,
        }
    key, method = work_key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return {
        "document_id": document_id,
        "work_id": f"W_{digest[:12]}",
        "work_key": key,
        "work_id_method": method,
        "status": "verified" if method in {"doi", "arxiv"} else "provisional",
        "legacy_paper_id": paper_id,
    }


def truncate_filename(stem: str, maximum_bytes: int) -> str:
    value = stem
    while value and len(f"{value}.pdf".encode("utf-8")) > maximum_bytes:
        value = value[:-1]
    return value.rstrip(" ._-") or "Untitled Paper"


def canonical_pdf_filename(
    title: str,
    maximum_bytes: int = MAX_FILENAME_BYTES,
) -> str:
    """把已核验标题转换成跨平台、安全且可读的 PDF 文件名。"""

    normalized = unicodedata.normalize("NFKC", title).strip()
    normalized = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    if not normalized:
        normalized = "Untitled Paper"
    if normalized.casefold().endswith(".pdf"):
        normalized = normalized[:-4].rstrip(" .")
    if normalized.upper() in RESERVED_FILENAMES:
        normalized = f"_{normalized}"
    normalized = truncate_filename(normalized, maximum_bytes)
    return f"{normalized}.pdf"


def project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def rename_raw_pdf_to_verified_title(
    project_root: Path,
    paper_id: str,
    document: dict[str, Any],
    verified_title: str,
) -> RawPDFRename:
    root = project_root.expanduser().resolve()
    source = document.get("source")
    if not isinstance(source, dict):
        raise ValueError("paper.json.source 必须是 JSON 对象。")
    raw_pdf = source.get("raw_pdf")
    sha256 = source.get("sha256")
    if not isinstance(raw_pdf, str) or not raw_pdf.strip():
        raise ValueError("paper.json.source.raw_pdf 不能为空。")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("paper.json.source.sha256 必须是完整 SHA-256。")

    old_path = (root / raw_pdf).resolve()
    expected_directory = (root / "data" / "raw" / paper_id).resolve()
    if old_path.parent != expected_directory:
        raise ValueError("raw_pdf 不在对应 paper_id 的受控目录中。")
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    if calculate_sha256(old_path) != sha256:
        raise ValueError("raw PDF 的 SHA-256 与 paper.json 不一致。")

    target = expected_directory / canonical_pdf_filename(verified_title)
    if target == old_path:
        rename = RawPDFRename(old_path=old_path, new_path=target, changed=False)
    else:
        if target.exists():
            target = target.with_name(f"{target.stem} [{paper_id}].pdf")
            if target.exists():
                raise FileExistsError(target)
        old_path.rename(target)
        rename = RawPDFRename(old_path=old_path, new_path=target, changed=True)
    source["stored_filename"] = target.name
    source["raw_pdf"] = project_relative(target, root)
    source["filename_policy"] = FILENAME_POLICY
    history = source.get("filename_history")
    filename_history = (
        [value for value in history if isinstance(value, str)]
        if isinstance(history, list)
        else []
    )
    if old_path.name not in filename_history:
        filename_history.append(old_path.name)
    source["filename_history"] = filename_history

    precheck = document.get("precheck")
    if isinstance(precheck, dict):
        precheck["source_path"] = project_relative(target, root)
    return rename


def rollback_raw_pdf_rename(rename: RawPDFRename) -> None:
    if rename.changed and rename.new_path.exists() and not rename.old_path.exists():
        rename.new_path.rename(rename.old_path)
