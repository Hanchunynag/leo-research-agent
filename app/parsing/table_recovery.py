"""Recover MinerU image-only tables with the isolated PaddleOCR environment.

Only this case is handled here: MinerU emitted a ``table`` block with an
image but no usable HTML. The module does not search pages for missed tables
and never changes the semantic type of an existing block.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


TABLE_RECOVERY_POLICY_VERSION = "1.0"
DEFAULT_TABLE_RECOVERY_TIMEOUT_SECONDS = 900


def is_usable_table_html(value: Any) -> bool:
    """Return whether a value contains a minimally usable HTML table."""

    if not isinstance(value, str):
        return False

    normalized = value.strip().lower()
    return (
        "<table" in normalized
        and ("<td" in normalized or "<th" in normalized)
    )


def resolve_paddleocr_executable(
    project_root: Path,
    configured_path: Path | None = None,
) -> Path:
    """Resolve the Python executable in the dedicated PaddleOCR venv."""

    candidates: list[Path] = []

    if configured_path is not None:
        candidates.append(configured_path)

    environment_path = os.environ.get("LEO_PADDLEOCR_EXECUTABLE")
    if environment_path:
        candidates.append(Path(environment_path))

    candidates.extend(
        [
            project_root / ".venv-paddleocr" / "bin" / "python",
            project_root / ".venv-paddleocr" / "Scripts" / "python.exe",
        ]
    )

    for candidate in candidates:
        # Do not call Path.resolve() here. The venv's ``bin/python`` is often
        # a symlink; resolving it would silently switch to the system Python
        # and lose the PaddleOCR site-packages.
        path = candidate.expanduser().absolute()
        if path.is_file() and os.access(path, os.X_OK):
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "未找到 PaddleOCR 专用环境的 Python 可执行文件。\n"
        "请创建 .venv-paddleocr，或设置 LEO_PADDLEOCR_EXECUTABLE。\n"
        f"已检查：\n{searched}"
    )


def _relative_path(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(image_path: Path) -> str:
    payload = f"{TABLE_RECOVERY_POLICY_VERSION}:{_sha256_file(image_path)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _load_cached_result(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict) or payload.get("done") is not True:
        return None
    return payload


def _run_worker(
    executable: Path,
    worker: Path,
    image_path: Path,
    output_path: Path,
    cache_home: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        str(executable),
        str(worker),
        "--input-image",
        str(image_path),
        "--output-json",
        str(output_path),
        "--cache-home",
        str(cache_home),
    ]

    completed = subprocess.run(
        command,
        cwd=worker.parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "无输出"
        raise RuntimeError(
            f"PaddleOCR 表格恢复失败，退出码 {completed.returncode}：{details[-4000:]}"
        )

    result = _load_cached_result(output_path)
    if result is None:
        raise RuntimeError("PaddleOCR worker 未生成有效的表格恢复 JSON。")
    return result


def recover_image_only_tables(
    document: dict[str, Any],
    project_root: Path,
    paddleocr_executable: Path | None = None,
    enabled: bool = True,
    timeout_seconds: int = DEFAULT_TABLE_RECOVERY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Recover only MinerU ``table`` blocks whose HTML is missing.

    The operation is fail-soft: the canonical document is still produced and
    an unsuccessful table remains image-only with a diagnostic record.
    """

    blocks = document.get("blocks")
    candidates = (
        [
            block
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "table"
            and not is_usable_table_html(block.get("table_html"))
            and isinstance(block.get("image_path"), str)
            and bool(block.get("image_path"))
        ]
        if isinstance(blocks, list)
        else []
    )

    report: dict[str, Any] = {
        "policy_version": TABLE_RECOVERY_POLICY_VERSION,
        "enabled": enabled,
        "candidate_count": len(candidates),
        "recovered_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "records": [],
    }

    if not enabled or not candidates:
        return report

    worker = project_root / "scripts" / "paddleocr_table_worker.py"

    try:
        executable = resolve_paddleocr_executable(
            project_root=project_root,
            configured_path=paddleocr_executable,
        )
    except FileNotFoundError as error:
        for block in candidates:
            block["table_recovery"] = {
                "status": "unavailable",
                "engine": "paddleocr",
                "error": str(error),
            }
        report["skipped_count"] = len(candidates)
        report["error"] = str(error)
        return report

    if not worker.is_file():
        unavailable_error = f"PaddleOCR worker 不存在：{worker}"
        for block in candidates:
            block["table_recovery"] = {
                "status": "unavailable",
                "engine": "paddleocr",
                "error": unavailable_error,
            }
        report["skipped_count"] = len(candidates)
        report["error"] = unavailable_error
        return report

    output_root = (
        project_root
        / "data"
        / "parsed"
        / str(document.get("paper_id", "unknown"))
        / "paddleocr"
        / "table"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cache_home = project_root / "data" / "models" / "paddleocr"

    for block in candidates:
        block_id = str(block.get("block_id", "table"))
        image_path = Path(str(block["image_path"])).expanduser().resolve()
        record: dict[str, Any] = {
            "engine": "paddleocr",
            "policy_version": TABLE_RECOVERY_POLICY_VERSION,
            "block_id": block_id,
            "input_image": _relative_path(image_path, project_root),
        }

        if not image_path.is_file():
            record.update(
                {
                    "status": "failed",
                    "error": f"表格图片不存在：{image_path}",
                }
            )
            block["table_recovery"] = record
            report["failed_count"] += 1
            report["records"].append(record)
            continue

        cache_key = _cache_key(image_path)
        output_path = output_root / f"{cache_key}.json"
        record["output_json"] = _relative_path(output_path, project_root)
        record["cache_key"] = cache_key

        try:
            result = _load_cached_result(output_path)
            cache_hit = result is not None
            if result is None:
                result = _run_worker(
                    executable=executable,
                    worker=worker,
                    image_path=image_path,
                    output_path=output_path,
                    cache_home=cache_home,
                    timeout_seconds=timeout_seconds,
                )

            html = result.get("pred_html")
            if not is_usable_table_html(html):
                raise RuntimeError("PaddleOCR 未返回可用的 HTML 表格。")

            block["table_html_paddleocr"] = html
            block["table_cells_paddleocr"] = result.get("table_ocr_pred")
            block["table_html"] = html
            quality = block.get("quality")
            if not isinstance(quality, dict):
                quality = {}
                block["quality"] = quality
            issues = quality.get("issues")
            if not isinstance(issues, list):
                issues = []
            quality["issues"] = [
                issue for issue in issues if issue != "table_html_empty"
            ]
            quality["status"] = "usable"
            quality["retrieval_enabled"] = True

            record.update(
                {
                    "status": "recovered",
                    "cache_hit": cache_hit,
                    "cell_count": len(result.get("cell_box_list", [])),
                    "paddleocr_version": result.get("paddleocr_version"),
                }
            )
            block["table_recovery"] = record
            report["recovered_count"] += 1
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            record.update({"status": "failed", "error": str(error)})
            block["table_recovery"] = record
            report["failed_count"] += 1

        report["records"].append(record)

    return report
