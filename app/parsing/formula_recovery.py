"""Two-rule MinerU formula fallback through PaddleOCR-VL.

MinerU remains the default formula recognizer. PaddleOCR-VL is invoked only
for deterministic hard failures or formulas whose structure is sufficiently
complex to justify a second pass. This module intentionally contains no LLM
or Qwen dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from app.parsing.quality import formula_hard_error_issues
from app.parsing.table_recovery import (
    _load_cached_result,
    _relative_path,
    resolve_paddleocr_executable,
)


FORMULA_RECOVERY_POLICY_VERSION = "1.0"
DEFAULT_FORMULA_RECOVERY_TIMEOUT_SECONDS = 900


def is_usable_formula_latex(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def formula_risk_reasons(latex: Any, image_path: Path | None = None) -> list[str]:
    """Classify formula complexity without judging mathematical correctness."""

    value = latex if isinstance(latex, str) else ""
    reasons: list[str] = []
    if re.search(
        r"\\begin\s*\{(?:array|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|aligned|split|gather|multline)\}",
        value,
    ):
        reasons.append("matrix_or_multiline_environment")
    if len(re.findall(r"\\\\", value)) >= 1:
        reasons.append("multiple_formula_lines")
    if len(re.findall(r"\\frac\b", value)) >= 2:
        reasons.append("multiple_fractions")
    if len(re.findall(r"(?:\\sum|\\int)\b", value)) >= 2:
        reasons.append("multiple_sum_or_integral")
    if value.count("_") + value.count("^") >= 8:
        reasons.append("many_subscripts_or_superscripts")
    if len(value) >= 350:
        reasons.append("long_latex")

    if image_path is not None and image_path.is_file():
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
            if height > 0 and width / height >= 4.0:
                reasons.append("wide_formula_image")
        except (OSError, ValueError):
            pass
    return list(dict.fromkeys(reasons))


def is_high_risk_formula(latex: Any, image_path: Path | None = None) -> bool:
    reasons = formula_risk_reasons(latex, image_path)
    return bool(
        reasons
        and (
            any(
                reason
                in {
                    "matrix_or_multiline_environment",
                    "multiple_formula_lines",
                    "long_latex",
                    "wide_formula_image",
                }
                for reason in reasons
            )
            or len(reasons) >= 2
        )
    )


def _cache_key(image_path: Path, latex: Any) -> str:
    digest = hashlib.sha256()
    digest.update(FORMULA_RECOVERY_POLICY_VERSION.encode("utf-8"))
    digest.update(image_path.read_bytes())
    digest.update(str(latex or "").encode("utf-8"))
    return digest.hexdigest()[:24]


def _normalize_formula(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    result = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", result, flags=re.I)
    result = result.strip()
    if result.startswith("$$") and result.endswith("$$"):
        result = result[2:-2].strip()
    elif result.startswith("$") and result.endswith("$"):
        result = result[1:-1].strip()
    return result or None


def _run_worker_batch(
    executable: Path,
    worker: Path,
    manifest_path: Path,
    cache_home: Path,
    timeout_seconds: int,
) -> None:
    completed = subprocess.run(
        [
            str(executable),
            str(worker),
            "--input-manifest",
            str(manifest_path),
            "--cache-home",
            str(cache_home),
            "--device",
            "cpu",
        ],
        cwd=worker.parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "无输出"
        raise RuntimeError(
            f"PaddleOCR-VL 公式恢复 worker 失败，退出码 {completed.returncode}："
            f"{details[-4000:]}"
        )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def recover_high_risk_formulas(
    document: dict[str, Any],
    project_root: Path,
    paddleocr_executable: Path | None = None,
    enabled: bool = True,
    timeout_seconds: int = DEFAULT_FORMULA_RECOVERY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    blocks = document.get("blocks")
    candidates = (
        [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "equation"
        ]
        if isinstance(blocks, list)
        else []
    )
    report: dict[str, Any] = {
        "policy_version": FORMULA_RECOVERY_POLICY_VERSION,
        "enabled": enabled,
        "formula_count": len(candidates),
        "hard_error_count": 0,
        "high_risk_count": 0,
        "fallback_count": 0,
        "recovered_count": 0,
        "failed_count": 0,
        "skipped_no_image_count": 0,
        "records": [],
    }
    if not enabled or not candidates:
        return report

    pending: list[tuple[dict[str, Any], Any, Path | None, list[str], list[str], bool]] = []
    for block in candidates:
        latex = block.get("latex") or block.get("latex_raw")
        image_value = block.get("image_path")
        image_path = None
        if isinstance(image_value, str) and image_value.strip():
            raw_image_path = Path(image_value).expanduser()
            image_path = (
                raw_image_path
                if raw_image_path.is_absolute()
                else project_root / raw_image_path
            ).resolve()
        hard_errors = formula_hard_error_issues(latex)
        risk_reasons = formula_risk_reasons(latex, image_path)
        high_risk = is_high_risk_formula(latex, image_path)
        if hard_errors:
            report["hard_error_count"] += 1
        if high_risk:
            report["high_risk_count"] += 1
        if not hard_errors and not high_risk:
            continue
        pending.append((block, latex, image_path, hard_errors, risk_reasons, high_risk))

    if not pending:
        return report

    worker = project_root / "scripts" / "paddleocr_formula_worker.py"
    if not worker.is_file():
        error_message = f"PaddleOCR-VL worker 不存在：{worker}"
        report["error"] = error_message
        for block, _latex, image_path, hard_errors, risk_reasons, _high_risk in pending:
            unavailable_record_no_executable: dict[str, Any] = {
                "block_id": str(block.get("block_id", "equation")),
                "hard_errors": hard_errors,
                "risk_reasons": risk_reasons,
                "status": "unavailable",
                "error": error_message,
            }
            if image_path is None or not image_path.is_file():
                unavailable_record_no_executable["status"] = "skipped_no_image"
                report["skipped_no_image_count"] += 1
            else:
                report["fallback_count"] += 1
                report["failed_count"] += 1
            report["records"].append(unavailable_record_no_executable)
        return report

    try:
        executable = resolve_paddleocr_executable(
            project_root=project_root,
            configured_path=paddleocr_executable,
        )
    except FileNotFoundError as error:
        report["error"] = str(error)
        for block, _latex, image_path, hard_errors, risk_reasons, _high_risk in pending:
            unavailable_record_no_venv: dict[str, Any] = {
                "block_id": str(block.get("block_id", "equation")),
                "hard_errors": hard_errors,
                "risk_reasons": risk_reasons,
                "status": "unavailable",
                "error": str(error),
            }
            if image_path is None or not image_path.is_file():
                unavailable_record_no_venv["status"] = "skipped_no_image"
                report["skipped_no_image_count"] += 1
            else:
                report["fallback_count"] += 1
                report["failed_count"] += 1
            report["records"].append(unavailable_record_no_venv)
        return report

    output_root = (
        project_root
        / "data"
        / "parsed"
        / str(document.get("paper_id", "unknown"))
        / "paddleocr"
        / "formula"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cache_home = project_root / "data" / "models" / "paddleocr"

    batch_jobs: list[dict[str, str]] = []
    preexisting_cache_paths: set[Path] = set()
    for _block, latex, image_path, _hard_errors, _risk_reasons, _high_risk in pending:
        if image_path is None or not image_path.is_file():
            continue
        cache_key = _cache_key(image_path, latex)
        output_path = output_root / f"{cache_key}.json"
        if _load_cached_result(output_path) is None:
            batch_jobs.append(
                {
                    "input_image": str(image_path),
                    "output_json": str(output_path),
                }
            )
        else:
            preexisting_cache_paths.add(output_path)

    batch_error: str | None = None
    if batch_jobs:
        manifest_path = output_root / "formula_manifest.json"
        manifest_path.write_text(
            json.dumps(batch_jobs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            _run_worker_batch(
                executable=executable,
                worker=worker,
                manifest_path=manifest_path,
                cache_home=cache_home,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            batch_error = str(error)

    for block, latex, image_path, hard_errors, risk_reasons, _high_risk in pending:
        record: dict[str, Any] = {
            "block_id": str(block.get("block_id", "equation")),
            "hard_errors": hard_errors,
            "risk_reasons": risk_reasons,
            "status": "skipped_no_image",
        }
        if image_path is None or not image_path.is_file():
            report["skipped_no_image_count"] += 1
            report["records"].append(record)
            continue
        report["fallback_count"] += 1
        cache_key = _cache_key(image_path, latex)
        output_path = output_root / f"{cache_key}.json"
        record.update(
            {
                "input_image": _relative_path(image_path, project_root),
                "output_json": _relative_path(output_path, project_root),
                "cache_key": cache_key,
            }
        )
        try:
            result = _load_cached_result(output_path)
            cache_hit = output_path in preexisting_cache_paths
            if result is None:
                worker_payload = _load_json(output_path)
                worker_error = (
                    worker_payload.get("error")
                    if isinstance(worker_payload, dict)
                    else None
                )
                if batch_error:
                    raise RuntimeError(batch_error)
                raise RuntimeError(
                    str(worker_error)
                    if worker_error
                    else "PaddleOCR-VL worker 未生成有效的公式恢复 JSON。"
                )
            recovered = _normalize_formula(result.get("latex"))
            if not is_usable_formula_latex(recovered):
                raise RuntimeError("PaddleOCR-VL 未返回可用 LaTeX。")
            block["latex_mineru"] = latex
            block["latex_paddleocr_vl"] = recovered
            block["latex"] = recovered
            quality = block.get("quality")
            if isinstance(quality, dict):
                issues = quality.get("issues")
                if isinstance(issues, list):
                    quality["issues"] = [
                        issue
                        for issue in issues
                        if not str(issue).startswith("formula_")
                    ]
                quality["retrieval_enabled"] = True
                quality["status"] = "usable"
            record.update(
                {
                    "status": "recovered",
                    "cache_hit": cache_hit,
                    "engine": "paddleocr-vl",
                    "paddleocr_version": result.get("paddleocr_version"),
                }
            )
            report["recovered_count"] += 1
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            record.update({"status": "failed", "error": str(error)})
            report["failed_count"] += 1
        report["records"].append(record)
    return report
