"""Canonical block quality checks before structure and chunk construction.

The validator is deliberately conservative about semantic correction. It can
remove transport-level control/format characters from emitted fields and
record the finding, but it never guesses whether a word such as ``diferent``
should be corrected. The original MinerU block remains available under
``raw_block`` for audit.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


QUALITY_POLICY_VERSION = "1.0"

_TEXT_FIELDS = (
    "text",
    "caption",
    "latex",
    "latex_raw",
    "table_html",
    "table_html_raw",
)
_SAFE_CONTROLS = {"\t", "\n", "\r"}
_UNSAFE_FORMAT_CHARS = {
    "\u061c",  # Arabic letter mark
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2060",  # word joiner
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
    "\ufeff",  # BOM / zero-width no-break space
}
_FORMULA_TYPES = {"equation", "formula"}
_TEXT_TYPES = {"paragraph", "list", "title", "algorithm"}
_MEANINGFUL_CATEGORIES = {
    "Ll",
    "Lm",
    "Lo",
    "Lt",
    "Lu",
    "Nd",
}


def _sanitize_value(value: str) -> tuple[str, list[str], list[str]]:
    cleaned: list[str] = []
    control_codes: list[str] = []
    format_codes: list[str] = []
    for character in value:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if character in _SAFE_CONTROLS:
            cleaned.append(character)
        elif (
            category in {"Cc", "Cs"}
            or character in _UNSAFE_FORMAT_CHARS
        ):
            code = f"U+{codepoint:04X}"
            if category in {"Cc", "Cs"}:
                control_codes.append(code)
            else:
                format_codes.append(code)
            cleaned.append(" ")
        else:
            cleaned.append(character)
    return "".join(cleaned), control_codes, format_codes


def _formula_issues(value: str) -> list[str]:
    issues: list[str] = []
    if "\x00" in value:
        issues.append("formula_control_character")

    environment_stack: list[str] = []
    environment_pattern = re.compile(
        r"\\(begin|end)\s*\{([^{}]+)\}"
    )
    for match in environment_pattern.finditer(value):
        action, name = match.groups()
        if action == "begin":
            environment_stack.append(name)
        elif not environment_stack or environment_stack[-1] != name:
            issues.append("formula_environment_mismatch")
            break
        else:
            environment_stack.pop()
    if environment_stack and "formula_environment_mismatch" not in issues:
        issues.append("formula_environment_mismatch")

    # TeX groups and visual delimiters can legally cross each other (for
    # example ``\bigg[ \frac{...}{...} \bigg]``), so a character-level stack
    # produces false positives. Count imbalance is deliberately weaker but
    # safe for the extraction-quality gate.
    delimiter_pairs = (("{", "}"), ("[", "]"), ("(", ")"))
    if any(value.count(left) != value.count(right) for left, right in delimiter_pairs):
        issues.append("formula_unbalanced_delimiter")
    if len(re.findall(r"\\left\b", value)) != len(
        re.findall(r"\\right\b", value)
    ):
        issues.append("formula_left_right_mismatch")
    return issues


def formula_hard_error_issues(value: Any) -> list[str]:
    """Return only deterministic formula failures for fallback gating."""

    if not isinstance(value, str) or not value.strip():
        return ["formula_latex_empty"]
    cleaned, control_codes, _ = _sanitize_value(value)
    issues = ["formula_control_character"] if control_codes else []
    issues.extend(_formula_issues(cleaned))
    if cleaned.rstrip().endswith("\\"):
        issues.append("formula_truncated")
    return list(dict.fromkeys(issues))


def _text_coverage_issue(value: str) -> bool:
    visible = [character for character in value if not character.isspace()]
    if len(visible) < 32:
        return False
    meaningful = sum(
        unicodedata.category(character) in _MEANINGFUL_CATEGORIES
        for character in visible
    )
    return meaningful / len(visible) < 0.35


def validate_canonical_document(document: dict[str, Any]) -> dict[str, Any]:
    """Validate and minimally sanitize Canonical blocks in place.

    Hard structural failures disable retrieval for the affected block. A
    control character is replaced with a space so it cannot reach Structure or
    Chunk output, while the issue is retained for audit and the raw MinerU
    payload remains unchanged.
    """

    blocks = document.get("blocks")
    issue_counts: Counter[str] = Counter()
    invalid_block_count = 0
    sanitized_block_count = 0

    if not isinstance(blocks, list):
        return {
            "policy_version": QUALITY_POLICY_VERSION,
            "block_count": 0,
            "invalid_block_count": 0,
            "sanitized_block_count": 0,
            "issue_counts": {},
        }

    for block in blocks:
        if not isinstance(block, dict):
            continue
        quality = block.get("quality")
        if not isinstance(quality, dict):
            quality = {"status": "usable", "issues": [], "retrieval_enabled": True}
            block["quality"] = quality
        issues = quality.get("issues")
        if not isinstance(issues, list):
            issues = []

        block_issues: list[str] = []
        for field in _TEXT_FIELDS:
            value = block.get(field)
            if not isinstance(value, str):
                continue
            cleaned, control_codes, format_codes = _sanitize_value(value)
            if cleaned != value:
                block[field] = cleaned
                sanitized_block_count += 1
            if control_codes:
                block_issues.append("control_character")
                block.setdefault("quality_details", {})["control_codes"] = sorted(
                    set(control_codes)
                )
            if format_codes:
                block_issues.append("unsafe_unicode_format_character")
                block.setdefault("quality_details", {})["format_codes"] = sorted(
                    set(format_codes)
                )
            if "\ufffd" in value:
                block_issues.append("replacement_character")

        block_type = str(block.get("type", ""))
        latex = block.get("latex") or block.get("latex_raw")
        if block_type in _FORMULA_TYPES:
            block_issues.extend(formula_hard_error_issues(latex))

        if block_type in {"figure", "table"}:
            has_caption_or_text = any(
                isinstance(block.get(field), str) and block[field].strip()
                for field in ("caption", "text")
            )
            has_table = block_type == "table" and bool(
                block.get("table_html") or block.get("table_html_raw")
            )
            if block_type == "figure" and not has_caption_or_text:
                block_issues.append("empty_figure")
            elif block_type == "table" and not has_caption_or_text and not has_table and not block.get("image_path"):
                block_issues.append(f"empty_{block_type}")

        text = block.get("text")
        if block_type in _TEXT_TYPES and isinstance(text, str):
            if _text_coverage_issue(text):
                block_issues.append("text_character_coverage_low")

        block_issues = list(dict.fromkeys(block_issues))
        for issue in block_issues:
            if issue not in issues:
                issues.append(issue)
            issue_counts[issue] += 1

        quality["issues"] = issues
        if block_issues:
            quality["status"] = "degraded"
        hard_issues = {
            "empty_figure",
            "empty_table",
            "formula_environment_mismatch",
            "formula_unbalanced_delimiter",
            "formula_left_right_mismatch",
            "formula_latex_empty",
            "formula_truncated",
            "replacement_character",
            "text_character_coverage_low",
        }
        if hard_issues.intersection(block_issues):
            quality["retrieval_enabled"] = False
            invalid_block_count += 1
        quality["validation_passed"] = not bool(
            hard_issues.intersection(block_issues)
        )

    return {
        "policy_version": QUALITY_POLICY_VERSION,
        "block_count": len(blocks),
        "invalid_block_count": invalid_block_count,
        "sanitized_block_count": sanitized_block_count,
        "issue_counts": dict(issue_counts),
    }
