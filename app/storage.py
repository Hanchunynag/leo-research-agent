"""项目内 JSON 和 JSONL 文件的原子写入工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json_atomic(path: Path, payload: Any) -> None:
    """先写临时文件，再替换目标 JSON，避免留下半截文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_jsonl_atomic(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """原子写入一条记录一行的 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    content = "\n".join(lines)

    if lines:
        content += "\n"

    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
