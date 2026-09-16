"""Compare two completed legacy/CrewAI orchestration evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make direct script invocation resolve the repository package.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# ruff: noqa: E402
from app.orchestration.evaluation import OrchestrationEvaluationReport, compare_reports
from app.storage import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--crewai-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _load(path: Path) -> OrchestrationEvaluationReport:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    evaluation = value.get("evaluation") if isinstance(value, dict) else None
    if not isinstance(evaluation, dict):
        raise ValueError(f"{path} 不包含 evaluation report。")
    metrics = evaluation.get("metrics")
    records = evaluation.get("records")
    if not isinstance(metrics, dict) or not isinstance(records, list):
        raise ValueError(f"{path} 的 evaluation contract 无效。")
    backend = str(
        evaluation.get("backend")
        or value.get("backend")
        or path.stem
    )
    return OrchestrationEvaluationReport(
        backend=backend,
        evaluated_at=str(evaluation.get("evaluated_at") or ""),
        records=tuple(
            row for row in records if isinstance(row, dict)
        ),
        metrics={
            str(key): value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) or value is None
        },
    )


def main() -> int:
    args = _parser().parse_args()
    result = compare_reports(
        _load(args.legacy_report),
        _load(args.crewai_report),
    )
    if args.output is not None:
        write_json_atomic(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
