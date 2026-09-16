"""Run the production Scholar Worker as a separate process.

Examples:
    uv run python scripts/run_scholar_worker.py
    uv run python scripts/run_scholar_worker.py --once --max-jobs 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(
    os.getenv("LEO_PROJECT_ROOT") or Path(__file__).resolve().parents[1]
).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="LEO Scholar CrewAI Worker")
    parser.add_argument("--once", action="store_true", help="处理当前队列后退出")
    parser.add_argument("--max-jobs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--stale-after-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_jobs < 1 or args.poll_seconds <= 0:
        raise SystemExit("--max-jobs 必须为正数，--poll-seconds 必须为正数。")

    from app.scholar.runs import ScholarRunManager, ScholarRunWorker

    manager = ScholarRunManager(PROJECT_ROOT)
    worker = ScholarRunWorker(manager)
    try:
        recovered = worker.recover_after_restart(
            stale_after_seconds=args.stale_after_seconds
        )
        processed = []
        if args.once:
            processed = list(worker.run_until_idle(max_jobs=args.max_jobs))
        else:
            while len(processed) < args.max_jobs:
                result = worker.run_once()
                if result is None:
                    time.sleep(args.poll_seconds)
                    continue
                processed.append(result)
    finally:
        status = worker.status()
        worker.close()
    print(
        json.dumps(
            {
                "worker_id": status["worker_id"],
                "recovered": [asdict(value) for value in recovered],
                "processed": [asdict(value) for value in processed],
                "status": status,
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
