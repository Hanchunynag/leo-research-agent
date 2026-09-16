"""Run/Worker metrics projection.

Metrics are derived from the durable Run and Job stores so a process restart
does not reset the control-plane view.  The contract is JSON-first; a future
Prometheus adapter can consume the same names without changing the domain.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from app.scholar.runs import ScholarRunManager


def _duration_ms(started: str | None, finished: str | None) -> float | None:
    if not started or not finished:
        return None
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds() * 1000)


def build_metrics_snapshot(project_root: Path, *, run_manager: ScholarRunManager | None = None) -> dict[str, Any]:
    manager = run_manager or ScholarRunManager(project_root)
    repository = manager.repository
    jobs = repository.list()
    runs = manager.list_runs()
    scholar_jobs = [job for job in jobs if job.job_type.startswith("scholar.")]
    scholar_runs = [run for run in runs if run.get("project_id")]
    durations = [
        value
        for value in (_duration_ms(run.get("started_at"), run.get("completed_at")) for run in scholar_runs)
        if value is not None
    ]
    active = [job for job in scholar_jobs if job.status == "RUNNING"]
    waiting = [run for run in scholar_runs if run.get("status") == "WAITING_HUMAN_APPROVAL"]
    succeeded = [run for run in scholar_runs if run.get("status") == "COMPLETED"]
    failed = [run for run in scholar_runs if run.get("status") == "FAILED"]
    agent_calls = 0
    tool_calls = 0
    token_usage = 0
    cost_values: list[float] = []
    for run in scholar_runs:
        run_id = str(run.get("run_id") or "")
        for event in manager.event_store.list(run_id):
            metadata = event.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            kind = str(metadata.get("kind") or "")
            status = str(event.get("status") or "")
            if kind == "agent" and status == "RUNNING":
                agent_calls += 1
            if kind == "tool" and status == "RUNNING":
                tool_calls += 1
            usage = metadata.get("usage")
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
                if isinstance(total, (int, float)):
                    token_usage += int(total)
                reported_cost = next(
                    (
                        usage.get(field)
                        for field in ("cost", "total_cost", "total_cost_usd")
                        if isinstance(usage.get(field), (int, float))
                    ),
                    None,
                )
                if isinstance(reported_cost, (int, float)):
                    cost_values.append(float(reported_cost))
            direct_total = metadata.get("total_tokens")
            if isinstance(direct_total, (int, float)) and not isinstance(usage, dict):
                token_usage += int(direct_total)
            direct_cost = metadata.get("cost")
            if isinstance(direct_cost, (int, float)) and not isinstance(usage, dict):
                cost_values.append(float(direct_cost))
    return {
        "metrics": {
            "runs_total": len(scholar_runs),
            "runs_success": len(succeeded),
            "runs_failed": len(failed),
            "runs_waiting_approval": len(waiting),
            "run_latency_p50_ms": sorted(durations)[(len(durations) - 1) // 2] if durations else None,
            "run_latency_p95_ms": sorted(durations)[min(len(durations) - 1, round((len(durations) - 1) * 0.95))] if durations else None,
            "agent_calls": agent_calls,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
            "cost": round(sum(cost_values), 8) if cost_values else None,
            "provider_failures": sum(
                "provider" in str(job.error_type or "").casefold()
                or "provider" in str(job.error_summary or "").casefold()
                for job in scholar_jobs
            ),
            "queue_depth": sum(job.status in {"QUEUED", "RETRY_PENDING"} for job in scholar_jobs),
            "worker_active": len(active),
        },
        "labels": {
            "orchestration_backend": "crewai",
            "store": "sqlite-wal",
        },
    }
