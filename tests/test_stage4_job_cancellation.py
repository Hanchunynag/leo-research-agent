from __future__ import annotations

import threading
import time
from pathlib import Path

from app.jobs import LongTaskSubmitter, PersistentJobRepository, PersistentJobWorker


def test_queued_job_can_be_cancelled_without_execution(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "document.parse",
        workspace_id="default",
        scope_version=1,
        idempotency_key="parse:cancel-queued",
    )

    cancelled = repository.request_cancel(job.job_id)

    assert cancelled.status == "CANCELLED"
    assert repository.claim_next("worker") is None


def test_running_job_cooperatively_cancels_at_checkpoint(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "knowledge.index",
        workspace_id="default",
        scope_version=1,
        idempotency_key="index:cancel-running",
    )
    started = threading.Event()

    def handler(record, context):  # type: ignore[no-untyped-def]
        started.set()
        while True:
            time.sleep(0.005)
            context.checkpoint({"safe_offset": 1})

    worker = PersistentJobWorker(
        repository,
        {"knowledge.index": handler},
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=1)

    requested = repository.request_cancel(job.job_id)
    thread.join(timeout=2)

    assert requested.status == "CANCEL_REQUESTED"
    assert not thread.is_alive()
    assert repository.get(job.job_id).status == "CANCELLED"


def test_long_tool_submission_returns_job_id_and_deduplicates(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    handler = LongTaskSubmitter(repository).handler("document.parse")
    context = {"workspace_id": "default", "scope_version": 1}
    arguments = {"path": "data/runtime/uploads/P_1.pdf", "mode": "lightweight"}

    first = handler(arguments, context)
    duplicate = handler(arguments, context)

    assert first["job_id"] == duplicate["job_id"]
    assert first["status"] == "queued"
    assert first["created"] is True
    assert duplicate["created"] is False

