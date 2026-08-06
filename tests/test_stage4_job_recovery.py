from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app.jobs import PersistentJobRepository, PersistentJobWorker


def test_job_repository_persists_and_deduplicates_submission(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    first, created = repository.submit(
        "document.parse",
        workspace_id="default",
        scope_version=1,
        document_id="D_1",
        payload={"source_reference": "uploads/U_1"},
        idempotency_key="document.parse:default:D_1:mineru-v1",
    )
    duplicate, duplicate_created = PersistentJobRepository(tmp_path).submit(
        "document.parse",
        workspace_id="default",
        scope_version=1,
        document_id="D_1",
        payload={"source_reference": "uploads/U_1"},
        idempotency_key="document.parse:default:D_1:mineru-v1",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.job_id == first.job_id
    assert duplicate.status == "QUEUED"


def test_job_claim_and_heartbeat_survive_repository_reopen(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "knowledge.index",
        workspace_id="default",
        scope_version=2,
        generation_id="IG_1",
        payload={"profile_id": "profile-1"},
        idempotency_key="knowledge.index:IG_1",
    )
    claimed = repository.claim_next("worker-1")
    assert claimed is not None
    repository.heartbeat(
        job.job_id, worker_id="worker-1", checkpoint={"document_offset": 2}
    )

    restored = PersistentJobRepository(tmp_path).get(job.job_id)

    assert restored.status == "RUNNING"
    assert restored.attempt == 1
    assert restored.heartbeat_at is not None
    assert restored.checkpoint == {"document_offset": 2}


@pytest.mark.parametrize(
    "payload",
    [
        {"pdf_bytes": "binary"},
        {"api_key": "secret"},
        {"prompt": "hidden"},
        {"model_output": "full response"},
    ],
)
def test_job_repository_rejects_sensitive_or_large_artifacts(
    tmp_path: Path, payload: dict[str, str]
) -> None:
    repository = PersistentJobRepository(tmp_path)

    with pytest.raises(ValueError, match="禁止持久化字段"):
        repository.submit(
            "document.parse",
            workspace_id="default",
            scope_version=1,
            payload=payload,
            idempotency_key=f"bad:{next(iter(payload))}",
        )


def test_process_restart_marks_interrupted_then_retry_pending(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "document.parse",
        workspace_id="default",
        scope_version=1,
        payload={"source_reference": "upload-1"},
        idempotency_key="parse:restart",
        max_attempts=2,
    )
    repository.claim_next("dead-worker")
    future = datetime.now(timezone.utc) + timedelta(seconds=1)

    interrupted = repository.mark_interrupted(
        heartbeat_before=future.isoformat()
    )
    assert interrupted[0].status == "INTERRUPTED"
    recovered = repository.recover_interrupted()

    assert recovered[0].status == "RETRY_PENDING"
    assert PersistentJobRepository(tmp_path).get(job.job_id).attempt == 1


def test_process_restart_fails_job_at_retry_limit(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "document.parse",
        workspace_id="default",
        scope_version=1,
        idempotency_key="parse:no-retries",
        max_attempts=1,
    )
    repository.claim_next("dead-worker")
    PersistentJobWorker(
        repository, {}, worker_id="replacement"
    ).recover_after_restart(stale_after_seconds=0)

    failed = repository.get(job.job_id)
    assert failed.status == "FAILED"
    assert failed.finished_at is not None
    assert failed.error_type == "ProcessRestart"


def test_worker_retries_then_succeeds_with_reference_only(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "knowledge.index",
        workspace_id="default",
        scope_version=1,
        generation_id="IG_1",
        idempotency_key="index:retry",
        max_attempts=2,
    )
    attempts = 0

    def handler(record, context):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        context.checkpoint({"stage": "indexing", "attempt": attempts})
        if attempts == 1:
            raise TimeoutError("temporary")
        return "data/results/index-1.json"

    worker = PersistentJobWorker(
        repository,
        {"knowledge.index": handler},
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )

    results = worker.run_until_idle()

    assert [value.status for value in results] == ["RETRY_PENDING", "SUCCEEDED"]
    completed = repository.get(job.job_id)
    assert completed.attempt == 2
    assert completed.result_reference == "data/results/index-1.json"
    assert completed.checkpoint == {"stage": "indexing", "attempt": 2}
