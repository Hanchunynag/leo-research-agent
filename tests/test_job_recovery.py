from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app.jobs import PersistentJobRepository, PersistentJobWorker
from app.web.jobs import JobManager


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


def test_stale_cancel_request_is_finalized_but_recent_cancel_is_preserved(
    tmp_path: Path,
) -> None:
    repository = PersistentJobRepository(tmp_path)
    stale, _ = repository.submit(
        "scholar.run",
        workspace_id="default",
        scope_version=1,
        payload={"run_id": "RUN_STALE_CANCEL"},
        idempotency_key="scholar:stale-cancel",
    )
    recent, _ = repository.submit(
        "scholar.run",
        workspace_id="default",
        scope_version=1,
        payload={"run_id": "RUN_RECENT_CANCEL"},
        idempotency_key="scholar:recent-cancel",
    )
    repository.claim(stale.job_id, "dead-worker")
    repository.claim(recent.job_id, "live-worker")
    repository.request_cancel(stale.job_id)
    repository.request_cancel(recent.job_id)

    old_heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    with repository._connect() as connection:  # noqa: SLF001 - fixture setup
        connection.execute(
            "UPDATE jobs SET heartbeat_at=? WHERE job_id=?",
            (old_heartbeat, stale.job_id),
        )

    stale_before = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    marked = repository.mark_interrupted(heartbeat_before=stale_before)

    assert [job.job_id for job in marked] == [stale.job_id]
    assert repository.get(stale.job_id).status == "CANCELLED"
    assert repository.get(stale.job_id).finished_at is not None
    assert repository.get(recent.job_id).status == "CANCEL_REQUESTED"


def test_recent_cancel_request_is_not_finalized_by_restart_recovery(
    tmp_path: Path,
) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "scholar.run",
        workspace_id="default",
        scope_version=1,
        payload={"run_id": "RUN_LIVE_CANCEL"},
        idempotency_key="scholar:live-cancel",
    )
    repository.claim(job.job_id, "live-worker")
    repository.request_cancel(job.job_id)

    before_recent_heartbeat = (
        datetime.now(timezone.utc) - timedelta(seconds=60)
    ).isoformat()
    assert repository.mark_interrupted(heartbeat_before=before_recent_heartbeat) == ()
    assert repository.get(job.job_id).status == "CANCEL_REQUESTED"


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


def test_worker_heartbeat_retries_transient_sqlite_errors(tmp_path: Path) -> None:
    class FlakyRepository(PersistentJobRepository):
        failures = 1
        recovered = threading.Event()

        def heartbeat(self, job_id: str, *, worker_id: str, checkpoint=None):  # type: ignore[no-untyped-def]
            if self.failures:
                self.failures -= 1
                raise sqlite3.OperationalError("database is locked")
            value = super().heartbeat(
                job_id,
                worker_id=worker_id,
                checkpoint=checkpoint,
            )
            self.recovered.set()
            return value

    repository = FlakyRepository(tmp_path)
    job, _ = repository.submit(
        "knowledge.index",
        workspace_id="default",
        scope_version=1,
        idempotency_key="index:heartbeat-retry",
    )
    repository.claim_next("worker-1")
    worker = PersistentJobWorker(
        repository,
        {},
        worker_id="worker-1",
        heartbeat_interval_seconds=0.01,
    )
    stop = threading.Event()
    thread = threading.Thread(target=worker._heartbeat_loop, args=(job.job_id, stop))
    thread.start()

    assert repository.recovered.wait(timeout=1)
    stop.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert repository.get(job.job_id).status == "RUNNING"


def test_worker_does_not_clobber_job_recovered_by_another_worker(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "knowledge.index",
        workspace_id="default",
        scope_version=1,
        idempotency_key="index:ownership-fence",
        max_attempts=2,
    )

    def handler(record, context):  # type: ignore[no-untyped-def]
        repository.set_status(
            record.job_id,
            "INTERRUPTED",
            error_type="ProcessRestart",
            error_summary="recovered by replacement worker",
        )
        raise RuntimeError("old worker lost ownership")

    worker = PersistentJobWorker(
        repository,
        {"knowledge.index": handler},
        worker_id="old-worker",
        heartbeat_interval_seconds=0.01,
    )

    result = worker.run_once()

    assert result is not None
    assert result.status == "INTERRUPTED"
    assert repository.get(job.job_id).status == "INTERRUPTED"


def test_web_job_result_and_events_survive_manager_restart(tmp_path: Path) -> None:
    manager = JobManager(max_workers=1, project_root=tmp_path)
    created = manager.submit(
        "parse", lambda emit: (emit("parse", "parsed", 0.8), {"document_id": "D_1"})[1]
    )
    for _ in range(100):
        snapshot = manager.snapshot(created.job_id)
        if snapshot.status == "succeeded":
            break
        __import__("time").sleep(0.005)
    manager.close()

    restored_manager = JobManager(max_workers=1, project_root=tmp_path)
    restored = restored_manager.snapshot(created.job_id)
    restored_manager.close()

    assert restored.status == "succeeded"
    assert restored.result == {"document_id": "D_1"}
    assert [value.stage for value in restored.events] == [
        "queued",
        "running",
        "parse",
        "completed",
    ]


def test_web_running_closure_is_explicitly_failed_after_restart(
    tmp_path: Path,
) -> None:
    repository = PersistentJobRepository(tmp_path)
    job, _ = repository.submit(
        "web.parse",
        workspace_id="default",
        scope_version=1,
        idempotency_key="web.parse:orphan",
        max_attempts=2,
    )
    repository.claim(job.job_id, "dead-web-thread")

    manager = JobManager(max_workers=1, project_root=tmp_path)
    snapshot = manager.snapshot(job.job_id)
    manager.close()

    assert snapshot.status == "failed"
    assert "cannot be reconstructed" in str(snapshot.error)
