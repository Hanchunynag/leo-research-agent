from __future__ import annotations

from pathlib import Path

import pytest

from app.jobs import PersistentJobRepository


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

