from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.jobs import PersistentJobRepository, PersistentJobWorker
from app.orchestration.contracts import OrchestrationRequest
from app.scholar import (
    ManuscriptSynchronizer,
    ProjectRevisionConflict,
    ScholarProjectStore,
    ScholarRunManager,
)
from app.session import SessionManager
from app.tenancy import TenantPrincipal


def test_same_session_is_serialized_but_different_identities_do_not_share_idempotency(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    sessions.create(
        "shared session",
        session_id="shared",
        project_id="PROJECT_SHARED",
        tenant_id="tenant-a",
        principal_id="user-a",
    )
    manager = ScholarRunManager(tmp_path, session_manager=sessions)

    def create(value: tuple[str, str, str]) -> object:
        tenant, principal, key = value
        try:
            return manager.create(
                OrchestrationRequest(
                    request_id=key,
                    project_id="PROJECT_SHARED",
                    instruction=key,
                    session_id="shared",
                    tenant_id=tenant,
                    principal_id=principal,
                ),
                idempotency_key=key,
            )
        except Exception as error:  # the assertion below checks the stable code
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        same_session = list(
            pool.map(
                create,
                (
                    ("tenant-a", "user-a", "session-key-1"),
                    ("tenant-a", "user-a", "session-key-2"),
                ),
            )
        )

    assert sum(isinstance(value, dict) for value in same_session) == 1
    busy = next(value for value in same_session if isinstance(value, ValueError))
    assert getattr(busy, "code", None) == "SESSION_BUSY"

    # An identical caller-supplied key is namespaced by tenant and principal;
    # it cannot make one user's Job idempotently resolve to another user's Run.
    first = manager.create(
        OrchestrationRequest(
            request_id="same-key-a",
            project_id="PROJECT_SHARED",
            instruction="tenant A",
            tenant_id="tenant-a",
            principal_id="user-a",
        ),
        idempotency_key="same-key",
    )
    second = manager.create(
        OrchestrationRequest(
            request_id="same-key-b",
            project_id="PROJECT_SHARED",
            instruction="tenant B",
            tenant_id="tenant-b",
            principal_id="user-b",
        ),
        idempotency_key="same-key",
    )
    assert first["run_id"] != second["run_id"]
    assert len(manager.repository.list()) == 3


def test_project_acl_and_manuscript_revision_cas(tmp_path: Path) -> None:
    store = ScholarProjectStore(tmp_path)
    member = TenantPrincipal(tenant_id="tenant-a", principal_id="writer-a")
    with pytest.raises(ValueError) as denied:
        store.check_access(member)
    assert getattr(denied.value, "code", None) == "PROJECT_ACCESS_DENIED"

    store.grant_member(member, role="writer")
    assert store.check_access(member) == "writer"

    root = tmp_path / "paper"
    (root / "sections").mkdir(parents=True)
    (root / "main.tex").write_text("\\input{sections/introduction}\n", encoding="utf-8")
    (root / "sections" / "introduction.tex").write_text("v1\n", encoding="utf-8")
    synchronizer = ManuscriptSynchronizer(root)
    project_store = ScholarProjectStore(root)
    first = synchronizer.scan()
    project_store.save_manuscript_state(first)
    next_state = synchronizer.scan(previous=first)
    project_store.save_manuscript_state(next_state, expected_version=first.version)

    stale = synchronizer.scan(previous=first)
    with pytest.raises(ProjectRevisionConflict):
        project_store.save_manuscript_state(stale, expected_version=first.version)


def test_worker_fair_claim_retry_backoff_and_lease_fencing(tmp_path: Path) -> None:
    repository = PersistentJobRepository(tmp_path)
    first, _ = repository.submit(
        "work",
        workspace_id="default",
        scope_version=1,
        idempotency_key="a-1",
        tenant_id="tenant-a",
        principal_id="user-a",
    )
    second, _ = repository.submit(
        "work",
        workspace_id="default",
        scope_version=1,
        idempotency_key="a-2",
        tenant_id="tenant-a",
        principal_id="user-a",
    )
    third, _ = repository.submit(
        "work",
        workspace_id="default",
        scope_version=1,
        idempotency_key="b-1",
        tenant_id="tenant-b",
        principal_id="user-b",
    )
    claimed = repository.claim_next("worker-a", max_running_per_scope=1)
    assert claimed is not None and claimed.job_id == first.job_id
    fair = repository.claim_next("worker-b", max_running_per_scope=1)
    assert fair is not None and fair.job_id == third.job_id

    retry_job, _ = repository.submit(
        "retry",
        workspace_id="default",
        scope_version=1,
        idempotency_key="retry-1",
        max_attempts=2,
    )

    def fail_temporarily(record, context):  # type: ignore[no-untyped-def]
        raise TimeoutError("temporary")

    worker = PersistentJobWorker(
        repository,
        {"retry": fail_temporarily},
        worker_id="retry-worker",
        retry_backoff_seconds=0.02,
        retry_backoff_max_seconds=0.02,
    )
    first_attempt = worker.run_once()
    assert first_attempt is not None and first_attempt.status == "RETRY_PENDING"
    assert first_attempt.available_at is not None
    assert worker.run_once() is None

    # The previous retry attempt is still delayed; recover a separate job to
    # exercise the stale-worker fencing path without waiting for the backoff.
    fence_job, _ = repository.submit(
        "fence",
        workspace_id="default",
        scope_version=1,
        idempotency_key="fence-1",
        max_attempts=2,
    )
    old_lease = repository.claim(fence_job.job_id, "old-worker")
    future = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    repository.mark_interrupted(heartbeat_before=future)
    repository.recover_interrupted()
    new_lease = repository.claim(fence_job.job_id, "new-worker")
    with pytest.raises(KeyError):
        repository.set_status(
            fence_job.job_id,
            "SUCCEEDED",
            worker_id="old-worker",
            lease_token=old_lease.lease_token,
        )
    assert repository.get(fence_job.job_id).lease_token == new_lease.lease_token
    assert [item["status"] for item in repository.list_actions(first.job_id)][:2] == [
        "PLANNED",
        "STARTED",
    ]

    dead, _ = repository.submit(
        "dead",
        workspace_id="default",
        scope_version=1,
        idempotency_key="dead-1",
        max_attempts=1,
    )

    def fail_permanently(record, context):  # type: ignore[no-untyped-def]
        raise RuntimeError("permanent")

    PersistentJobWorker(
        repository,
        {"dead": fail_permanently},
        worker_id="dead-worker",
    ).run_once()
    assert [item.job_id for item in repository.list_dead_letters()] == [dead.job_id]
    assert repository.requeue_failed(dead.job_id, reset_attempts=True).status == "RETRY_PENDING"
    assert second.job_id != fair.job_id
