"""Tool Gateway 的长任务提交适配器；不在 Harness 调用栈内执行任务。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from app.jobs.repository import PersistentJobRepository


LONG_RUNNING_JOB_TYPES = frozenset(
    {
        "literature.download",
        "document.parse",
        "knowledge.index",
        "index_generation.build",
        "document.update",
        "document.delete",
    }
)


class LongTaskSubmitter:
    def __init__(self, repository: PersistentJobRepository, structured_repository: Any | None = None) -> None:
        self.repository = repository
        self.structured_repository = structured_repository

    @staticmethod
    def _key(
        job_type: str, arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> str:
        identity = {
            "job_type": job_type,
            "workspace_id": context.get("workspace_id"),
            "scope_version": context.get("scope_version"),
            "document_id": arguments.get("document_id"),
            "generation_id": arguments.get("generation_id"),
            "profile_id": arguments.get("profile_id"),
            "paper_id": arguments.get("paper_id"),
            "path": arguments.get("path"),
            "mode": arguments.get("mode"),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"{job_type}:{digest}"

    def handler(self, job_type: str):  # type: ignore[no-untyped-def]
        if job_type not in LONG_RUNNING_JOB_TYPES:
            raise ValueError(f"不是已声明的长任务：{job_type}")

        def submit(
            arguments: Mapping[str, Any], context: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            workspace_id = str(context.get("workspace_id") or "")
            scope_version = int(context.get("scope_version") or 0)
            record, created = self.repository.submit(
                job_type,
                workspace_id=workspace_id,
                scope_version=scope_version,
                document_id=(
                    str(arguments["document_id"])
                    if arguments.get("document_id")
                    else None
                ),
                generation_id=(
                    str(arguments["generation_id"])
                    if arguments.get("generation_id")
                    else None
                ),
                payload=dict(arguments),
                idempotency_key=self._key(job_type, arguments, context),
            )
            if self.structured_repository is not None:
                try:
                    self.structured_repository.record_job({
                        "job_id": record.job_id,
                        "job_type": job_type,
                        "paper_id": arguments.get("paper_id"),
                        "document_id": arguments.get("document_id"),
                        "status": record.status.casefold(),
                        "payload": dict(arguments),
                    })
                except Exception:
                    pass
            return {
                "job_id": record.job_id,
                "status": record.status.casefold(),
                "created": created,
            }

        return submit
