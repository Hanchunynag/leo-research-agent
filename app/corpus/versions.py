"""Document 内容版本与抽取去重收据；不修改 Canonical Corpus Schema。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage import write_json_atomic


SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class DocumentVersionReceipt:
    version_id: str
    work_id: str
    document_id: str
    content_hash: str
    extraction_profile: str
    version_number: int
    created_at: str


@dataclass(frozen=True, slots=True)
class ExtractionDecision:
    receipt: DocumentVersionReceipt
    requires_extraction: bool


class DocumentVersionRepository:
    """以 ``content_hash + extraction_profile`` 实现幂等抽取判定。"""

    def __init__(self, project_root: Path, *, path: Path | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.path = path or root / "data" / "knowledge" / "document_versions.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": SCHEMA_VERSION, "versions": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Document version schema 不受支持。")
        return payload

    def list(self, work_id: str | None = None) -> tuple[DocumentVersionReceipt, ...]:
        values = self._load()["versions"]
        return tuple(
            DocumentVersionReceipt(**value)
            for value in values
            if work_id is None or value.get("work_id") == work_id
        )

    def register(
        self,
        *,
        work_id: str,
        document_id: str,
        content_hash: str,
        extraction_profile: str,
    ) -> ExtractionDecision:
        if not all((work_id, document_id, content_hash, extraction_profile)):
            raise ValueError("Document version 登记字段不能为空。")
        existing = next(
            (
                value
                for value in self.list()
                if value.content_hash == content_hash
                and value.extraction_profile == extraction_profile
            ),
            None,
        )
        if existing is not None:
            return ExtractionDecision(existing, requires_extraction=False)
        version_number = max(
            (value.version_number for value in self.list(work_id)), default=0
        ) + 1
        receipt = DocumentVersionReceipt(
            version_id=f"{work_id}:v{version_number}",
            work_id=work_id,
            document_id=document_id,
            content_hash=content_hash,
            extraction_profile=extraction_profile,
            version_number=version_number,
            created_at=_now(),
        )
        payload = self._load()
        payload["versions"].append(asdict(receipt))
        write_json_atomic(self.path, payload)
        return ExtractionDecision(receipt, requires_extraction=True)

