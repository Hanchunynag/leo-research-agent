"""现有 JSON/JSONL canonical 产物上的 Repository。

这些 Repository 不创建平行数据库，也不修改阶段一已经冻结的 Schema。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.contracts import Document


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
        values.append(value)
    return values


class DocumentRepository:
    """以 ``data/canonical/*/paper.json`` 为文档事实源。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.canonical_root = self.project_root / "data" / "canonical"

    def _records(self) -> list[tuple[Path, dict[str, Any]]]:
        if not self.canonical_root.exists():
            return []
        return [(path, _object(path)) for path in sorted(self.canonical_root.glob("*/paper.json"))]

    def list(self) -> tuple[Document, ...]:
        return tuple(self._to_domain(path, value) for path, value in self._records())

    def get(self, document_id: str) -> Document | None:
        for path, value in self._records():
            identity = value.get("identity")
            if isinstance(identity, dict) and identity.get("document_id") == document_id:
                return self._to_domain(path, value)
        return None

    def canonical(self, document_id: str) -> dict[str, Any] | None:
        for _, value in self._records():
            identity = value.get("identity")
            if isinstance(identity, dict) and identity.get("document_id") == document_id:
                return value
        return None

    def find_by_content_hash(self, content_hash: str) -> Document | None:
        """用原 PDF SHA-256 阻止同内容被重复解析。"""

        for document in self.list():
            if document.source_sha256 == content_hash:
                return document
        return None

    def _to_domain(self, path: Path, value: dict[str, Any]) -> Document:
        identity = value.get("identity")
        source = value.get("source")
        if not isinstance(identity, dict) or not isinstance(source, dict):
            raise ValueError(f"{path} 缺少 identity 或 source。")
        document_id = str(identity.get("document_id") or "")
        sections_path = self.project_root / "data" / "knowledge" / "structures" / f"{document_id}.structure.json"
        chunks_path = self.project_root / "data" / "knowledge" / "chunks" / f"{document_id}.chunks.json"
        sections = _object(sections_path).get("sections", []) if sections_path.is_file() else []
        chunks = _object(chunks_path).get("chunks", []) if chunks_path.is_file() else []
        raw_blocks = value.get("blocks")
        blocks: list[Any] = raw_blocks if isinstance(raw_blocks, list) else []
        raw_metadata = value.get("metadata")
        metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        return Document(
            document_id=document_id,
            paper_id=str(value.get("paper_id") or ""),
            work_id=str(identity.get("work_id") or "") or None,
            source_sha256=str(source.get("sha256") or ""),
            canonical_path=path.relative_to(self.project_root).as_posix(),
            section_ids=tuple(str(item["section_id"]) for item in sections if isinstance(item, dict) and item.get("section_id")),
            chunk_ids=tuple(str(item["chunk_id"]) for item in chunks if isinstance(item, dict) and item.get("chunk_id")),
            block_ids=tuple(str(item["block_id"]) for item in blocks if isinstance(item, dict) and item.get("block_id")),
            metadata=metadata,
        )


class SectionRepository:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.expanduser().resolve() / "data" / "knowledge" / "structures"

    def list_for_document(self, document_id: str) -> tuple[dict[str, Any], ...]:
        path = self.root / f"{document_id}.structure.json"
        if not path.is_file():
            return ()
        values = _object(path).get("sections")
        return tuple(dict(value) for value in values if isinstance(value, dict)) if isinstance(values, list) else ()

    def get(self, section_id: str) -> dict[str, Any] | None:
        document_id = section_id.split("_s", 1)[0]
        return next((value for value in self.list_for_document(document_id) if value.get("section_id") == section_id), None)


class ChunkRepository:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.path = self.project_root / "data" / "knowledge" / "chunks.jsonl"

    def list(self) -> tuple[dict[str, Any], ...]:
        return tuple(_jsonl(self.path))

    def list_for_document(self, document_id: str) -> tuple[dict[str, Any], ...]:
        document_path = self.path.parent / "chunks" / f"{document_id}.chunks.json"
        if document_path.is_file():
            values = _object(document_path).get("chunks")
            if isinstance(values, list):
                return tuple(dict(value) for value in values if isinstance(value, dict))
        return tuple(value for value in self.list() if value.get("document_id") == document_id)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return next((value for value in self.list() if value.get("chunk_id") == chunk_id), None)


class CitationRepository:
    """Canonical 引用块视图；当前解析器尚未产出独立引用边 Schema。"""

    def __init__(self, project_root: Path, documents: DocumentRepository | None = None) -> None:
        self.documents = documents or DocumentRepository(project_root)

    def list_for_document(self, document_id: str) -> tuple[dict[str, Any], ...]:
        canonical = self.documents.canonical(document_id)
        if canonical is None:
            return ()
        blocks = canonical.get("blocks")
        if not isinstance(blocks, list):
            return ()
        # 独立 Citation 尚不存在；只暴露引用区原始 Block，避免虚构 citation edge。
        return tuple(
            {
                "document_id": document_id,
                "block_id": value.get("block_id"),
                "page_number": value.get("page_number"),
                "content": value.get("text") or value.get("content") or "",
                "status": "raw_reference_block",
            }
            for value in blocks
            if isinstance(value, dict)
            and (
                value.get("content_zone") == "references"
                or str(value.get("text") or "").strip().casefold() in {"references", "bibliography"}
            )
        )
