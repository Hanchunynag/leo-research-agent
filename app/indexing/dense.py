"""BGE-M3 单向量与 Qdrant local dense 索引构建。"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.indexing.bm25 import chunks_digest
from app.retrieval.search import load_chunks
from app.storage import write_json_atomic


DENSE_INDEX_SCHEMA_VERSION = "1.0"
DENSE_TEXT_POLICY_VERSION = "1.0"
DEFAULT_DENSE_COLLECTION = "leo_paper_chunks_dense"
VECTOR_NAME = "dense"
POINT_NAMESPACE = uuid.UUID("62a7ff24-0bb8-43ef-a793-c193d69d25bf")


@dataclass(frozen=True)
class DenseBuildReport:
    status: str
    collection_name: str
    qdrant_path: str
    manifest_path: str
    chunk_count: int
    embedded_count: int
    vector_dimension: int
    chunks_digest: str
    model_name: str
    model_revision: str | None
    normalized: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dense_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "index" / "qdrant_dense"


def dense_manifest_path(project_root: Path) -> Path:
    return (
        project_root.expanduser().resolve() / "data" / "index" / "dense_manifest.json"
    )


def dense_chunk_text(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title") or "")
    section_path = chunk.get("section_path")
    section = (
        " > ".join(value for value in section_path if isinstance(value, str))
        if isinstance(section_path, list)
        else ""
    )
    parts = [f"Title: {title}", f"Section: {section}"]
    parent_contexts = chunk.get("parent_contexts")
    if isinstance(parent_contexts, list):
        for context in parent_contexts:
            if not isinstance(context, dict):
                continue
            path = context.get("section_path")
            context_section = (
                " > ".join(value for value in path if isinstance(value, str))
                if isinstance(path, list)
                else ""
            )
            parts.append(f"Parent section: {context_section}")
            parts.append(str(context.get("content") or ""))
    overlap = chunk.get("overlap_context")
    if isinstance(overlap, dict) and overlap.get("content"):
        parts.append(f"Previous context: {overlap.get('content')}")
    parts.append(f"Content: {chunk.get('content') or ''}")
    return "\n".join(part for part in parts if part.strip())


def dense_text_sha256(chunk: dict[str, Any]) -> str:
    return hashlib.sha256(dense_chunk_text(chunk).encode("utf-8")).hexdigest()


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def load_dense_manifest(project_root: Path) -> dict[str, Any]:
    path = dense_manifest_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _provider_metadata(provider: EmbeddingProvider) -> tuple[str, str | None, bool]:
    model_name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    normalized = getattr(provider, "normalized", True)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("EmbeddingProvider 必须暴露非空 model_name。")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("EmbeddingProvider revision 必须是字符串或 None。")
    if not isinstance(normalized, bool):
        raise ValueError("EmbeddingProvider normalized 必须是布尔值。")
    return model_name, revision, normalized


def _manifest_matches(
    manifest: dict[str, Any],
    digest: str,
    model_name: str,
    revision: str | None,
    normalized: bool,
    collection_name: str,
) -> bool:
    return all(
        (
            manifest.get("dense_index_schema_version") == DENSE_INDEX_SCHEMA_VERSION,
            manifest.get("dense_text_policy_version") == DENSE_TEXT_POLICY_VERSION,
            manifest.get("chunks_digest") == digest,
            manifest.get("model_name") == model_name,
            manifest.get("model_revision") == revision,
            manifest.get("normalized") is normalized,
            manifest.get("collection_name") == collection_name,
        )
    )


def _safe_remove_index_directory(path: Path, expected_parent: Path) -> None:
    if path.parent.resolve() != expected_parent.resolve():
        raise ValueError(f"拒绝删除意外的 dense index 路径：{path}")
    if path.exists():
        shutil.rmtree(path)


def build_dense_index(
    project_root: Path,
    provider: EmbeddingProvider,
    force: bool = False,
    collection_name: str = DEFAULT_DENSE_COLLECTION,
) -> DenseBuildReport:
    root = project_root.expanduser().resolve()
    chunks = load_chunks(root)
    if not chunks:
        raise ValueError("chunks.jsonl 为空，无法建立 Dense 索引。")
    digest = chunks_digest(chunks)
    model_name, revision, normalized = _provider_metadata(provider)
    index_path = dense_index_path(root)
    manifest_path = dense_manifest_path(root)
    if not force and index_path.is_dir() and manifest_path.is_file():
        manifest = load_dense_manifest(root)
        if _manifest_matches(
            manifest,
            digest,
            model_name,
            revision,
            normalized,
            collection_name,
        ):
            return DenseBuildReport(
                status="reused",
                collection_name=collection_name,
                qdrant_path=str(index_path),
                manifest_path=str(manifest_path),
                chunk_count=len(chunks),
                embedded_count=0,
                vector_dimension=int(manifest.get("vector_dimension", 0)),
                chunks_digest=digest,
                model_name=model_name,
                model_revision=revision,
                normalized=normalized,
            )

    texts = [dense_chunk_text(chunk) for chunk in chunks]
    vectors = provider.embed_documents(texts)
    if len(vectors) != len(chunks) or not vectors:
        raise RuntimeError("EmbeddingProvider 返回的向量数量与 Chunk 不一致。")
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("Dense 向量维度为空或不一致。")

    index_parent = index_path.parent
    index_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_parent / f"{index_path.name}.tmp"
    backup_path = index_parent / f"{index_path.name}.backup"
    _safe_remove_index_directory(temporary_path, index_parent)
    _safe_remove_index_directory(backup_path, index_parent)

    client = QdrantClient(path=str(temporary_path))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            },
            metadata={
                "dense_index_schema_version": DENSE_INDEX_SCHEMA_VERSION,
                "dense_text_policy_version": DENSE_TEXT_POLICY_VERSION,
            },
        )
        for start in range(0, len(chunks), 64):
            points: list[models.PointStruct] = []
            for chunk, vector in zip(
                chunks[start : start + 64],
                vectors[start : start + 64],
                strict=True,
            ):
                chunk_id = chunk.get("chunk_id")
                if not isinstance(chunk_id, str):
                    raise ValueError("Chunk 缺少 chunk_id。")
                payload = {**chunk, "dense_text_sha256": dense_text_sha256(chunk)}
                points.append(
                    models.PointStruct(
                        id=point_id_for_chunk(chunk_id),
                        vector={VECTOR_NAME: vector},
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
    finally:
        client.close()

    manifest = {
        "dense_index_schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "dense_text_policy_version": DENSE_TEXT_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "vector_name": VECTOR_NAME,
        "distance": "Cosine",
        "model_name": model_name,
        "model_revision": revision,
        "normalized": normalized,
        "vector_dimension": dimension,
        "chunk_count": len(chunks),
        "chunks_digest": digest,
        "point_id_policy": "uuid5(chunk_id)",
    }

    swapped = False
    try:
        if index_path.exists():
            index_path.replace(backup_path)
        temporary_path.replace(index_path)
        swapped = True
        write_json_atomic(manifest_path, manifest)
    except Exception:
        if swapped and index_path.exists():
            _safe_remove_index_directory(index_path, index_parent)
        if backup_path.exists():
            backup_path.replace(index_path)
        raise
    finally:
        if temporary_path.exists():
            _safe_remove_index_directory(temporary_path, index_parent)
    _safe_remove_index_directory(backup_path, index_parent)

    return DenseBuildReport(
        status="built",
        collection_name=collection_name,
        qdrant_path=str(index_path),
        manifest_path=str(manifest_path),
        chunk_count=len(chunks),
        embedded_count=len(chunks),
        vector_dimension=dimension,
        chunks_digest=digest,
        model_name=model_name,
        model_revision=revision,
        normalized=normalized,
    )
