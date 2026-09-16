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

from app.chunking.chunker import CHUNK_POLICY_VERSION
from app.embeddings.base import EmbeddingProvider
from app.indexing.bm25 import chunks_digest
from app.indexing.tokenization import TOKENIZER_VERSION
from app.retrieval.search import load_chunks
from app.qdrant import build_qdrant_client, qdrant_is_remote, qdrant_storage_description
from app.storage import write_json_atomic


DENSE_INDEX_SCHEMA_VERSION = "1.2"
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
    model_artifact_fingerprint: str | None
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


def point_id_for_chunk(paper_id: str, chunk_id: str) -> str:
    """Return a stable physical ID for a paper-owned chunk.

    ``chunk_id`` is scoped to a paper.  It must therefore never be used alone
    as a Qdrant point ID: two papers are allowed to both contain ``C001``.
    """

    return str(uuid.uuid5(POINT_NAMESPACE, f"{paper_id}:{chunk_id}"))


def load_dense_manifest(project_root: Path) -> dict[str, Any]:
    path = dense_manifest_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _provider_metadata(
    provider: EmbeddingProvider,
) -> tuple[str, str | None, str | None, bool]:
    model_name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    artifact_fingerprint = getattr(provider, "artifact_fingerprint", None)
    normalized = getattr(provider, "normalized", True)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("EmbeddingProvider 必须暴露非空 model_name。")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("EmbeddingProvider revision 必须是字符串或 None。")
    if artifact_fingerprint is not None and not isinstance(artifact_fingerprint, str):
        raise ValueError(
            "EmbeddingProvider artifact_fingerprint 必须是字符串或 None。"
        )
    if revision is None and not artifact_fingerprint:
        raise ValueError(
            "EmbeddingProvider 必须提供精确 revision 或稳定 artifact_fingerprint。"
        )
    if not isinstance(normalized, bool):
        raise ValueError("EmbeddingProvider normalized 必须是布尔值。")
    return model_name, revision, artifact_fingerprint, normalized


def _manifest_matches(
    manifest: dict[str, Any],
    digest: str,
    model_name: str,
    revision: str | None,
    artifact_fingerprint: str | None,
    normalized: bool,
    collection_name: str,
) -> bool:
    return all(
        (
            manifest.get("dense_index_schema_version") == DENSE_INDEX_SCHEMA_VERSION,
            manifest.get("dense_text_policy_version") == DENSE_TEXT_POLICY_VERSION,
            manifest.get("chunk_policy_version") == CHUNK_POLICY_VERSION,
            manifest.get("tokenizer_version") == TOKENIZER_VERSION,
            manifest.get("chunks_digest") == digest,
            manifest.get("model_name") == model_name,
            manifest.get("model_revision") == revision,
            manifest.get("model_artifact_fingerprint") == artifact_fingerprint,
            manifest.get("normalized") is normalized,
            manifest.get("collection_name") == collection_name,
        )
    )


def _paper_chunk_digests(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Fingerprint each paper's embedding inputs independently."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        paper_id = str(chunk.get("paper_id") or "")
        if paper_id:
            grouped.setdefault(paper_id, []).append(chunk)
    digests: dict[str, str] = {}
    for paper_id, values in grouped.items():
        payload = [
            {
                "chunk_id": value.get("chunk_id"),
                "level": value.get("level", 2),
                "content_hash": value.get("content_hash")
                or hashlib.sha256(str(value.get("content") or "").encode("utf-8")).hexdigest(),
                "dense_text_hash": dense_text_sha256(value),
            }
            for value in sorted(values, key=lambda item: str(item.get("chunk_id") or ""))
        ]
        digests[paper_id] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return digests


def _delete_paper_points(client: QdrantClient, collection: str, paper_ids: set[str]) -> int:
    deleted = 0
    for paper_id in sorted(paper_ids):
        while True:
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=paper_id)
                )]),
                limit=256,
                # Delete every returned page immediately. Reusing a cursor
                # after deletion can skip points in a changed filtered set.
                offset=None,
                with_payload=False,
                with_vectors=False,
            )
            ids = [point.id for point in points]
            if ids:
                client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=ids),
                    wait=True,
                )
                deleted += len(ids)
            if not points:
                break
    return deleted


def _incremental_update(
    root: Path,
    provider: EmbeddingProvider,
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    changed_paper_ids: set[str],
    collection_name: str,
    model_name: str,
    revision: str | None,
    artifact_fingerprint: str | None,
    normalized: bool,
) -> DenseBuildReport:
    index_path = dense_index_path(root)
    current_digests = _paper_chunk_digests(chunks)
    previous_digests = manifest.get("paper_digests")
    if not isinstance(previous_digests, dict):
        raise RuntimeError("Dense manifest 缺少按论文记录，必须执行一次管理员 rebuild。")
    changed = {
        paper_id
        for paper_id in set(map(str, previous_digests)) | set(current_digests) | changed_paper_ids
        if previous_digests.get(paper_id) != current_digests.get(paper_id)
        or paper_id in changed_paper_ids
    }
    client = build_qdrant_client(index_path)
    try:
        existing = {value.name for value in client.get_collections().collections}
        if collection_name not in existing:
            raise RuntimeError("Dense collection 不存在，必须执行一次管理员 rebuild。")
        _delete_paper_points(client, collection_name, changed)
        changed_chunks = [
            chunk for chunk in chunks if str(chunk.get("paper_id") or "") in changed
        ]
        embedded_count = 0
        if changed_chunks:
            vectors = provider.embed_documents([dense_chunk_text(chunk) for chunk in changed_chunks])
            if len(vectors) != len(changed_chunks):
                raise RuntimeError("EmbeddingProvider 返回的按论文向量数量不一致。")
            points = []
            for chunk, vector in zip(changed_chunks, vectors, strict=True):
                points.append(models.PointStruct(
                    id=point_id_for_chunk(
                        str(chunk["paper_id"]), str(chunk["chunk_id"])
                    ),
                    vector={VECTOR_NAME: vector},
                    payload={
                        **dict(chunk),
                        "level": 2,
                        "content_hash": chunk.get("content_hash") or hashlib.sha256(
                            str(chunk.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                        "dense_text_sha256": dense_text_sha256(chunk),
                        "embedding_revision": revision,
                        "embedding_model": model_name,
                    },
                ))
            dimension = len(vectors[0]) if vectors else int(manifest.get("vector_dimension", 0))
            if not dimension or any(len(vector) != dimension for vector in vectors):
                raise RuntimeError("Dense 向量维度为空或不一致。")
            for start in range(0, len(points), 64):
                client.upsert(collection_name=collection_name, points=points[start : start + 64], wait=True)
            embedded_count = len(changed_chunks)
        else:
            dimension = int(manifest.get("vector_dimension", 0))
    finally:
        client.close()

    digest = chunks_digest(chunks)
    updated_manifest = {
        **manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_count": len(current_digests),
        "chunk_count": len(chunks),
        "chunks_digest": digest,
        "paper_digests": current_digests,
        "model_name": model_name,
        "model_revision": revision,
        "model_artifact_fingerprint": artifact_fingerprint,
        "normalized": normalized,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "index_epoch": f"CH_{digest[:16]}",
    }
    write_json_atomic(dense_manifest_path(root), updated_manifest)
    return DenseBuildReport(
        status="incremental" if changed else "reused",
        collection_name=collection_name,
        qdrant_path=qdrant_storage_description(index_path),
        manifest_path=str(dense_manifest_path(root)),
        chunk_count=len(chunks),
        embedded_count=embedded_count,
        vector_dimension=dimension,
        chunks_digest=digest,
        model_name=model_name,
        model_revision=revision,
        model_artifact_fingerprint=artifact_fingerprint,
        normalized=normalized,
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
    changed_paper_ids: set[str] | None = None,
) -> DenseBuildReport:
    root = project_root.expanduser().resolve()
    chunks = load_chunks(root)
    if not chunks:
        raise ValueError("chunks.jsonl 为空，无法建立 Dense 索引。")
    digest = chunks_digest(chunks)
    model_name, revision, artifact_fingerprint, normalized = _provider_metadata(provider)
    index_path = dense_index_path(root)
    manifest_path = dense_manifest_path(root)
    if not force and manifest_path.is_file():
        manifest = load_dense_manifest(root)
        if not isinstance(manifest.get("paper_digests"), dict):
            raise RuntimeError(
                "Dense manifest 缺少按论文 paper_digests；请管理员显式执行 force rebuild。"
            )
        if not qdrant_is_remote() and not index_path.is_dir():
            raise RuntimeError(
                "Dense collection 不存在；请管理员显式执行 force rebuild。"
            )
        if _manifest_matches(
            manifest,
            digest if manifest.get("chunks_digest") == digest else str(manifest.get("chunks_digest")),
            model_name,
            revision,
            artifact_fingerprint,
            normalized,
            collection_name,
        ):
            return _incremental_update(
                root,
                provider,
                chunks,
                manifest,
                changed_paper_ids=set(changed_paper_ids or ()),
                collection_name=collection_name,
                model_name=model_name,
                revision=revision,
                artifact_fingerprint=artifact_fingerprint,
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

    if qdrant_is_remote():
        # The Compose API and Worker share one Qdrant service. Recreate only
        # this collection after all vectors have been computed; the embedded
        # directory is never opened in production, so concurrent retrievals
        # cannot contend on Qdrant's single-process storage lock.
        client = build_qdrant_client(index_path)
        try:
            existing = {value.name for value in client.get_collections().collections}
            if collection_name in existing:
                client.delete_collection(collection_name=collection_name)
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
                    payload = {
                        **chunk,
                        "level": 2,
                        "content_hash": chunk.get("content_hash") or hashlib.sha256(
                            str(chunk.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                        "dense_text_sha256": dense_text_sha256(chunk),
                        "embedding_revision": revision,
                        "embedding_model": model_name,
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id_for_chunk(str(chunk["paper_id"]), chunk_id),
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
        # Keep a harmless local marker for the existing readiness projection;
        # vector data itself lives in the shared Qdrant service.
        index_path.mkdir(parents=True, exist_ok=True)
    else:
        temporary_path = index_parent / f"{index_path.name}.tmp"
        client = build_qdrant_client(temporary_path)
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
                    payload = {
                        **chunk,
                        "level": 2,
                        "content_hash": chunk.get("content_hash") or hashlib.sha256(
                            str(chunk.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                        "dense_text_sha256": dense_text_sha256(chunk),
                        "embedding_revision": revision,
                        "embedding_model": model_name,
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id_for_chunk(str(chunk["paper_id"]), chunk_id),
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

        swapped = False
        try:
            if index_path.exists():
                index_path.replace(backup_path)
            temporary_path.replace(index_path)
            swapped = True
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

    manifest = {
        "dense_index_schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "dense_text_policy_version": DENSE_TEXT_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "vector_name": VECTOR_NAME,
        "distance": "Cosine",
        "model_name": model_name,
        "model_revision": revision,
        "model_artifact_fingerprint": artifact_fingerprint,
        "normalized": normalized,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "vector_dimension": dimension,
        "chunk_count": len(chunks),
        "chunks_digest": digest,
        "paper_digests": _paper_chunk_digests(chunks),
        "point_id_policy": "uuid5(paper_id:chunk_id)",
        "index_epoch": f"CH_{digest[:16]}",
        "tokenizer_version": TOKENIZER_VERSION,
        "chunker_version": "app.chunking.chunker.v2.2",
    }

    write_json_atomic(manifest_path, manifest)

    return DenseBuildReport(
        status="built",
        collection_name=collection_name,
        qdrant_path=qdrant_storage_description(index_path),
        manifest_path=str(manifest_path),
        chunk_count=len(chunks),
        embedded_count=len(chunks),
        vector_dimension=dimension,
        chunks_digest=digest,
        model_name=model_name,
        model_revision=revision,
        model_artifact_fingerprint=artifact_fingerprint,
        normalized=normalized,
    )
