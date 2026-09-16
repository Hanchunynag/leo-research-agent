"""Paper-level BGE-M3 vector index.

The collection is separate from the existing Chunk collection so that Paper and
Evidence retrieval can never be mixed accidentally.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import models

from app.chunking.chunker import CHUNK_POLICY_VERSION
from app.embeddings.base import EmbeddingProvider
from app.indexing.paper import load_paper_records, papers_digest, paper_retrieval_text
from app.indexing.tokenization import TOKENIZER_VERSION
from app.qdrant import build_qdrant_client, qdrant_is_remote, qdrant_storage_description
from app.storage import write_json_atomic


PAPER_DENSE_SCHEMA_VERSION = "1.2"
PAPER_DENSE_TEXT_POLICY_VERSION = "1.0"
DEFAULT_PAPER_DENSE_COLLECTION = "leo_papers_dense"
PAPER_VECTOR_NAME = "dense"
PAPER_POINT_NAMESPACE = uuid.UUID("2d47c1db-11ae-4e1d-9cc2-8d1bbf7e6a18")


@dataclass(frozen=True)
class PaperDenseBuildReport:
    status: str
    collection_name: str
    qdrant_path: str
    manifest_path: str
    paper_count: int
    embedded_count: int
    vector_dimension: int
    papers_digest: str
    model_name: str
    model_revision: str | None
    model_artifact_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paper_dense_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "index" / "qdrant_papers_dense"


def paper_dense_manifest_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "index" / "paper_dense_manifest.json"


def paper_point_id(paper_id: str) -> str:
    return str(uuid.uuid5(PAPER_POINT_NAMESPACE, paper_id))


def load_paper_dense_manifest(project_root: Path) -> dict[str, Any]:
    path = paper_dense_manifest_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _normalized_papers(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize the one Level-1 metadata vector owned by each paper."""

    output: list[dict[str, Any]] = []
    for value in values:
        paper_id = str(value.get("paper_id") or "")
        if not paper_id:
            continue
        text = paper_retrieval_text(value)
        output.append({
            **dict(value),
            "paper_id": paper_id,
            "content": text,
            "level": 1,
            "chunk_id": f"{paper_id}_metadata",
            "section_path": [],
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return output


def _paper_digests(papers: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(paper["paper_id"]): hashlib.sha256(
            json.dumps(
                {
                    "paper_id": paper.get("paper_id"),
                    "level": 1,
                    "content_hash": paper.get("content_hash"),
                    "text": paper_retrieval_text(paper),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for paper in papers
    }


def _paper_point(
    paper: dict[str, Any],
    vector: list[float],
    *,
    model_name: str,
    revision: str | None,
) -> models.PointStruct:
    return models.PointStruct(
        id=paper_point_id(str(paper["paper_id"])),
        vector={PAPER_VECTOR_NAME: vector},
        payload={
            **dict(paper),
            "level": 1,
            "chunk_id": f"{paper['paper_id']}_metadata",
            "section_path": [],
            "content_hash": paper.get("content_hash") or hashlib.sha256(
                paper_retrieval_text(paper).encode("utf-8")
            ).hexdigest(),
            "embedding_model": model_name,
            "embedding_revision": revision,
        },
    )


def build_paper_dense_index(
    project_root: Path,
    provider: EmbeddingProvider,
    *,
    force: bool = False,
    collection_name: str = DEFAULT_PAPER_DENSE_COLLECTION,
    changed_paper_ids: set[str] | None = None,
) -> PaperDenseBuildReport:
    """Build the global Paper-level semantic space.

    ``changed_paper_ids`` is retained for the shared indexing API, but it is
    intentionally not used to select a subset here.  Whenever the active
    Paper metadata corpus changes, every Paper vector is regenerated together;
    only an unchanged corpus can reuse its existing vectors.
    """

    root = project_root.expanduser().resolve()
    papers = _normalized_papers(load_paper_records(root))
    if not papers:
        raise ValueError("没有 Paper metadata，无法建立 Paper-level Dense 索引。")
    digest = papers_digest(papers)
    current_paper_digests = _paper_digests(papers)
    model_name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    artifact_fingerprint = getattr(provider, "artifact_fingerprint", None)
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
    index_path = paper_dense_index_path(root)
    manifest_path = paper_dense_manifest_path(root)
    if not force and manifest_path.is_file():
        manifest = load_paper_dense_manifest(root)
        if not isinstance(manifest.get("paper_digests"), dict):
            raise RuntimeError(
                "Paper Dense manifest 缺少按论文 paper_digests；请管理员显式执行 force rebuild。"
            )
        if not qdrant_is_remote() and not index_path.is_dir():
            raise RuntimeError(
                "Paper Dense collection 不存在；请管理员显式执行 force rebuild。"
            )
        if all(
            (
                manifest.get("paper_dense_schema_version") == PAPER_DENSE_SCHEMA_VERSION,
                manifest.get("paper_dense_text_policy_version") == PAPER_DENSE_TEXT_POLICY_VERSION,
                manifest.get("chunk_policy_version") == CHUNK_POLICY_VERSION,
                manifest.get("tokenizer_version") == TOKENIZER_VERSION,
                manifest.get("model_name") == model_name,
                manifest.get("model_revision") == revision,
                manifest.get("model_artifact_fingerprint") == artifact_fingerprint,
                manifest.get("collection_name") == collection_name,
                # Level-1 is the global Paper semantic space.  A changed
                # metadata corpus (including a newly added or removed paper)
                # must therefore regenerate every Paper vector together.
                manifest.get("papers_digest") == digest,
                manifest.get("paper_digests") == current_paper_digests,
            )
        ):
            return PaperDenseBuildReport(
                "reused",
                collection_name,
                str(index_path),
                str(manifest_path),
                len(papers),
                0,
                int(manifest.get("vector_dimension", 0)),
                digest,
                model_name,
                revision,
                artifact_fingerprint,
            )

    vectors = provider.embed_documents([paper_retrieval_text(value) for value in papers])
    if len(vectors) != len(papers) or not vectors:
        raise RuntimeError("EmbeddingProvider 返回的 Paper 向量数量不一致。")
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("Paper Dense 向量维度为空或不一致。")

    index_parent = index_path.parent
    index_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_parent / f"{index_path.name}.tmp"
    backup_path = index_parent / f"{index_path.name}.backup"
    for path in (temporary_path, backup_path):
        if path.exists():
            if path.parent.resolve() != index_parent.resolve():
                raise ValueError(f"拒绝删除意外的 Paper dense 路径：{path}")
            shutil.rmtree(path)

    points = [
        _paper_point(paper, vector, model_name=model_name, revision=revision)
        for paper, vector in zip(papers, vectors, strict=True)
    ]
    if qdrant_is_remote():
        # Compose uses a shared Qdrant server so API and Worker can query the
        # global Paper-level collection concurrently.  Replacing this one
        # collection does not touch the independent Level-2 collection.
        client = build_qdrant_client(index_path)
        try:
            existing = {value.name for value in client.get_collections().collections}
            if collection_name in existing:
                client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={PAPER_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
                metadata={
                    "paper_dense_schema_version": PAPER_DENSE_SCHEMA_VERSION,
                    "paper_dense_text_policy_version": PAPER_DENSE_TEXT_POLICY_VERSION,
                },
            )
            for start in range(0, len(points), 64):
                client.upsert(collection_name=collection_name, points=points[start : start + 64], wait=True)
        finally:
            client.close()
        index_path.mkdir(parents=True, exist_ok=True)
    else:
        client = build_qdrant_client(temporary_path)
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={PAPER_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
                metadata={
                    "paper_dense_schema_version": PAPER_DENSE_SCHEMA_VERSION,
                    "paper_dense_text_policy_version": PAPER_DENSE_TEXT_POLICY_VERSION,
                },
            )
            for start in range(0, len(points), 64):
                client.upsert(collection_name=collection_name, points=points[start : start + 64], wait=True)
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
                shutil.rmtree(index_path)
            if backup_path.exists():
                backup_path.replace(index_path)
            raise
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)

    manifest = {
        "paper_dense_schema_version": PAPER_DENSE_SCHEMA_VERSION,
        "paper_dense_text_policy_version": PAPER_DENSE_TEXT_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "vector_name": PAPER_VECTOR_NAME,
        "distance": "Cosine",
        "model_name": model_name,
        "model_revision": revision,
        "model_artifact_fingerprint": artifact_fingerprint,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "vector_dimension": dimension,
        "paper_count": len(papers),
        "papers_digest": digest,
        "paper_digests": _paper_digests(papers),
        "point_id_policy": "uuid5(paper_id)",
        "index_epoch": f"PA_{digest[:16]}",
        "tokenizer_version": TOKENIZER_VERSION,
        "chunker_version": "app.chunking.chunker.v2.2",
    }
    write_json_atomic(manifest_path, manifest)
    return PaperDenseBuildReport(
        "built", collection_name, qdrant_storage_description(index_path), str(manifest_path), len(papers),
        len(papers), dimension, digest, model_name, revision, artifact_fingerprint,
    )
