"""Paper-level BGE-M3 vector index.

The collection is separate from the existing Chunk collection so that Paper and
Evidence retrieval can never be mixed accidentally.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.embeddings.base import EmbeddingProvider
from app.indexing.paper import load_paper_records, papers_digest, paper_retrieval_text
from app.storage import write_json_atomic


PAPER_DENSE_SCHEMA_VERSION = "1.0"
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


def build_paper_dense_index(
    project_root: Path,
    provider: EmbeddingProvider,
    *,
    force: bool = False,
    collection_name: str = DEFAULT_PAPER_DENSE_COLLECTION,
) -> PaperDenseBuildReport:
    root = project_root.expanduser().resolve()
    papers = load_paper_records(root)
    if not papers:
        raise ValueError("没有 Paper metadata，无法建立 Paper-level Dense 索引。")
    digest = papers_digest(papers)
    model_name = getattr(provider, "model_name", None)
    revision = getattr(provider, "revision", None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("EmbeddingProvider 必须暴露非空 model_name。")
    index_path = paper_dense_index_path(root)
    manifest_path = paper_dense_manifest_path(root)
    if not force and index_path.is_dir() and manifest_path.is_file():
        manifest = load_paper_dense_manifest(root)
        if all(
            (
                manifest.get("paper_dense_schema_version") == PAPER_DENSE_SCHEMA_VERSION,
                manifest.get("paper_dense_text_policy_version") == PAPER_DENSE_TEXT_POLICY_VERSION,
                manifest.get("papers_digest") == digest,
                manifest.get("model_name") == model_name,
                manifest.get("model_revision") == revision,
                manifest.get("collection_name") == collection_name,
            )
        ):
            return PaperDenseBuildReport(
                "reused", collection_name, str(index_path), str(manifest_path),
                len(papers), 0, int(manifest.get("vector_dimension", 0)), digest,
                model_name, revision,
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

    client = QdrantClient(path=str(temporary_path))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={PAPER_VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            metadata={
                "paper_dense_schema_version": PAPER_DENSE_SCHEMA_VERSION,
                "paper_dense_text_policy_version": PAPER_DENSE_TEXT_POLICY_VERSION,
            },
        )
        points = [
            models.PointStruct(
                id=paper_point_id(str(paper["paper_id"])),
                vector={PAPER_VECTOR_NAME: vector},
                payload=dict(paper),
            )
            for paper, vector in zip(papers, vectors, strict=True)
        ]
        for start in range(0, len(points), 64):
            client.upsert(collection_name=collection_name, points=points[start : start + 64], wait=True)
    finally:
        client.close()

    manifest = {
        "paper_dense_schema_version": PAPER_DENSE_SCHEMA_VERSION,
        "paper_dense_text_policy_version": PAPER_DENSE_TEXT_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "vector_name": PAPER_VECTOR_NAME,
        "distance": "Cosine",
        "model_name": model_name,
        "model_revision": revision,
        "vector_dimension": dimension,
        "paper_count": len(papers),
        "papers_digest": digest,
        "point_id_policy": "uuid5(paper_id)",
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
            shutil.rmtree(index_path)
        if backup_path.exists():
            backup_path.replace(index_path)
        raise
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)
    return PaperDenseBuildReport(
        "built", collection_name, str(index_path), str(manifest_path), len(papers),
        len(papers), dimension, digest, model_name, revision,
    )
