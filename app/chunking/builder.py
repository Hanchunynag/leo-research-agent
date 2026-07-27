"""全库结构、Chunk 和 BM25 索引的增量构建入口。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.chunking.chunker import (
    CHUNK_POLICY_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    build_chunks,
    chunk_input_fingerprint,
    document_chunks_path,
    write_document_chunks,
)
from app.chunking.structure import (
    STRUCTURE_POLICY_VERSION,
    build_structure,
    structure_input_fingerprint,
    structure_output_path,
    write_structure,
)
from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.storage import write_json_atomic, write_jsonl_atomic


@dataclass(frozen=True)
class KnowledgeBuildIssue:
    path: str
    error_type: str
    message: str


@dataclass(frozen=True)
class KnowledgeDocumentResult:
    paper_id: str
    document_id: str
    work_id: str
    structure_status: str
    chunk_status: str
    structure_path: str
    chunks_path: str
    section_count: int
    searchable_block_count: int
    chunk_count: int
    absorbed_parent_chunk_count: int
    overlap_context_count: int


@dataclass(frozen=True)
class KnowledgeBuildReport:
    built_at: str
    structure_policy_version: str
    chunk_policy_version: str
    maximum_tokens: int
    minimum_chunk_tokens: int
    overlap_tokens: int
    canonical_count: int
    document_count: int
    structure_built_count: int
    structure_reused_count: int
    chunks_built_count: int
    chunks_reused_count: int
    total_section_count: int
    total_searchable_block_count: int
    total_chunk_count: int
    total_absorbed_parent_chunk_count: int
    total_overlap_context_count: int
    chunks_jsonl: str
    bm25_index: str
    issues: list[KnowledgeBuildIssue]
    documents: list[KnowledgeDocumentResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "issues": [asdict(issue) for issue in self.issues],
            "documents": [asdict(document) for document in self.documents],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return payload


def project_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def chunks_jsonl_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data" / "knowledge" / "chunks.jsonl"


def build_knowledge_base(
    project_root: Path,
    force: bool = False,
    maximum_tokens: int = DEFAULT_MAX_TOKENS,
    minimum_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> KnowledgeBuildReport:
    if maximum_tokens < 50:
        raise ValueError("maximum_tokens 不能小于 50。")
    if minimum_chunk_tokens < 0 or minimum_chunk_tokens >= maximum_tokens:
        raise ValueError("minimum_chunk_tokens 必须大于等于 0 且小于 maximum_tokens。")
    if overlap_tokens < 0 or overlap_tokens >= maximum_tokens:
        raise ValueError("overlap_tokens 必须大于等于 0 且小于 maximum_tokens。")
    root = project_root.expanduser().resolve()
    canonical_root = root / "data" / "canonical"
    canonical_files = (
        sorted(canonical_root.glob("*/paper.json")) if canonical_root.exists() else []
    )
    issues: list[KnowledgeBuildIssue] = []
    document_results: list[KnowledgeDocumentResult] = []
    all_chunks: list[dict[str, Any]] = []

    for canonical_file in canonical_files:
        try:
            canonical = load_json_object(canonical_file)
            identity = canonical.get("identity")
            if not isinstance(identity, dict):
                raise ValueError("canonical identity 缺失。")
            paper_id = canonical.get("paper_id")
            document_id = identity.get("document_id")
            work_id = identity.get("work_id")
            if not all(
                isinstance(value, str) and value
                for value in (paper_id, document_id, work_id)
            ):
                raise ValueError("论文尚未生成可靠的 work/document identity。")
            assert isinstance(paper_id, str)
            assert isinstance(document_id, str)
            assert isinstance(work_id, str)

            expected_structure_fingerprint = structure_input_fingerprint(canonical)
            structure_path = structure_output_path(root, document_id)
            structure_status = "built"
            structure: dict[str, Any]
            if not force and structure_path.is_file():
                existing_structure = load_json_object(structure_path)
                if (
                    existing_structure.get("input_fingerprint")
                    == expected_structure_fingerprint
                    and existing_structure.get("structure_policy_version")
                    == STRUCTURE_POLICY_VERSION
                ):
                    structure = existing_structure
                    structure_status = "reused"
                else:
                    structure = build_structure(canonical)
            else:
                structure = build_structure(canonical)
            if structure_status == "built":
                structure["canonical_path"] = project_relative(canonical_file, root)
                write_structure(root, structure)

            expected_chunk_fingerprint = chunk_input_fingerprint(
                structure,
                maximum_tokens,
                minimum_chunk_tokens,
                overlap_tokens,
            )
            chunks_path = document_chunks_path(root, document_id)
            chunk_status = "built"
            chunk_collection: dict[str, Any]
            if not force and chunks_path.is_file():
                existing_chunks = load_json_object(chunks_path)
                if (
                    existing_chunks.get("input_fingerprint")
                    == expected_chunk_fingerprint
                ):
                    chunk_collection = existing_chunks
                    chunk_status = "reused"
                else:
                    chunk_collection = build_chunks(
                        structure,
                        maximum_tokens,
                        minimum_chunk_tokens,
                        overlap_tokens,
                    )
            else:
                chunk_collection = build_chunks(
                    structure,
                    maximum_tokens,
                    minimum_chunk_tokens,
                    overlap_tokens,
                )
            if chunk_status == "built":
                write_document_chunks(root, chunk_collection)

            raw_chunks = chunk_collection.get("chunks")
            if not isinstance(raw_chunks, list) or not all(
                isinstance(chunk, dict) for chunk in raw_chunks
            ):
                raise ValueError("document chunks 结构无效。")
            all_chunks.extend(raw_chunks)
            document_results.append(
                KnowledgeDocumentResult(
                    paper_id=paper_id,
                    document_id=document_id,
                    work_id=work_id,
                    structure_status=structure_status,
                    chunk_status=chunk_status,
                    structure_path=project_relative(structure_path, root),
                    chunks_path=project_relative(chunks_path, root),
                    section_count=int(structure.get("section_count", 0)),
                    searchable_block_count=int(
                        structure.get("searchable_block_count", 0)
                    ),
                    chunk_count=int(chunk_collection.get("chunk_count", 0)),
                    absorbed_parent_chunk_count=int(
                        chunk_collection.get("absorbed_parent_chunk_count", 0)
                    ),
                    overlap_context_count=int(
                        chunk_collection.get("overlap_context_count", 0)
                    ),
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            issues.append(
                KnowledgeBuildIssue(
                    path=project_relative(canonical_file, root),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    all_chunks.sort(key=lambda chunk: str(chunk.get("chunk_id")))
    chunks_output = chunks_jsonl_path(root)
    write_jsonl_atomic(chunks_output, all_chunks)
    bm25_output = write_bm25_index(root, build_bm25_index(all_chunks))
    report = KnowledgeBuildReport(
        built_at=utc_now_iso(),
        structure_policy_version=STRUCTURE_POLICY_VERSION,
        chunk_policy_version=CHUNK_POLICY_VERSION,
        maximum_tokens=maximum_tokens,
        minimum_chunk_tokens=minimum_chunk_tokens,
        overlap_tokens=overlap_tokens,
        canonical_count=len(canonical_files),
        document_count=len(document_results),
        structure_built_count=sum(
            result.structure_status == "built" for result in document_results
        ),
        structure_reused_count=sum(
            result.structure_status == "reused" for result in document_results
        ),
        chunks_built_count=sum(
            result.chunk_status == "built" for result in document_results
        ),
        chunks_reused_count=sum(
            result.chunk_status == "reused" for result in document_results
        ),
        total_section_count=sum(result.section_count for result in document_results),
        total_searchable_block_count=sum(
            result.searchable_block_count for result in document_results
        ),
        total_chunk_count=len(all_chunks),
        total_absorbed_parent_chunk_count=sum(
            result.absorbed_parent_chunk_count for result in document_results
        ),
        total_overlap_context_count=sum(
            result.overlap_context_count for result in document_results
        ),
        chunks_jsonl=project_relative(chunks_output, root),
        bm25_index=project_relative(bm25_output, root),
        issues=issues,
        documents=document_results,
    )
    write_json_atomic(
        root / "data" / "knowledge" / "last_knowledge_build.json",
        report.to_dict(),
    )
    return report
