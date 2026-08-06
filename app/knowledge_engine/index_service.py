"""新增、更新、删除文档的唯一知识索引编排入口。"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from app.contracts import Document, IndexGeneration, IndexProfile, KnowledgeEngine
from app.knowledge_engine.generations import IndexGenerationRepository
from app.storage import write_json_atomic


Validation = Callable[[IndexGeneration, Mapping[str, Any]], tuple[bool, str]]


class KnowledgeIndexService:
    def __init__(
        self,
        engine: KnowledgeEngine | None = None,
        generations: IndexGenerationRepository | None = None,
        validator: Validation | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.engine = engine
        self.generations = generations
        self.validator = validator or self._default_validator
        self.project_root = project_root.expanduser().resolve() if project_root else None

    def _engine(self) -> KnowledgeEngine:
        if self.engine is None:
            raise RuntimeError("KnowledgeIndexService 未配置 KnowledgeEngine。")
        return self.engine

    def _generations(self) -> IndexGenerationRepository:
        if self.generations is None:
            raise RuntimeError("KnowledgeIndexService 未配置 IndexGenerationRepository。")
        return self.generations

    @staticmethod
    def _default_validator(generation: IndexGeneration, metrics: Mapping[str, Any]) -> tuple[bool, str]:
        indexed = int(metrics.get("indexed_document_count", 0))
        mapped = int(metrics.get("mapped_chunk_count", 0))
        if indexed < generation.document_count:
            return False, "indexed_document_count 小于 generation.document_count"
        if generation.chunk_count and mapped < generation.chunk_count:
            return False, "LightRAG Chunk 未全部映射回 Canonical Corpus"
        return True, "validated"

    def build_generation(self, documents: Sequence[Document], *, workspace_id: str, scope_version: int, corpus_version: str, profile: IndexProfile, activate: bool = False) -> tuple[IndexGeneration, Mapping[str, Any]]:
        generations = self._generations()
        engine = self._engine()
        previous = generations.active(workspace_id)
        generation = IndexGeneration(
            generation_id=f"IG_{secrets.token_hex(8)}",
            corpus_version=corpus_version,
            scope_version=scope_version,
            state="pending",
            document_count=len(documents),
            chunk_count=sum(len(value.chunk_ids) for value in documents),
            previous_generation_id=previous.generation_id if previous else None,
            workspace_id=workspace_id,
            index_profile_id=profile.profile_id,
        )
        generation = generations.create(generation)
        try:
            metrics = engine.index_documents(documents, generation=generation, profile=profile)
            generation = generations.transition(generation.generation_id, "validating")
            valid, reason = self.validator(generation, metrics)
            if not valid:
                raise RuntimeError(reason)
            if activate:
                generation = generations.activate(generation.generation_id)
            return generation, metrics
        except BaseException as error:
            generations.fail(generation.generation_id, error)
            raise

    def update_documents(self, generation_id: str, documents: Sequence[Document], profile: IndexProfile) -> Mapping[str, Any]:
        generation = self._generations().get(generation_id)
        if generation is None:
            raise KeyError(generation_id)
        return self._engine().update_documents(documents, generation=generation, profile=profile)

    def delete_documents(self, generation_id: str, document_ids: Sequence[str]) -> Mapping[str, Any]:
        generation = self._generations().get(generation_id)
        if generation is None:
            raise KeyError(generation_id)
        return self._engine().delete_documents(document_ids, generation=generation)

    def rollback(self, workspace_id: str, generation_id: str) -> IndexGeneration:
        generations = self._generations()
        target = generations.get(generation_id)
        if target is None or target.workspace_id != workspace_id:
            raise KeyError(generation_id)
        return generations.activate(generation_id)

    def build_incremental_generation(
        self,
        *,
        operation: Literal["add", "update", "delete"],
        workspace_id: str,
        scope_version: int,
        corpus_version: str,
        profile: IndexProfile,
        documents: Sequence[Document] = (),
        document_ids: Sequence[str] = (),
        activate: bool = False,
    ) -> tuple[IndexGeneration, Mapping[str, Any]]:
        """从 active 快照分叉，只处理变更文档，不重新抽取全库。"""

        generations = self._generations()
        engine = self._engine()
        previous = generations.active(workspace_id)
        if previous is None:
            if operation == "delete":
                raise RuntimeError("没有 active generation，不能执行增量删除。")
            return self.build_generation(
                documents,
                workspace_id=workspace_id,
                scope_version=scope_version,
                corpus_version=corpus_version,
                profile=profile,
                activate=activate,
            )
        changed_ids = set(document_ids) | {value.document_id for value in documents}
        ordered_changed_ids = tuple(sorted(changed_ids))
        raw_previous_mapping: Any = getattr(
            engine, "_load_mapping", lambda value: {}
        )(
            previous
        )
        previous_mapping: Mapping[str, Mapping[str, Any]] = (
            raw_previous_mapping
            if isinstance(raw_previous_mapping, Mapping)
            else {}
        )
        previous_docs = {str(value.get("document_id")) for value in previous_mapping.values()}
        previous_chunks_by_doc: dict[str, int] = {}
        for value in previous_mapping.values():
            document_id = str(value.get("document_id"))
            previous_chunks_by_doc[document_id] = previous_chunks_by_doc.get(document_id, 0) + 1
        if operation == "delete":
            next_document_count = max(0, previous.document_count - len(previous_docs & changed_ids))
            next_chunk_count = max(0, previous.chunk_count - sum(previous_chunks_by_doc.get(value, 0) for value in changed_ids))
        else:
            new_ids = changed_ids - previous_docs
            next_document_count = previous.document_count + len(new_ids)
            next_chunk_count = previous.chunk_count - sum(previous_chunks_by_doc.get(value, 0) for value in changed_ids) + sum(len(value.chunk_ids) for value in documents)
        generation = generations.create(
            IndexGeneration(
                generation_id=f"IG_{secrets.token_hex(8)}",
                corpus_version=corpus_version,
                scope_version=scope_version,
                state="pending",
                document_count=next_document_count,
                chunk_count=next_chunk_count,
                previous_generation_id=previous.generation_id,
                workspace_id=workspace_id,
                index_profile_id=profile.profile_id,
            )
        )
        try:
            fork = getattr(engine, "fork_generation", None)
            if not callable(fork):
                raise RuntimeError("KnowledgeEngine 不支持 generation 快照分叉。")
            fork(previous, generation)
            if operation == "delete":
                metrics = engine.delete_documents(ordered_changed_ids, generation=generation)
            else:
                metrics = engine.update_documents(documents, generation=generation, profile=profile)
            metrics = {
                **metrics,
                "indexed_document_count": next_document_count,
                "incremental_operation": operation,
                "changed_document_ids": list(ordered_changed_ids),
                "processed_document_ids": [value.document_id for value in documents]
                if operation != "delete"
                else list(ordered_changed_ids),
                "unrelated_history_reprocessed_count": 0,
                "full_document_extraction_count": 0,
                "full_rebuild": False,
            }
            generation = generations.transition(generation.generation_id, "validating")
            valid, reason = self.validator(generation, metrics)
            if not valid:
                raise RuntimeError(reason)
            if activate:
                generation = generations.activate(generation.generation_id)
            return generation, metrics
        except BaseException as error:
            generations.fail(generation.generation_id, error)
            raise

    def synchronize_after_parse(
        self,
        embedding_provider: Any,
        *,
        emit: Callable[[str, str, float, dict[str, Any] | None], None] | None = None,
        catalog_builder: Callable[[Path], Any] | None = None,
    ) -> dict[str, Any]:
        """Web/CLI 在解析后调用的唯一旧索引兼容编排点。

        LightRAG 影子 generation 由 ``build_generation``/``update_documents``
        独立推进；此方法只保持阶段一对 catalog/BM25/Dense 的外部行为。
        """

        if self.project_root is None:
            raise RuntimeError("synchronize_after_parse 需要 project_root。")
        root = self.project_root
        operation_id = f"IO_{secrets.token_hex(8)}"
        lifecycle = root / "data" / "knowledge" / "index_operations" / f"{operation_id}.json"

        def record(state: str, **details: Any) -> None:
            write_json_atomic(
                lifecycle,
                {
                    "schema_version": "1.0",
                    "operation_id": operation_id,
                    "state": state,
                    "recoverable": state == "FAILED",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    **details,
                },
            )

        record("REGISTERED")
        try:
            record("PARSING", note="PDF parse completed before index orchestration")
            if catalog_builder is None:
                from app.knowledge.catalog import rebuild_catalog

                catalog_builder = rebuild_catalog
            catalog = catalog_builder(root)
            record("CANONICAL_READY", catalog_record_count=len(getattr(catalog, "records", [])))
            if emit:
                emit("building_knowledge", "正在通过 KnowledgeIndexService 更新 Chunk 和 BM25 索引。", 0.96, None)
            record("INDEXING")
            from app.chunking.builder import build_knowledge_base

            knowledge = build_knowledge_base(root)
            if knowledge.issues:
                raise RuntimeError("知识库构建存在未解决问题。")
            from app.indexing.dense import build_dense_index

            if emit:
                emit("building_dense", "正在通过 KnowledgeIndexService 更新迁移期 Dense 索引。", 0.98, None)
            dense = build_dense_index(root, embedding_provider)
            from app.workspaces import WorkspaceService

            workspace = WorkspaceService(root).synchronize_default_documents()
            record(
                "KNOWLEDGE_READY",
                document_count=int(getattr(knowledge, "document_count", 0)),
                chunk_count=int(getattr(knowledge, "total_chunk_count", 0)),
                workspace_id=workspace.workspace_id,
                scope_version=workspace.scope_version,
            )
            result = {
                "operation_id": operation_id,
                "lifecycle_state": "COMPLETED",
                "catalog": catalog.summary(),
                "knowledge": knowledge.to_dict(),
                "dense": dense.to_dict(),
                "workspace_id": workspace.workspace_id,
                "scope_version": workspace.scope_version,
            }
            record("COMPLETED", result=result)
            return result
        except BaseException as error:
            record("FAILED", failure_type=type(error).__name__, message=str(error))
            raise
