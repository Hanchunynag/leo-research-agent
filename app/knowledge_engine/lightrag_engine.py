"""LightRAG 1.5.6 的 KnowledgeEngine 实现。

上层只看到 CandidateEvidence；LightRAG 原始 entity/relation/chunk 永不直接流向
Context Builder 或 Answer Generator。
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.contracts import CandidateEvidence, Document, EvidenceRequest, IndexGeneration, IndexProfile
from app.corpus import CanonicalCorpusService, CanonicalLocator
from app.storage import write_json_atomic
from app.workspaces import WorkspaceService


ClientFactory = Callable[[Path, str, IndexProfile], Any]


@dataclass(frozen=True, slots=True)
class LightRAGClientConfig:
    """显式注入 LightRAG 所需模型函数，不从 Agent/Harness 获取客户端。"""

    embedding_func: Any
    llm_model_func: Callable[..., object]
    llm_model_name: str
    options: Mapping[str, Any] | None = None

    def create(self, working_dir: Path, workspace_id: str, profile: IndexProfile) -> Any:
        from lightrag import LightRAG  # type: ignore[import-untyped]

        return LightRAG(
            working_dir=str(working_dir),
            workspace=workspace_id,
            embedding_func=self.embedding_func,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm_model_name,
            **dict(self.options or {}),
        )


def _sync(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("同步 LightRAGKnowledgeEngine 不能在运行中的 event loop 内调用。")


class LightRAGKnowledgeEngine:
    def __init__(
        self,
        project_root: Path,
        *,
        corpus: CanonicalCorpusService | None = None,
        workspaces: WorkspaceService | None = None,
        generations: Any | None = None,
        client_config: LightRAGClientConfig | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.corpus = corpus or CanonicalCorpusService(self.project_root)
        self.workspaces = workspaces or WorkspaceService(self.project_root, corpus=self.corpus)
        if generations is None:
            from app.knowledge_engine.generations import IndexGenerationRepository

            generations = IndexGenerationRepository(self.project_root)
        self.generations = generations
        if client_factory is None and client_config is None:
            raise ValueError("必须提供 LightRAGClientConfig 或 client_factory。")
        self.client_factory = client_factory or client_config.create  # type: ignore[union-attr]

    def _generation_dir(self, generation: IndexGeneration) -> Path:
        return self.project_root / "data" / "indexes" / "lightrag" / generation.workspace_id / generation.generation_id

    def _mapping_path(self, generation: IndexGeneration) -> Path:
        return self._generation_dir(generation) / "canonical_chunk_map.json"

    def fork_generation(self, previous: IndexGeneration, target: IndexGeneration) -> None:
        """复制已验证存储快照；后续只重算受影响文档及局部关系。"""

        source = self._generation_dir(previous)
        destination = self._generation_dir(target)
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        mapping_path = self._mapping_path(target)
        if mapping_path.is_file():
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            payload["generation_id"] = target.generation_id
            payload["scope_version"] = target.scope_version
            write_json_atomic(mapping_path, payload)

    def _load_mapping(self, generation: IndexGeneration) -> dict[str, dict[str, Any]]:
        path = self._mapping_path(generation)
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        chunks = value.get("chunks") if isinstance(value, dict) else None
        if not isinstance(chunks, dict):
            raise ValueError(f"LightRAG canonical mapping 损坏：{path}")
        return {str(key): dict(item) for key, item in chunks.items() if isinstance(item, dict)}

    def _save_mapping(self, generation: IndexGeneration, mapping: dict[str, dict[str, Any]]) -> None:
        write_json_atomic(
            self._mapping_path(generation),
            {
                "schema_version": "1.0",
                "generation_id": generation.generation_id,
                "workspace_id": generation.workspace_id,
                "scope_version": generation.scope_version,
                "chunks": mapping,
            },
        )

    @staticmethod
    async def _use_client(client: Any, operation: Callable[[Any], Any]) -> Any:
        await client.initialize_storages()
        try:
            return await operation(client)
        finally:
            await client.finalize_storages()

    @staticmethod
    def _internal_chunk_id(document_id: str, content: str) -> str:
        from lightrag.utils_pipeline import make_custom_chunk_id  # type: ignore[import-untyped]

        return make_custom_chunk_id(document_id, content)

    def _document_mapping(self, document: Document) -> tuple[list[str], dict[str, dict[str, Any]]]:
        allowed_chunk_ids = set(document.chunk_ids)
        locators = tuple(
            value
            for value in self.corpus.locate_document_chunks(document.document_id)
            if not allowed_chunk_ids or value.chunk_id in allowed_chunk_ids
        )
        texts = [value.content for value in locators if value.content]
        mapping = {
            self._internal_chunk_id(document.document_id, value.content): {
                "document_id": value.document_id,
                "chunk_id": value.chunk_id,
                "content_hash": __import__("hashlib").sha256(value.content.encode("utf-8")).hexdigest(),
            }
            for value in locators
            if value.content
        }
        return texts, mapping

    def index_documents(self, documents: Sequence[Document], *, generation: IndexGeneration, profile: IndexProfile) -> Mapping[str, Any]:
        async def operation(client: Any) -> dict[str, Any]:
            mapping: dict[str, dict[str, Any]] = {}
            indexed = 0
            for document in documents:
                chunks, additions = self._document_mapping(document)
                if not chunks:
                    continue
                await client.ainsert_custom_chunks("\n\n".join(chunks), chunks, doc_id=document.document_id)
                mapping.update(additions)
                indexed += 1
            graph = getattr(client, "chunk_entity_relation_graph", None)
            nodes = await graph.get_all_nodes() if graph is not None else []
            edges = await graph.get_all_edges() if graph is not None else []
            self._save_mapping(generation, mapping)
            return {
                "indexed_document_count": indexed,
                "mapped_chunk_count": len(mapping),
                "entity_count": len(nodes),
                "relation_count": len(edges),
                "incremental": True,
            }

        client = self.client_factory(self._generation_dir(generation), generation.workspace_id, profile)
        return _sync(self._use_client(client, operation))

    def update_documents(self, documents: Sequence[Document], *, generation: IndexGeneration, profile: IndexProfile) -> Mapping[str, Any]:
        async def operation(client: Any) -> dict[str, Any]:
            mapping = self._load_mapping(generation)
            updated = 0
            for document in documents:
                await client.adelete_by_doc_id(document.document_id)
                mapping = {key: value for key, value in mapping.items() if value.get("document_id") != document.document_id}
                chunks, additions = self._document_mapping(document)
                if chunks:
                    await client.ainsert_custom_chunks("\n\n".join(chunks), chunks, doc_id=document.document_id)
                    mapping.update(additions)
                updated += 1
            self._save_mapping(generation, mapping)
            return {"updated_document_count": updated, "mapped_chunk_count": len(mapping), "full_rebuild": False}

        client = self.client_factory(self._generation_dir(generation), generation.workspace_id, profile)
        return _sync(self._use_client(client, operation))

    def delete_documents(self, document_ids: Sequence[str], *, generation: IndexGeneration) -> Mapping[str, Any]:
        profile = IndexProfile(profile_id=generation.index_profile_id)

        async def operation(client: Any) -> dict[str, Any]:
            mapping = self._load_mapping(generation)
            results: list[dict[str, Any]] = []
            for document_id in document_ids:
                value = await client.adelete_by_doc_id(document_id)
                results.append({"document_id": document_id, "status": str(getattr(value, "status", "unknown")), "message": str(getattr(value, "message", ""))})
                mapping = {key: item for key, item in mapping.items() if item.get("document_id") != document_id}
            self._save_mapping(generation, mapping)
            return {"deleted_document_count": len(document_ids), "mapped_chunk_count": len(mapping), "results": results, "full_rebuild": False}

        client = self.client_factory(self._generation_dir(generation), generation.workspace_id, profile)
        return _sync(self._use_client(client, operation))

    def retrieve(self, request: EvidenceRequest) -> Sequence[CandidateEvidence]:
        return self.retrieve_candidates(request)

    def retrieve_candidates(self, request: EvidenceRequest) -> Sequence[CandidateEvidence]:
        generation = self.generations.active(request.workspace_id)
        if generation is None:
            return ()
        if generation.scope_version != request.scope_version:
            raise ValueError("请求 scope_version 与 active generation 不一致。")
        profile = IndexProfile(
            profile_id=generation.index_profile_id,
            query_mode=str(request.retrieval_options.get("mode", "mix")),  # type: ignore[arg-type]
            top_k=request.top_k,
            chunk_top_k=max(request.top_k, int(request.retrieval_options.get("chunk_top_k", request.top_k))),
        )

        async def operation(client: Any) -> dict[str, Any]:
            from lightrag import QueryParam  # type: ignore[import-untyped]

            param = QueryParam(
                mode=profile.query_mode,
                top_k=profile.top_k,
                chunk_top_k=profile.chunk_top_k,
                max_entity_tokens=profile.max_entity_tokens,
                max_relation_tokens=profile.max_relation_tokens,
                max_total_tokens=profile.max_total_tokens,
                enable_rerank=bool(request.retrieval_options.get("enable_rerank", False)),
            )
            value = await client.aquery_data(request.query, param)
            return value if isinstance(value, dict) else {}

        client = self.client_factory(self._generation_dir(generation), generation.workspace_id, profile)
        raw = _sync(self._use_client(client, operation))
        candidates = self._map_results(request, generation, raw)
        scoped = self.workspaces.filter_candidates(
            request.workspace_id, request.scope_version, candidates
        )
        # LightRAG 的 Chunk/Entity/Relation 分数不在同一量纲。Candidate 阶段按
        # 来源各保留 top_k，避免在 Evidence Intelligence 前由某一来源挤掉其余来源。
        source_counts: dict[str, int] = {}
        balanced: list[CandidateEvidence] = []
        for value in scoped:
            count = source_counts.get(value.retrieval_source, 0)
            if count >= request.top_k:
                continue
            source_counts[value.retrieval_source] = count + 1
            balanced.append(value)
        return tuple(balanced)

    def _source_locators(self, source_id: Any, mapping: Mapping[str, Mapping[str, Any]]) -> tuple[CanonicalLocator, ...]:
        ids = str(source_id or "").split("<SEP>")
        locators: list[CanonicalLocator] = []
        for internal_id in ids:
            mapped = mapping.get(internal_id)
            if mapped is None:
                continue
            locator = self.corpus.locate_chunk(str(mapped.get("chunk_id") or ""))
            if locator is not None:
                locators.append(locator)
        return tuple(locators)

    def _map_results(self, request: EvidenceRequest, generation: IndexGeneration, raw: Mapping[str, Any]) -> tuple[CandidateEvidence, ...]:
        data = raw.get("data")
        if not isinstance(data, Mapping):
            return ()
        mapping = self._load_mapping(generation)
        candidates: list[CandidateEvidence] = []
        source_positions: dict[str, int] = {}

        def append(source_type: str, item: Mapping[str, Any], locator: CanonicalLocator, rank: int, relation_path: tuple[str, ...] = ()) -> None:
            content = locator.content if source_type == "chunk" else str(item.get("description") or locator.content)
            identity = str(item.get("chunk_id") or item.get("reference_id") or f"{source_type}-{rank}-{locator.chunk_id}")
            raw_score = item.get("score") or item.get("weight")
            source_positions[source_type] = source_positions.get(source_type, 0) + 1
            source_position = source_positions[source_type]
            candidates.append(
                CandidateEvidence(
                    candidate_id=f"lightrag:{source_type}:{identity}:{locator.chunk_id}",
                    request_id=request.request_id,
                    workspace_id=request.workspace_id,
                    scope_version=request.scope_version,
                    content=content,
                    # 各来源的原始 score/weight 不可直接比较；对外提供源内倒数排名分数，
                    # 原始值仍保留在 metadata 供诊断和后续学习排序使用。
                    score=1.0 / source_position,
                    retrieval_source=f"lightrag_{source_type}",
                    work_id=locator.work_id,
                    document_id=locator.document_id,
                    chunk_id=locator.chunk_id,
                    section_id=locator.section_id,
                    page_start=locator.page_start,
                    page_end=locator.page_end,
                    block_ids=locator.block_ids,
                    relation_path=relation_path,
                    directness="direct" if source_type == "chunk" else None,
                    metadata={
                        "lightrag": dict(item),
                        "raw_retrieval_score": raw_score,
                        "source_rank": rank,
                        "source_candidate_position": source_position,
                        **locator.metadata,
                    },
                )
            )

        for source_type, key in (("chunk", "chunks"), ("entity", "entities"), ("relation", "relationships")):
            values = data.get(key)
            if not isinstance(values, list):
                continue
            for rank, item in enumerate((value for value in values if isinstance(value, Mapping)), 1):
                source_id = item.get("chunk_id") if source_type == "chunk" else item.get("source_id")
                relation_path = (str(item.get("src_id")), str(item.get("tgt_id"))) if source_type == "relation" else ()
                for locator in self._source_locators(source_id, mapping):
                    append(source_type, item, locator, rank, relation_path)
        unique: dict[str, CandidateEvidence] = {}
        for value in candidates:
            unique.setdefault(value.candidate_id, value)
        return tuple(unique.values())

    def get_status(self) -> Mapping[str, Any]:
        return {
            "engine": "lightrag",
            "active_generations": {value.workspace_id: value.generation_id for value in self.generations.list() if value.state == "active"},
            "generations": [value.generation_id for value in self.generations.list()],
        }
