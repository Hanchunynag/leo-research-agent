"""构建并验证首个 LightRAG shadow generation。

该脚本不会切换正式回答引擎；generation 的 active 仅表示 LightRAG 内部已验证。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from app.contracts import IndexProfile
from app.corpus import CanonicalCorpusService
from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
from app.generation.openai_compatible import OpenAICompatibleAnswerProvider, OpenAICompatibleConfig
from app.generation.settings import load_local_llm_settings
from app.knowledge_engine import (
    IndexGenerationRepository,
    KnowledgeIndexService,
    LightRAGKnowledgeEngine,
    build_lightrag_client_config,
)
from app.storage import write_json_atomic
from app.web.runtime import WebRuntimeConfig
from app.workspaces import WorkspaceService


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = CanonicalCorpusService(root)
    workspace = WorkspaceService(root, corpus=corpus).synchronize_default_documents()
    web = WebRuntimeConfig.from_environment(root)
    embedding = BGEM3EmbeddingProvider(
        BGEM3Config(
            model_name=web.embedding_model,
            revision=web.embedding_revision,
            device=web.device,
            cache_folder=web.model_cache,
            batch_size=web.embedding_batch_size,
            local_files_only=web.local_files_only,
            show_progress_bar=False,
        )
    )
    llm = load_local_llm_settings(root)
    if not llm.base_url or not llm.model:
        raise RuntimeError("缺少 LEO_LLM_BASE_URL/LEO_LLM_MODEL。")
    lightrag_model = os.getenv("LEO_LIGHTRAG_LLM_MODEL") or llm.model
    completion = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig(
            base_url=llm.base_url,
            model=lightrag_model,
            api_key=llm.api_key.get_secret_value() if llm.api_key else None,
            timeout_seconds=llm.timeout_seconds,
            max_tokens=min(llm.max_tokens, 8_192),
            prompt_layout=llm.prompt_layout or "context_first",
            # LightRAG extraction prompts own the response grammar. DeepSeek rejects
            # response_format=json_object when a prompt does not literally say JSON.
            json_mode=False,
        )
    )
    client_config, usage = build_lightrag_client_config(
        embedding,
        completion,
        llm_model_name=lightrag_model,
        embedding_model_name=web.embedding_model,
        options={
            "entity_extraction_use_json": True,
            "entity_extract_max_gleaning": 0,
            "entity_extract_max_records": 10,
            "entity_extract_max_entities": 6,
            "llm_model_max_async": 1,
        },
    )
    generations = IndexGenerationRepository(root)
    engine = LightRAGKnowledgeEngine(
        root,
        corpus=corpus,
        workspaces=WorkspaceService(root, corpus=corpus),
        generations=generations,
        client_config=client_config,
    )
    safe_model = re.sub(r"[^a-z0-9]+", "-", lightrag_model.casefold()).strip("-")
    profile = IndexProfile(
        profile_id=f"lightrag-1.5.6-bge-m3-{safe_model}-json-v1",
        embedding_model=web.embedding_model,
        llm_model=lightrag_model,
    )
    all_documents = corpus.list_documents()
    raw_limit = os.getenv("LEO_LIGHTRAG_DOCUMENT_LIMIT")
    document_limit = int(raw_limit) if raw_limit else len(all_documents)
    documents = all_documents[:document_limit]
    raw_chunk_limit = os.getenv("LEO_LIGHTRAG_CHUNK_LIMIT")
    if raw_chunk_limit:
        chunk_limit = int(raw_chunk_limit)
        documents = tuple(
            replace(value, chunk_ids=value.chunk_ids[:chunk_limit])
            for value in documents
        )
    activate_generation = document_limit >= len(all_documents)
    corpus_version = hashlib.sha256(
        "\n".join(f"{value.document_id}:{value.source_sha256}" for value in documents).encode("utf-8")
    ).hexdigest()
    started = perf_counter()
    def validate(generation: object, metrics: dict[str, object]) -> tuple[bool, str]:
        if int(metrics.get("mapped_chunk_count") or 0) < sum(len(value.chunk_ids) for value in documents):
            return False, "Canonical Chunk mapping incomplete"
        if int(metrics.get("relation_count") or 0) < 1:
            return False, "LightRAG relation_count below acceptance threshold 1"
        return True, "validated"

    try:
        generation, metrics = KnowledgeIndexService(engine, generations, validator=validate).build_generation(
            documents,
            workspace_id=workspace.workspace_id,
            scope_version=workspace.scope_version,
            corpus_version=corpus_version,
            profile=profile,
            activate=activate_generation,
        )
    except BaseException as error:
        report = {
            "schema_version": "1.0",
            "status": "failed",
            "failure_type": type(error).__name__,
            "message": str(error),
            "usage": usage.to_dict(),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "official_answer_engine": "legacy",
            "shadow_engine": "lightrag",
            "active_generation": None,
        }
        output = root / "data" / "evaluation" / "stage2_lightrag_migration.json"
        write_json_atomic(output, report)
        print(json.dumps({**report, "output": str(output)}, ensure_ascii=False, indent=2))
        raise
    report = {
        "schema_version": "1.0",
        "generation": {
            "generation_id": generation.generation_id,
            "state": generation.state,
            "workspace_id": generation.workspace_id,
            "scope_version": generation.scope_version,
            "document_count": generation.document_count,
            "chunk_count": generation.chunk_count,
            "profile_id": generation.index_profile_id,
        },
        "metrics": dict(metrics),
        "usage": usage.to_dict(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "official_answer_engine": "legacy",
        "shadow_engine": "lightrag",
        "activation_requested": activate_generation,
    }
    output = root / "data" / "evaluation" / "stage2_lightrag_migration.json"
    write_json_atomic(output, report)
    print(json.dumps({**report, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
