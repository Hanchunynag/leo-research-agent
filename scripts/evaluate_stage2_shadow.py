"""对 active LightRAG shadow generation 执行可复现的真实检索验收。

正式回答仍由 legacy RRF 提供。本脚本只读取 active LightRAG generation，
不会切换回答引擎，也不会修改索引内容。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from app.contracts import CandidateEvidence, EvidenceRequest, VerifiedEvidence
from app.contracts.adapters import LegacyKnowledgeEngineAdapter
from app.corpus import CanonicalCorpusService
from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
from app.evaluation.shadow import compare_retrievals
from app.evidence import EvidenceIntelligencePipeline
from app.generation.openai_compatible import (
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleConfig,
)
from app.generation.settings import load_local_llm_settings
from app.knowledge_engine import (
    IndexGenerationRepository,
    LightRAGKnowledgeEngine,
    LightRAGUsage,
    build_lightrag_client_config,
)
from app.runtime.retrieval import RetrievalRuntime
from app.storage import write_json_atomic
from app.web.runtime import WebRuntimeConfig
from app.workspaces import WorkspaceService


REPRESENTATIVE_QUERY = (
    "When ground-truth satellite ephemerides are unavailable, what target is used "
    "to train the orbit-prediction neural network?"
)
RELATION_QUERY = (
    "How are LEO-NNPON, SGP4, and TLE data related when training an orbit-prediction "
    "neural network without ground-truth satellite ephemerides?"
)
REPRESENTATIVE_RELEVANT = {"D_060e764f208c_cp02_c000012"}
GRAPH_BACKFILL_THRESHOLD = 0.95


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def _candidate_summary(values: tuple[CandidateEvidence, ...]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value.retrieval_source] = counts.get(value.retrieval_source, 0) + 1
    return {
        "count": len(values),
        "source_counts": counts,
        "chunk_ids": [value.chunk_id for value in values if value.chunk_id],
        "evidence_ids": [value.evidence_id for value in values],
    }


def _verified_summary(values: tuple[VerifiedEvidence, ...]) -> dict[str, Any]:
    inference = [value.evidence_id for value in values if value.evidence_grade == "graph_inference"]
    return {
        "count": len(values),
        "chunk_ids": [value.chunk_id for value in values],
        "evidence_ids": [value.evidence_id for value in values],
        "graph_inference_count": len(inference),
        "graph_inference_evidence_ids": inference,
    }


def _evaluate_query(
    *,
    query_id: str,
    query: str,
    relevant: set[str] | None,
    workspace_id: str,
    scope_version: int,
    official: LegacyKnowledgeEngineAdapter,
    shadow: LightRAGKnowledgeEngine,
    evidence: EvidenceIntelligencePipeline,
    usage: LightRAGUsage,
) -> dict[str, Any]:
    request = EvidenceRequest(
        request_id=query_id,
        workspace_id=workspace_id,
        scope_version=scope_version,
        query=query,
        top_k=10,
        retrieval_options={
            "candidate_limit": 40,
            "max_chunks_per_work": 20,
            "rrf_k": 60,
        },
    )

    official_started = perf_counter()
    official_values = tuple(official.retrieve_candidates(request))
    official_ms = (perf_counter() - official_started) * 1000

    before = usage.to_dict()
    shadow_started = perf_counter()
    shadow_candidates = tuple(shadow.retrieve_candidates(request))
    shadow_ms = (perf_counter() - shadow_started) * 1000
    query_usage = _usage_delta(before, usage.to_dict())

    bundle = evidence.verify(request, shadow_candidates)
    verified = tuple(bundle.evidence)
    selected = tuple(value.evidence for value in evidence.select(request, bundle))
    diagnostics = dict(evidence.last_diagnostics)
    comparison = compare_retrievals(
        query,
        official_values,
        selected,
        k=request.top_k,
        relevant_chunk_ids=relevant,
        official_elapsed_ms=official_ms,
        shadow_elapsed_ms=shadow_ms,
        shadow_diagnostics=diagnostics,
    )
    comparison["query_id"] = query_id
    comparison["official"]["llm_calls"] = 0
    comparison["official"]["llm_token_usage"] = 0
    comparison["shadow"]["llm_calls"] = query_usage["llm_calls"]
    comparison["shadow"]["llm_token_usage"] = query_usage["total_tokens"]
    comparison["shadow"]["usage"] = query_usage
    comparison["shadow"]["raw_candidates"] = _candidate_summary(shadow_candidates)
    comparison["shadow"]["verified"] = _verified_summary(verified)
    comparison["shadow"]["selected"] = _verified_summary(selected)
    comparison["shadow"]["rejected_candidate_ids"] = list(bundle.rejected_candidate_ids)
    comparison["shadow"]["evidence_diagnostics"] = diagnostics
    return comparison


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "evaluation" / "stage2_shadow_comparison.json"
    prior_report: dict[str, Any] = {}
    if output.is_file():
        value = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            prior_report = value
    corpus = CanonicalCorpusService(root)
    workspaces = WorkspaceService(root, corpus=corpus)
    workspace = workspaces.ensure_default()
    generations = IndexGenerationRepository(root)
    active = generations.active(workspace.workspace_id)
    if active is None:
        raise RuntimeError("default workspace 没有 active LightRAG generation。")
    if active.scope_version != workspace.scope_version:
        raise RuntimeError("active LightRAG generation 与 default workspace scope_version 不一致。")

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
    legacy = LegacyKnowledgeEngineAdapter(RetrievalRuntime(root, embedding))

    llm = load_local_llm_settings(root)
    if not llm.base_url or not llm.model:
        raise RuntimeError("缺少 LEO_LLM_BASE_URL/LEO_LLM_MODEL。")
    lightrag_model = os.getenv("LEO_LIGHTRAG_LLM_MODEL") or "deepseek-chat"
    completion = OpenAICompatibleAnswerProvider(
        OpenAICompatibleConfig(
            base_url=llm.base_url,
            model=lightrag_model,
            api_key=llm.api_key.get_secret_value() if llm.api_key else None,
            timeout_seconds=llm.timeout_seconds,
            max_tokens=min(llm.max_tokens, 8_192),
            prompt_layout=llm.prompt_layout or "context_first",
            json_mode=False,
        )
    )
    config, usage = build_lightrag_client_config(
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
    shadow = LightRAGKnowledgeEngine(
        root,
        corpus=corpus,
        workspaces=workspaces,
        generations=generations,
        client_config=config,
    )
    evidence = EvidenceIntelligencePipeline(
        corpus,
        workspaces,
        max_candidates_per_document=100,
        max_selected_per_document=100,
    )

    started = perf_counter()
    reports = [
        _evaluate_query(
            query_id="Q001",
            query=REPRESENTATIVE_QUERY,
            relevant=REPRESENTATIVE_RELEVANT,
            workspace_id=workspace.workspace_id,
            scope_version=workspace.scope_version,
            official=legacy,
            shadow=shadow,
            evidence=evidence,
            usage=usage,
        ),
        _evaluate_query(
            query_id="REL001",
            query=RELATION_QUERY,
            relevant=None,
            workspace_id=workspace.workspace_id,
            scope_version=workspace.scope_version,
            official=legacy,
            shadow=shadow,
            evidence=evidence,
            usage=usage,
        ),
    ]
    relation = reports[1]["shadow"]
    graph_count = int(relation["evidence_diagnostics"].get("graph_candidate_count") or 0)
    backfill_rate = relation.get("graph_backfill_rate")
    acceptance = {
        "representative_recall_preserved": reports[0]["shadow"]["recall_at_k"] == 1.0,
        "relation_candidates_present": graph_count > 0,
        "graph_backfill_threshold": GRAPH_BACKFILL_THRESHOLD,
        "graph_backfill_passed": isinstance(backfill_rate, (int, float))
        and backfill_rate >= GRAPH_BACKFILL_THRESHOLD,
        "cross_workspace_leakage_passed": all(
            value["shadow"]["cross_workspace_leakage_count"] == 0 for value in reports
        ),
    }
    acceptance["passed"] = all(
        value for key, value in acceptance.items() if key.endswith("_passed") or key in {"representative_recall_preserved", "relation_candidates_present"}
    )
    aggregate_usage = usage.to_dict()
    prior_usage = prior_report.get("cold_query_usage_observed")
    if not isinstance(prior_usage, dict) or int(prior_usage.get("llm_calls") or 0) < 1:
        prior_usage = prior_report.get("aggregate_usage")
    cold_usage = (
        dict(prior_usage)
        if aggregate_usage["llm_calls"] == 0
        and isinstance(prior_usage, dict)
        and int(prior_usage.get("llm_calls") or 0) > 0
        else aggregate_usage
    )
    report = {
        "schema_version": "1.0",
        "official_answer_engine": "legacy",
        "shadow_engine": "lightrag",
        "generation": {
            "generation_id": active.generation_id,
            "state": active.state,
            "workspace_id": active.workspace_id,
            "scope_version": active.scope_version,
            "profile_id": active.index_profile_id,
        },
        "queries": reports,
        "aggregate_usage": aggregate_usage,
        "cold_query_usage_observed": cold_usage,
        "llm_cache_hit": aggregate_usage["llm_calls"] == 0,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "acceptance": acceptance,
        "cost_note": "Token 数已实测；未配置供应商单价，因此不虚构货币成本。",
    }
    write_json_atomic(output, report)
    print(json.dumps({**report, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
