from __future__ import annotations

from pathlib import Path
from typing import Any

from app.contracts import CandidateEvidence
from app.corpus import CanonicalCorpusService
from app.evidence import EvidenceIntelligencePipeline
from app.evaluation.shadow import ShadowReportStore
from app.knowledge_engine import UnifiedKnowledgeService
from app.workspaces import WorkspaceService
from tests.test_stage2_corpus_workspace import write_fixture


class OfficialRuntime:
    embedding_provider = object()
    reranker_provider = None
    last_diagnostics = {"retrieval_mode": "legacy"}

    def retrieve(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "retriever": "legacy_rrf",
            "results": [
                {
                    "rank": 1,
                    "score": 0.9,
                    "retrieval_source": "hybrid_rrf",
                    "work_id": "W_001",
                    "document_id": "D_001",
                    "chunk_id": "D_001_c1",
                    "page_start": 2,
                    "page_end": 2,
                    "block_ids": ["P_001_b1"],
                    "content": "canonical text",
                }
            ],
        }


class ShadowEngine:
    def retrieve_candidates(self, request: Any) -> tuple[CandidateEvidence, ...]:
        return (
            CandidateEvidence(
                "LIGHT-1",
                request.request_id,
                request.workspace_id,
                request.scope_version,
                "canonical text",
                0.8,
                "lightrag_chunk",
                work_id="W_001",
                document_id="D_001",
                chunk_id="D_001_c1",
            ),
        )


def test_unified_service_governs_official_and_records_non_serving_shadow(tmp_path: Path) -> None:
    write_fixture(tmp_path, "D_001")
    corpus = CanonicalCorpusService(tmp_path)
    workspaces = WorkspaceService(tmp_path, corpus=corpus)
    workspaces.ensure_default()
    evidence = EvidenceIntelligencePipeline(
        corpus,
        workspaces,
        max_candidates_per_document=20,
        max_selected_per_document=20,
    )
    reports = ShadowReportStore(tmp_path)
    service = UnifiedKnowledgeService(
        OfficialRuntime(),
        evidence,
        shadow_engine=ShadowEngine(),
        report_store=reports,
    )

    result = service.retrieve("query", limit=5)

    assert result["retriever"] == "legacy_rrf"
    assert result["results"][0]["evidence_state"] == "selected"
    assert service.last_diagnostics["active_engine"] == "legacy"
    assert service.last_diagnostics["shadow"]["comparison"]["top_k_overlap"] == 0.2
    assert Path(service.last_diagnostics["shadow_report"]).is_file()
