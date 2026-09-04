from __future__ import annotations

from pathlib import Path
from typing import Any

from app.indexing.bm25 import build_bm25_index, write_bm25_index
from app.indexing.paper import papers_digest
from app.langchain_agent.tools import BilingualRetrievalTool
from app.research.context import ResearchContextManager
from app.retrieval.hierarchical import _paper_relevance_gate
from app.retrieval.paper import paper_retrieval_text
from app.storage import write_jsonl_atomic
from app.retrieval.search import search_evidence


def _chunk(chunk_id: str, paper_id: str, content: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "work_id": paper_id,
        "document_id": paper_id.replace("P_", "D_"),
        "title": "Paper",
        "section_path": ["Methods"],
        "page_start": 2,
        "page_end": 3,
        "content": content,
        "parent_contexts": [],
        "overlap_context": None,
    }


def test_paper_digest_ignores_mysql_operational_columns_and_order() -> None:
    first = {
        "paper_id": "P_1",
        "title": "A",
        "abstract": "abstract",
        "authors": ["Author"],
        "year": 2024,
        "keywords": ["LEO"],
        "doi": None,
        "work_id": "W_1",
        "document_id": "D_1",
    }
    second = {**first, "created_time": "2026-01-01", "metadata": {"index_epoch": "x"}}
    assert papers_digest([first]) == papers_digest([second])
    assert papers_digest([first, {**first, "paper_id": "P_0"}]) == papers_digest(
        [{**first, "paper_id": "P_0"}, first]
    )


def test_chunk_bm25_persists_paper_id_and_supports_hierarchical_scope(tmp_path: Path) -> None:
    chunks = [
        _chunk("C_1", "P_1", "alpha navigation evidence"),
        _chunk("C_2", "P_2", "alpha unrelated evidence"),
    ]
    write_jsonl_atomic(tmp_path / "data" / "knowledge" / "chunks.jsonl", chunks)
    write_bm25_index(tmp_path, build_bm25_index(chunks))

    index = (tmp_path / "data" / "index" / "bm25.json").read_text()
    assert '"paper_id": "P_1"' in index
    result = search_evidence(tmp_path, "alpha navigation", paper_ids=["P_1"], max_chunks_per_work=20)
    assert result["result_count"] == 1
    assert result["results"][0]["paper_id"] == "P_1"


def test_bilingual_tool_maps_target_documents_to_bounded_paper_filter() -> None:
    class Knowledge:
        workspace_id = "default"
        last_diagnostics: dict[str, Any] = {}

        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def retrieve_multi(self, queries: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
            self.kwargs = kwargs
            return {"results": []}

    knowledge = Knowledge()
    BilingualRetrievalTool(knowledge).retrieve(
        ["query"], "default", 1, target_document_ids=["D_1", "D_2"]
    )
    assert knowledge.kwargs["paper_filters"] == {"document_ids": ["D_1", "D_2"]}


def test_generator_context_balances_papers_and_keeps_citation_metadata() -> None:
    evidence = []
    for paper_id in ("P_1", "P_2", "P_3", "P_4", "P_5"):
        for ordinal in range(3):
            evidence.append(
                {
                    "evidence_id": f"E_{paper_id}_{ordinal}",
                    "paper_id": paper_id,
                    "document_id": paper_id.replace("P_", "D_"),
                    "chunk_id": f"C_{paper_id}_{ordinal}",
                    "title": f"Title {paper_id}",
                    "year": 2024,
                    "section_path": ["Methods", "Experiment"],
                    "page_start": 4,
                    "page_end": 5,
                    "content": "method result limitation " * 500,
                    "evidence_state": "selected",
                }
            )
    context = ResearchContextManager().generator_pack(
        query="compare",
        workspace_id="default",
        scope_version=1,
        scope_constraints=[],
        evidence=evidence,
        conflicts=[],
        output_format={},
    )
    assert len(context.selected_evidence) == 8
    counts: dict[str, int] = {}
    for value in context.selected_evidence:
        counts[str(value["paper_id"])] = counts.get(str(value["paper_id"]), 0) + 1
        assert value["title"]
        assert value["year"] == 2024
        assert value["section_path"] == ["Methods", "Experiment"]
        assert value["page_start"] == 4
        assert value["page_end"] == 5
    assert max(counts.values()) <= 2
    assert all(len(str(value["content"])) < 2000 for value in context.selected_evidence)


def test_relevance_gate_ignores_generic_bm25_overlap() -> None:
    paper = {
        "paper_id": "P_LEO",
        "title": "LEO ephemeris error correction",
        "abstract": "Experimental results for satellite navigation.",
        "authors": [],
        "keywords": [],
    }
    accepted, diagnostics = _paper_relevance_gate(
        "What are qubit error correction methods for quantum drug discovery?",
        {
            "branch_results": {
                "bm25": [{"paper_id": "P_LEO", "score": 1.0}],
                "dense": [{"paper_id": "P_LEO", "score": 0.42, **paper}],
            }
        },
    )
    assert not accepted
    assert diagnostics["metadata_overlap"] == []
    assert paper_retrieval_text(paper)
