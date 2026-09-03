from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import inspect

from app.langchain_agent.graph import LLMActionDecider
from app.persistence import MySQLConfig, MySQLKnowledgeRepository
from app.research.provisional import register_abstract_evidence
from app.indexing.paper import load_paper_records
from app.retrieval.search import load_chunks


def _repository(tmp_path: Path) -> MySQLKnowledgeRepository:
    return MySQLKnowledgeRepository(
        tmp_path,
        MySQLConfig(
            url=f"sqlite:///{tmp_path / 'structured.sqlite3'}",
            enabled=True,
        ),
    )


def test_structured_schema_contains_research_tables_and_never_vectors(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.initialize()
    names = set(inspect(repository.engine).get_table_names())  # type: ignore[arg-type]
    assert {
        "papers",
        "paper_authors",
        "paper_versions",
        "paper_sections",
        "chunks",
        "query_history",
        "citation_records",
        "temporary_evidence",
        "research_jobs",
        "index_epochs",
    } <= names
    assert not any("vector" in name.casefold() for name in names)


def test_repository_round_trip_and_abstract_only_audit(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path)
    repository.initialize()
    monkeypatch.setenv("LEO_MYSQL_ENABLED", "true")
    monkeypatch.setenv("LEO_MYSQL_URL", f"sqlite:///{tmp_path / 'structured.sqlite3'}")
    repository.upsert_paper(
        {
            "paper_id": "P1",
            "document_id": "D1",
            "title": "Doppler positioning",
            "authors": ["A"],
            "year": 2024,
            "abstract": "Abstract",
            "keywords": ["LEO"],
        }
    )
    repository.upsert_chunks(
        [
            {
                "chunk_id": "C1",
                "paper_id": "P1",
                "document_id": "D1",
                "section_path": ["Method"],
                "page_start": 3,
                "page_end": 4,
                "content": "A method passage.",
                "token_count": 3,
            }
        ]
    )
    assert repository.list_papers()[0]["paper_id"] == "P1"
    assert repository.list_chunks()[0]["chunk_id"] == "C1"

    result = register_abstract_evidence(
        tmp_path,
        "run-1",
        [{"paper_id": "P2", "title": "New paper", "abstract": "A provisional abstract", "year": 2025}],
    )
    assert result["evidence"][0]["abstract_based"] is True
    assert result["evidence"][0]["page_start"] is None
    assert (tmp_path / "data/candidate_knowledge/provisional_evidence").exists()


def test_mysql_is_first_source_for_index_projections(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path)
    repository.initialize()
    repository.upsert_paper(
        {"paper_id": "P_DB", "document_id": "D_DB", "title": "Database paper", "abstract": "A", "authors": [], "keywords": []}
    )
    repository.upsert_chunks(
        [{"chunk_id": "C_DB", "paper_id": "P_DB", "document_id": "D_DB", "content": "db fact", "section_path": []}]
    )
    monkeypatch.setenv("LEO_MYSQL_ENABLED", "true")
    monkeypatch.setenv("LEO_MYSQL_URL", f"sqlite:///{tmp_path / 'structured.sqlite3'}")
    assert load_paper_records(tmp_path)[0]["paper_id"] == "P_DB"
    assert load_chunks(tmp_path)[0]["chunk_id"] == "C_DB"


def test_llm_controller_cannot_expand_fixed_tool_gateway() -> None:
    class Provider:
        def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": json.dumps({"type": "tool", "tool_name": "literature.search"})}}],
            }

    action = LLMActionDecider(Provider())(
        {
            "original_query": "find papers",
            "iteration": 0,
            "max_iterations": 1,
        }
    )
    assert action["type"] == "tool"
    assert action["tool_name"] == "knowledge.retrieve"
