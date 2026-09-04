from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.agent import evaluate_agent_predictions, load_agent_questions
from app.evaluation.generation import evaluate_grounded_generation
from app.evaluation.hierarchical import evaluate_hierarchical


def test_agent_evaluator_checks_fixed_tools_budget_and_failure_preservation(tmp_path: Path) -> None:
    questions_path = tmp_path / "agent_questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "AG1",
                "query": "What is the method?",
                "expected_task_type": "direct_qa",
                "expected_retrieval_sufficient": True,
                "expected_paper_ids": [],
                "expected_evidence_ids": [],
                "required_tools": ["knowledge.retrieve"],
                "max_tool_calls": 2,
                "max_retrieval_rounds": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions = load_agent_questions(questions_path)
    report = evaluate_agent_predictions(
        questions,
        [
            {
                "question_id": "AG1",
                "selected_evidence": [
                    {
                        "evidence_id": "E1",
                        "page_start": 2,
                        "page_end": 3,
                    }
                ],
                "outcome": {"code": "generation_failed"},
                "diagnostics": {
                    "langgraph": {
                        "task_type": "direct_qa",
                        "tool_calls": [{"tool_name": "knowledge.retrieve"}],
                    },
                    "harness": {
                        "usage": {"tool_calls": 1, "retrieval_rounds": 1},
                        "policy": {"max_tool_calls": 2, "max_retrieval_rounds": 1},
                        "trace": [{"status": "succeeded"}],
                    },
                },
            }
        ],
    )
    metrics = report["metrics"]
    assert metrics["planner_task_accuracy"] == 1.0
    assert metrics["tool_validity"] == 1.0
    assert metrics["trajectory_budget_compliance"] == 1.0
    assert metrics["evidence_preserved_on_generation_failure"] == 1.0
    assert metrics["retrieval_sufficiency_accuracy"] == 1.0


def test_hierarchical_evaluator_reports_cascade_and_scope(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "query": "alpha method",
                "relevant_work_ids": [],
                "relevant_document_ids": ["D1"],
                "relevant_block_ids": [],
                "relevant_chunk_ids": ["C1"],
                "acceptable_evidence_ids": ["C1"],
                "excluded_document_ids": [],
                "question_type": "method",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunks = [{
        "chunk_id": "C1",
        "paper_id": "P1",
        "work_id": "W1",
        "document_id": "D1",
        "block_ids": ["B1"],
        "content": "alpha method",
    }]
    monkeypatch.setattr("app.evaluation.hierarchical.load_chunks", lambda _: chunks)
    monkeypatch.setattr(
        "app.evaluation.hierarchical.load_paper_records",
        lambda _: [{"paper_id": "P1", "document_id": "D1"}],
    )
    monkeypatch.setattr(
        "app.evaluation.hierarchical.search_hierarchical_evidence",
        lambda *args, **kwargs: {
            "candidate_paper_ids": ["P1"],
            "paper_retrieval": {
                "results": [{"paper_id": "P1", "rank": 1}],
                "branch_results": {
                    "bm25": [{"paper_id": "P1", "rank": 1}],
                    "dense": [{"paper_id": "P1", "rank": 1}],
                },
            },
            "results": [{"chunk_id": "C1", "paper_id": "P1"}],
        },
    )
    report = evaluate_hierarchical(
        tmp_path,
        questions_path,
        object(),  # type: ignore[arg-type]
        reranker_provider=None,
        k_values=(1,),
    )
    assert report["metrics"]["paper_recall@1"] == 1.0
    assert report["metrics"]["evidence_recall@1"] == 1.0
    assert report["metrics"]["cascade_recall"] == 1.0
    assert report["metrics"]["scope_leakage_count"] == 0


def test_generation_evaluator_separates_citation_contract_from_ragas() -> None:
    report = evaluate_grounded_generation(
        [
            {
                "question_id": "G1",
                "answerable": True,
                "expected_answerable": True,
                "reference_evidence_ids": ["E1"],
                "selected_evidence": [{"evidence_id": "E1"}],
                "claims": [{"evidence_ids": ["E1"]}],
                "citations": [{"evidence_id": "E1", "page_start": 2, "page_end": 2}],
            }
        ]
    )
    assert report["metrics"]["citation_scope_precision"] == 1.0
    assert report["metrics"]["citation_precision"] == 1.0
    assert report["metrics"]["citation_recall"] == 1.0
    assert report["metrics"]["answerable_accuracy"] == 1.0
    assert "faithfulness" in report["ragas_metrics_available"]
