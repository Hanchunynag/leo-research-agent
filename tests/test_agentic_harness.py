from __future__ import annotations

import json

import pytest

from app.agentic.harness import (
    AgenticRunHarness,
    AgenticRunPolicy,
    AgenticStage,
    HarnessConstraintError,
    TerminationReason,
)
from app.agentic.config import AgenticRAGConfig


def test_harness_rejects_invalid_policy_and_state_transition() -> None:
    with pytest.raises(ValueError, match="max_retrieval_rounds"):
        AgenticRunPolicy(max_retrieval_rounds=0)
    with pytest.raises(ValueError, match="fail_closed"):
        AgenticRAGConfig(fail_closed=False)
    harness = AgenticRunHarness(AgenticRunPolicy())

    with pytest.raises(HarnessConstraintError, match="非法 Harness 状态迁移"):
        with harness.stage(AgenticStage.GENERATING):
            pass

    assert harness.state == AgenticStage.INITIALIZED


def test_harness_enforces_retrieval_and_answer_repair_budgets() -> None:
    harness = AgenticRunHarness(
        AgenticRunPolicy(
            max_retrieval_rounds=1,
            max_answer_repairs=1,
        )
    )

    assert harness.begin_retrieval_round() == 1
    assert harness.can_retrieve() is False
    with pytest.raises(HarnessConstraintError, match="检索轮数预算"):
        harness.begin_retrieval_round()
    assert harness.begin_answer_repair() == 1
    assert harness.can_repair_answer() is False
    with pytest.raises(HarnessConstraintError, match="答案 Repair 预算"):
        harness.begin_answer_repair()


def test_harness_deadline_stops_new_work() -> None:
    values = iter([0.0, 0.002, 0.002, 0.002])
    harness = AgenticRunHarness(
        AgenticRunPolicy(max_total_latency_ms=1),
        clock=lambda: next(values),
    )

    assert harness.deadline_exceeded() is True
    assert harness.can_retrieve() is False
    assert harness.diagnostics()["termination_reason"] == "budget_exhausted"


def test_harness_trace_is_ordered_and_redacts_exception_body() -> None:
    secret = "secret-value-that-must-not-enter-trace"
    harness = AgenticRunHarness(AgenticRunPolicy())

    with pytest.raises(ValueError, match="secret-value"):
        with harness.stage(
            AgenticStage.ROUTING,
            details={"candidate_count": 3, "api_key": secret},
        ):
            raise ValueError(secret)

    diagnostics = harness.diagnostics()
    serialized = json.dumps(diagnostics, ensure_ascii=False)
    trace = diagnostics["trace"]
    assert diagnostics["state"] == "failed"
    assert diagnostics["termination_reason"] == "internal_error"
    assert trace[0]["ordinal"] == 1
    assert trace[0]["stage"] == "routing"
    assert trace[0]["status"] == "failed"
    assert trace[0]["error_type"] == "ValueError"
    assert trace[0]["details"]["api_key"] == "[REDACTED]"
    assert secret not in serialized


def test_harness_records_valid_terminal_run() -> None:
    harness = AgenticRunHarness(AgenticRunPolicy())

    with harness.stage(AgenticStage.ROUTING):
        pass
    with harness.stage(AgenticStage.PLANNING):
        pass
    harness.begin_retrieval_round()
    with harness.stage(AgenticStage.RETRIEVING, attempt=1):
        pass
    with harness.stage(AgenticStage.RERANKING, attempt=1):
        pass
    with harness.stage(AgenticStage.COVERAGE_CHECKING, attempt=1):
        pass
    with harness.stage(AgenticStage.CONTEXT_BUILDING):
        pass
    with harness.stage(AgenticStage.GENERATING):
        pass
    with harness.stage(AgenticStage.STRUCTURAL_VALIDATING):
        pass
    with harness.stage(AgenticStage.SEMANTIC_VALIDATING):
        pass
    with harness.stage(AgenticStage.PERSISTING):
        pass
    harness.finish(answerable=True, reason=TerminationReason.COMPLETED)

    diagnostics = harness.diagnostics()
    assert diagnostics["state"] == "completed"
    assert diagnostics["termination_reason"] == "completed"
    assert diagnostics["budget"]["retrieval_rounds_used"] == 1
    assert [item["ordinal"] for item in diagnostics["trace"]] == list(
        range(1, len(diagnostics["trace"]) + 1)
    )
