from __future__ import annotations

from typing import Any, Mapping

import pytest

from app.evaluation import ProviderPricing, RealCostMetricsCollector
from app.research import ResearchRuntime, WorkflowName, WorkflowRequest, build_default_gateway


class MeasuredProviderGenerator:
    def generate(self, context: Any) -> Mapping[str, Any]:
        return {
            "answerable": True,
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Canonical evidence supports navigation.",
                    "evidence_ids": ["E1"],
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 20,
                "first_token_latency_ms": 12.5,
            },
        }


def _measured_result() -> Mapping[str, Any]:
    handlers = {
        "workspace.read_scope": lambda arguments, context: {
            "workspace_id": "default",
            "scope_version": 1,
            "constraints": [],
        },
        "knowledge.retrieve": lambda arguments, context: {
            "results": [
                {
                    "evidence_id": "E1",
                    "evidence_state": "selected",
                    "document_id": "D_1",
                    "chunk_id": "D_1_c1",
                    "page_start": 1,
                    "page_end": 1,
                    "block_ids": ["B_1"],
                    "content": "Canonical evidence supports navigation.",
                    "evidence_grade": "primary",
                    "directness": "direct",
                }
            ]
        },
    }
    return ResearchRuntime(
        build_default_gateway(handlers), MeasuredProviderGenerator()
    ).run(
        WorkflowRequest(
            query="What supports navigation?",
            workspace_id="default",
            scope_version=1,
            workflow=WorkflowName.DIRECT_QA,
        )
    )


def test_real_cost_metrics_separate_context_provider_cache_and_ingestion() -> None:
    result = _measured_result()
    metrics = RealCostMetricsCollector().collect(
        result,
        measurement_source="test-provider-measured",
        provider_measured=True,
        pricing=ProviderPricing(
            input_per_million=1.0,
            output_per_million=2.0,
            cached_input_per_million=0.1,
        ),
        lightrag_query_tokens=30,
        lightrag_ingestion_tokens=50_000,
    )

    assert metrics.router_context_tokens > 0
    assert metrics.selected_evidence_tokens > 0
    assert metrics.generator_input_tokens == 100
    assert metrics.generator_output_tokens == 50
    assert metrics.cached_input_tokens == 20
    assert metrics.retrieval_tool_calls == 1
    assert metrics.llm_calls == 1
    assert metrics.first_token_latency_ms == 12.5
    assert metrics.cache_hit_rate == 0.2
    assert metrics.monetary_cost == pytest.approx(0.000182)
    assert metrics.lightrag_query_tokens == 30
    assert metrics.lightrag_ingestion_tokens == 50_000


@pytest.mark.parametrize(
    "source", ["fixture", "control_flow_fixture"]
)
def test_control_flow_fixture_cannot_be_reported_as_real_cost(source: str) -> None:
    with pytest.raises(ValueError, match="拒绝 fixture Token"):
        RealCostMetricsCollector().collect(
            _measured_result(),
            measurement_source=source,
            provider_measured=False,
        )


def test_missing_provider_usage_cannot_be_silently_estimated() -> None:
    with pytest.raises(ValueError, match="缺少 Provider usage"):
        RealCostMetricsCollector().collect(
            {
                "workflow": "bootstrap",
                "diagnostics": {
                    "usage": {"total_tokens": 999},
                    "provider_usage": [],
                },
            },
            measurement_source="production-run",
            provider_measured=True,
        )

