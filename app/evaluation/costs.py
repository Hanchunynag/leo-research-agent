"""阶段四真实 Provider 成本与延迟分账。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ProviderPricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if min(
            self.input_per_million,
            self.output_per_million,
            self.cached_input_per_million,
        ) < 0:
            raise ValueError("Provider pricing 不能为负数。")


@dataclass(frozen=True, slots=True)
class RealRunCostMetrics:
    workflow: str
    measurement_source: str
    router_context_tokens: int
    selected_evidence_tokens: int
    generator_input_tokens: int
    generator_output_tokens: int
    semantic_validator_tokens: int
    cached_input_tokens: int
    retrieval_tool_calls: int
    llm_calls: int
    total_latency_ms: float
    first_token_latency_ms: float | None
    cache_hit_rate: float | None
    monetary_cost: float | None
    currency: str | None
    lightrag_query_tokens: int
    lightrag_ingestion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealCostMetricsCollector:
    """只接受 Provider 实测 usage；fixture 数据必须走阶段三快照。"""

    @staticmethod
    def _provider_totals(values: list[Mapping[str, Any]]) -> dict[str, int | float | None]:
        return {
            "input": sum(
                int(value.get("prompt_tokens") or value.get("input_tokens") or 0)
                for value in values
            ),
            "output": sum(
                int(value.get("completion_tokens") or value.get("output_tokens") or 0)
                for value in values
            ),
            "cached": sum(
                int(
                    value.get("prompt_cache_hit_tokens")
                    or value.get("cached_tokens")
                    or 0
                )
                for value in values
            ),
            "first_token_ms": next(
                (
                    float(value["first_token_latency_ms"])
                    for value in values
                    if value.get("first_token_latency_ms") is not None
                ),
                None,
            ),
        }

    def collect(
        self,
        result: Mapping[str, Any],
        *,
        measurement_source: str,
        provider_measured: bool,
        pricing: ProviderPricing | None = None,
        lightrag_query_tokens: int = 0,
        lightrag_ingestion_tokens: int = 0,
    ) -> RealRunCostMetrics:
        if not provider_measured or measurement_source in {
            "fixture",
            "control_flow_fixture",
        }:
            raise ValueError("真实成本报告拒绝 fixture Token。")
        diagnostics = result.get("diagnostics")
        harness = diagnostics if isinstance(diagnostics, Mapping) else {}
        if "harness" in harness and isinstance(harness.get("harness"), Mapping):
            harness = harness["harness"]  # type: ignore[assignment]
        raw_provider = harness.get("provider_usage")
        provider_values = (
            [value for value in raw_provider if isinstance(value, Mapping)]
            if isinstance(raw_provider, list)
            else []
        )
        if not provider_values:
            raise ValueError("真实成本报告缺少 Provider usage。")
        totals = self._provider_totals(provider_values)
        context = harness.get("context_usage")
        context_usage = context if isinstance(context, Mapping) else {}
        usage = harness.get("usage")
        budget_usage = usage if isinstance(usage, Mapping) else {}
        trace = harness.get("trace")
        trace_values = trace if isinstance(trace, list) else []
        retrieval_calls = sum(
            1
            for value in trace_values
            if isinstance(value, Mapping)
            and value.get("kind") == "tool"
            and value.get("name") == "knowledge.retrieve"
            and value.get("status") == "succeeded"
        )
        validation = result.get("validation")
        semantic = validation if isinstance(validation, Mapping) else {}
        semantic_tokens = int(semantic.get("semantic_input_tokens") or 0) + int(
            semantic.get("semantic_output_tokens") or 0
        )
        input_tokens = int(totals["input"] or 0)
        output_tokens = int(totals["output"] or 0)
        cached_tokens = int(totals["cached"] or 0)
        cost: float | None = None
        currency: str | None = None
        if pricing is not None:
            uncached = max(0, input_tokens - cached_tokens)
            cost = (
                uncached * pricing.input_per_million
                + cached_tokens * pricing.cached_input_per_million
                + output_tokens * pricing.output_per_million
            ) / 1_000_000
            currency = pricing.currency
        cache_hit_rate = (
            cached_tokens / input_tokens if input_tokens > 0 else None
        )
        return RealRunCostMetrics(
            workflow=str(result.get("workflow") or harness.get("workflow") or ""),
            measurement_source=measurement_source,
            router_context_tokens=int(context_usage.get("router") or 0),
            selected_evidence_tokens=int(
                context_usage.get("selected_evidence") or 0
            ),
            generator_input_tokens=input_tokens,
            generator_output_tokens=output_tokens,
            semantic_validator_tokens=semantic_tokens,
            cached_input_tokens=cached_tokens,
            retrieval_tool_calls=retrieval_calls,
            llm_calls=int(budget_usage.get("llm_calls") or len(provider_values)),
            total_latency_ms=float(harness.get("elapsed_ms") or 0.0),
            first_token_latency_ms=(
                float(totals["first_token_ms"])
                if totals["first_token_ms"] is not None
                else None
            ),
            cache_hit_rate=cache_hit_rate,
            monetary_cost=cost,
            currency=currency,
            lightrag_query_tokens=lightrag_query_tokens,
            # 保留独立字段并明确不参与本次问答 monetary_cost。
            lightrag_ingestion_tokens=lightrag_ingestion_tokens,
        )

