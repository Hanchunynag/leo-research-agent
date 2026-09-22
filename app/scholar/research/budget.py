"""Bounded capability budget and trace for one Scholar research call."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from time import perf_counter
from typing import Any, Iterator


class CapabilityState(StrEnum):
    CREATED = "created"
    CONTEXT_PREPARING = "context_preparing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryLevel(IntEnum):
    FORMAT_OR_ARGUMENT = 0
    TRANSIENT_TOOL_RETRY = 1
    BACKEND_DEGRADATION = 2
    SCOPE_REDUCTION = 3
    LIMITED_RETRIEVAL = 4
    SAFE_TERMINATION = 5


@dataclass(frozen=True, slots=True)
class CapabilityBudgetPolicy:
    max_steps: int = 12
    max_llm_calls: int = 2
    max_tool_calls: int = 6
    max_retrieval_rounds: int = 2
    max_external_searches: int = 1
    max_repairs: int = 1
    max_context_tokens: int = 4_000
    max_total_tokens: int = 12_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} 不能为负数。")
        if self.max_steps < 1 or self.max_context_tokens < 1 or self.max_total_tokens < 1:
            raise ValueError("步骤和 Token 预算必须大于 0。")
        if self.max_context_tokens > self.max_total_tokens:
            raise ValueError("max_context_tokens 不能超过 max_total_tokens。")


@dataclass(slots=True)
class BudgetUsage:
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    retrieval_rounds: int = 0
    external_searches: int = 0
    repairs: int = 0
    context_tokens: int = 0
    total_tokens: int = 0


class CapabilityBudgetError(RuntimeError):
    pass


class BudgetExceeded(CapabilityBudgetError):
    pass


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if any(marker in str(key).casefold() for marker in ("secret", "api_key", "authorization", "prompt"))
            else _safe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else f"<{type(value).__name__}>"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    ordinal: int
    state: str
    kind: str
    name: str
    status: str
    elapsed_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class CapabilityBudget:
    """A small request-scoped budget used by external research capabilities."""

    def __init__(
        self,
        workflow: str,
        policy: CapabilityBudgetPolicy,
        *,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or f"CAP_{secrets.token_hex(8)}"
        self.workflow = workflow
        self.policy = policy
        self.usage = BudgetUsage()
        self.state = CapabilityState.CREATED
        self.state_history = [self.state.value]
        self.trace: list[TraceEvent] = []
        self.provider_usage: list[dict[str, Any]] = []
        self.termination_reason: str | None = None
        self._started = perf_counter()

    def transition(self, target: CapabilityState | str) -> None:
        target_state = CapabilityState(target)
        self.state = target_state
        self.state_history.append(target_state.value)

    def consume(self, resource: str, amount: int = 1) -> None:
        if amount < 0 or not hasattr(self.usage, resource):
            raise ValueError(f"非法预算资源：{resource}")
        current = int(getattr(self.usage, resource)) + amount
        limit = getattr(self.policy, f"max_{resource}")
        if current > limit:
            self.termination_reason = f"budget_exhausted:{resource}"
            raise BudgetExceeded(f"{resource} 预算耗尽：{current}>{limit}")
        setattr(self.usage, resource, current)

    @contextmanager
    def step(self, name: str, *, details: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        self.consume("steps")
        started = perf_counter()
        mutable = dict(details or {})
        try:
            yield mutable
        except Exception as error:
            self._append("step", name, "failed", started, mutable, error=error)
            raise
        self._append("step", name, "succeeded", started, mutable)

    def record_tool(self, name: str, status: str, elapsed_ms: float, details: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(len(self.trace) + 1, self.state.value, "tool", name, status, round(elapsed_ms, 3), _safe(details)))

    def record_provider_usage(self, stage: str, usage: dict[str, Any]) -> None:
        safe = {str(key): value for key, value in usage.items() if isinstance(value, (int, float, str, bool)) or value is None}
        safe["stage"] = stage
        self.provider_usage.append(safe)

    def recover(self, level: RecoveryLevel, action: str, *, outcome: str) -> None:
        self.trace.append(TraceEvent(len(self.trace) + 1, self.state.value, "recovery", action, outcome, 0.0, {"level": int(level)}))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "state": self.state.value,
            "state_history": list(self.state_history),
            "policy": asdict(self.policy),
            "usage": asdict(self.usage),
            "provider_usage": list(self.provider_usage),
            "trace": [asdict(item) for item in self.trace],
            "termination_reason": self.termination_reason,
            "elapsed_ms": round((perf_counter() - self._started) * 1000, 3),
        }

    def _append(self, kind: str, name: str, status: str, started: float, details: dict[str, Any], *, error: Exception | None = None) -> None:
        safe_details = _safe(details)
        if error is not None:
            safe_details["error_type"] = type(error).__name__
        self.trace.append(TraceEvent(len(self.trace) + 1, self.state.value, kind, name, status, round((perf_counter() - started) * 1000, 3), safe_details))


__all__ = [
    "BudgetExceeded",
    "CapabilityBudget",
    "CapabilityBudgetError",
    "CapabilityBudgetPolicy",
    "CapabilityState",
    "RecoveryLevel",
]
