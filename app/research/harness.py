"""Research Harness 顶层状态机、硬预算、Trace 与恢复阶梯。"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from app.storage import write_json_atomic


class HarnessState(StrEnum):
    CREATED = "created"
    CONTEXT_PREPARING = "context_preparing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED = "refused"


class RecoveryLevel(IntEnum):
    FORMAT_OR_ARGUMENT = 0
    TRANSIENT_TOOL_RETRY = 1
    BACKEND_DEGRADATION = 2
    SCOPE_REDUCTION = 3
    LIMITED_RETRIEVAL = 4
    SAFE_TERMINATION = 5


_TRANSITIONS: dict[HarnessState, frozenset[HarnessState]] = {
    HarnessState.CREATED: frozenset({HarnessState.CONTEXT_PREPARING}),
    HarnessState.CONTEXT_PREPARING: frozenset({HarnessState.PLANNING}),
    HarnessState.PLANNING: frozenset({HarnessState.EXECUTING}),
    HarnessState.EXECUTING: frozenset(
        {HarnessState.EVALUATING, HarnessState.RECOVERING, HarnessState.COMMITTING}
    ),
    HarnessState.EVALUATING: frozenset(
        {HarnessState.RECOVERING, HarnessState.COMMITTING}
    ),
    HarnessState.RECOVERING: frozenset(
        {
            HarnessState.EXECUTING,
            HarnessState.COMMITTING,
            HarnessState.FAILED,
            HarnessState.REFUSED,
        }
    ),
    HarnessState.COMMITTING: frozenset(
        {HarnessState.COMPLETED, HarnessState.FAILED, HarnessState.REFUSED}
    ),
    HarnessState.COMPLETED: frozenset(),
    HarnessState.FAILED: frozenset(),
    HarnessState.REFUSED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ResearchBudgetPolicy:
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


@dataclass(frozen=True, slots=True)
class TraceEvent:
    ordinal: int
    state: str
    kind: str
    name: str
    status: str
    elapsed_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class HarnessError(RuntimeError):
    pass


class BudgetExceeded(HarnessError):
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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


class ResearchRunHarness:
    """只感知通用状态、Step、Tool、Context、Budget、Evaluation 和 Recovery。"""

    def __init__(self, workflow: str, policy: ResearchBudgetPolicy) -> None:
        self.run_id = f"RR_{secrets.token_hex(8)}"
        self.workflow = workflow
        self.policy = policy
        self.usage = BudgetUsage()
        self.state = HarnessState.CREATED
        self.state_history: list[str] = [self.state.value]
        self.trace: list[TraceEvent] = []
        self.recovery_actions: list[dict[str, Any]] = []
        self.context_usage: dict[str, int] = {}
        self.provider_usage: list[dict[str, Any]] = []
        self.termination_reason: str | None = None
        self._started = perf_counter()

    def transition(self, target: HarnessState) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise HarnessError(f"非法 Research Harness 状态迁移：{self.state} -> {target}")
        self.state = target
        self.state_history.append(target.value)

    def consume(self, resource: str, amount: int = 1) -> None:
        if amount < 0 or not hasattr(self.usage, resource):
            raise ValueError(f"非法预算资源：{resource}")
        current = int(getattr(self.usage, resource)) + amount
        limit_name = f"max_{resource}"
        limit = getattr(self.policy, limit_name)
        if current > limit:
            self.termination_reason = f"budget_exhausted:{resource}"
            raise BudgetExceeded(f"{resource} 预算耗尽：{current}>{limit}")
        setattr(self.usage, resource, current)

    @contextmanager
    def step(self, name: str, *, details: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        if self.state not in {
            HarnessState.CONTEXT_PREPARING,
            HarnessState.PLANNING,
            HarnessState.EXECUTING,
            HarnessState.EVALUATING,
            HarnessState.RECOVERING,
            HarnessState.COMMITTING,
        }:
            raise HarnessError(f"当前状态不能执行 Step：{self.state}")
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
        self.trace.append(
            TraceEvent(
                len(self.trace) + 1,
                self.state.value,
                "tool",
                name,
                status,
                round(elapsed_ms, 3),
                _safe(details),
            )
        )

    def record_context(self, audience: str, token_count: int) -> None:
        if token_count < 0:
            raise ValueError("Context token_count 不能为负数。")
        self.consume("context_tokens", token_count)
        self.consume("total_tokens", token_count)
        self.context_usage[audience] = self.context_usage.get(audience, 0) + token_count

    def record_provider_usage(
        self, stage: str, usage: dict[str, Any]
    ) -> None:
        safe = {
            str(key): value
            for key, value in usage.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        safe["stage"] = stage
        self.provider_usage.append(safe)

    def recover(self, level: RecoveryLevel, action: str, *, outcome: str) -> None:
        if self.state != HarnessState.RECOVERING:
            raise HarnessError("Recovery action 只能在 RECOVERING 状态记录。")
        value = {"level": int(level), "action": action, "outcome": outcome}
        self.recovery_actions.append(value)
        self.trace.append(
            TraceEvent(
                len(self.trace) + 1,
                self.state.value,
                "recovery",
                action,
                outcome,
                0.0,
                {"level": int(level)},
            )
        )

    def finish(self, target: HarnessState, reason: str) -> None:
        if target not in {HarnessState.COMPLETED, HarnessState.FAILED, HarnessState.REFUSED}:
            raise ValueError("非法终止状态。")
        self.transition(target)
        self.termination_reason = reason

    def diagnostics(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "state": self.state.value,
            "state_history": list(self.state_history),
            "policy": asdict(self.policy),
            "usage": asdict(self.usage),
            "context_usage": dict(self.context_usage),
            "provider_usage": list(self.provider_usage),
            "trace": [asdict(item) for item in self.trace],
            "recovery_actions": list(self.recovery_actions),
            "termination_reason": self.termination_reason,
            "elapsed_ms": round((perf_counter() - self._started) * 1000, 3),
        }

    def _append(
        self,
        kind: str,
        name: str,
        status: str,
        started: float,
        details: dict[str, Any],
        *,
        error: Exception | None = None,
    ) -> None:
        safe_details = _safe(details)
        if error is not None:
            safe_details["error_type"] = type(error).__name__
        self.trace.append(
            TraceEvent(
                len(self.trace) + 1,
                self.state.value,
                kind,
                name,
                status,
                round((perf_counter() - started) * 1000, 3),
                safe_details,
            )
        )


class RunTraceStore:
    """每次 Run 单独持久化安全 Trace；不保存隐藏推理或完整工具日志。"""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.expanduser().resolve() / "data" / "research" / "runs"

    def write(self, harness: ResearchRunHarness, summary: dict[str, Any]) -> Path:
        path = self.root / f"{harness.run_id}.json"
        write_json_atomic(path, {**harness.diagnostics(), "summary": _safe(summary)})
        return path
