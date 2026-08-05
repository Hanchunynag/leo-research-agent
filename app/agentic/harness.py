"""Agentic RAG 的集中运行策略、有限状态机、预算与阶段轨迹。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any

from app.generation.security import redact_sensitive_text


_SENSITIVE_DETAIL_MARKERS = ("api_key", "authorization", "secret", "token")


def _safe_trace_value(key: str, value: Any) -> Any:
    if any(marker in key.lower() for marker in _SENSITIVE_DETAIL_MARKERS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): _safe_trace_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_trace_value(key, item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


class AgenticStage(StrEnum):
    """Harness 允许出现的稳定执行阶段。"""

    INITIALIZED = "initialized"
    ROUTING = "routing"
    PLANNING = "planning"
    QUERY_EXPANDING = "query_expanding"
    QUERY_VALIDATING = "query_validating"
    RETRIEVAL_DISPATCHING = "retrieval_dispatching"
    LEXICAL_RETRIEVING = "lexical_retrieving"
    DENSE_RETRIEVING = "dense_retrieving"
    GRAPH_RETRIEVING = "graph_retrieving"
    COMMUNITY_RETRIEVING = "community_retrieving"
    QUERY_FUSING = "query_fusing"
    # Backward-compatible legacy stage used by the ablation path.
    RETRIEVING = "retrieving"
    RERANKING = "reranking"
    COVERAGE_CHECKING = "coverage_checking"
    CONTEXT_BUILDING = "context_building"
    COMPACTING = "compacting"
    GENERATING = "generating"
    STRUCTURAL_VALIDATING = "structural_validating"
    SEMANTIC_VALIDATING = "semantic_validating"
    REPAIRING = "repairing"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"


class StageStatus(StrEnum):
    """单个阶段的执行结果。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TerminationReason(StrEnum):
    """一次 Agentic Run 的确定性终止原因。"""

    COMPLETED = "completed"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    GENERATION_FAILED = "generation_failed"
    SEMANTIC_VALIDATION_FAILED = "semantic_validation_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERNAL_ERROR = "internal_error"


class HarnessConstraintError(RuntimeError):
    """状态迁移或预算违反 Harness 约束。"""


@dataclass(frozen=True)
class AgenticRunPolicy:
    """与业务配置解耦的单次运行硬约束。"""

    max_query_expansion_calls: int = 1
    max_query_variants: int = 5
    max_graph_hops: int = 2
    max_graph_paths: int = 20
    max_focused_queries_per_round: int = 2
    max_cross_query_candidates: int = 40
    max_rerank_candidates: int = 20
    max_retrieval_rounds: int = 2
    max_structure_repairs: int = 1
    max_answer_repairs: int = 1
    max_total_latency_ms: int | None = None
    fail_closed: bool = True
    allow_model_downloads: bool = True

    def __post_init__(self) -> None:
        if self.max_query_expansion_calls != 1:
            raise ValueError("max_query_expansion_calls must equal 1")
        if not 1 <= self.max_query_variants <= 5:
            raise ValueError("max_query_variants must be 1..5")
        if not 1 <= self.max_graph_hops <= 2:
            raise ValueError("max_graph_hops must be 1..2")
        if not 1 <= self.max_graph_paths <= 100:
            raise ValueError("max_graph_paths must be 1..100")
        if not 1 <= self.max_focused_queries_per_round <= 2:
            raise ValueError("max_focused_queries_per_round must be 1..2")
        if not 1 <= self.max_cross_query_candidates <= 100:
            raise ValueError("max_cross_query_candidates must be 1..100")
        if not 1 <= self.max_rerank_candidates <= self.max_cross_query_candidates:
            raise ValueError("max_rerank_candidates exceeds fusion candidate budget")
        if not 1 <= self.max_retrieval_rounds <= 5:
            raise ValueError("max_retrieval_rounds 必须在 1 到 5 之间。")
        if self.max_structure_repairs not in {0, 1}:
            raise ValueError("max_structure_repairs 只能是 0 或 1。")
        if self.max_answer_repairs not in {0, 1}:
            raise ValueError("max_answer_repairs 只能是 0 或 1。")
        if self.max_total_latency_ms is not None and self.max_total_latency_ms < 1:
            raise ValueError("max_total_latency_ms 必须为空或大于 0。")

    def to_dict(self) -> dict[str, Any]:
        """返回不含秘密信息的稳定策略快照。"""

        return asdict(self)


@dataclass(frozen=True)
class StageTrace:
    """一个阶段的安全、可序列化运行记录。"""

    ordinal: int
    stage: AgenticStage
    attempt: int
    status: StageStatus
    elapsed_ms: float
    details: dict[str, Any]
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化 Trace，不包含异常正文或 Provider 秘密。"""

        return {
            "ordinal": self.ordinal,
            "stage": self.stage.value,
            "attempt": self.attempt,
            "status": self.status.value,
            "elapsed_ms": self.elapsed_ms,
            "details": {
                str(key): _safe_trace_value(str(key), value)
                for key, value in self.details.items()
            },
            "error_type": self.error_type,
        }


_ALLOWED_TRANSITIONS: dict[AgenticStage, set[AgenticStage]] = {
    AgenticStage.INITIALIZED: {AgenticStage.ROUTING},
    AgenticStage.ROUTING: {AgenticStage.PLANNING},
    AgenticStage.PLANNING: {AgenticStage.QUERY_EXPANDING, AgenticStage.RETRIEVING,
                            AgenticStage.CONTEXT_BUILDING},
    AgenticStage.QUERY_EXPANDING: {AgenticStage.QUERY_VALIDATING},
    AgenticStage.QUERY_VALIDATING: {AgenticStage.RETRIEVAL_DISPATCHING},
    AgenticStage.RETRIEVAL_DISPATCHING: {
        AgenticStage.LEXICAL_RETRIEVING, AgenticStage.DENSE_RETRIEVING,
        AgenticStage.GRAPH_RETRIEVING, AgenticStage.COMMUNITY_RETRIEVING,
        AgenticStage.QUERY_FUSING,
    },
    AgenticStage.LEXICAL_RETRIEVING: {
        AgenticStage.DENSE_RETRIEVING, AgenticStage.GRAPH_RETRIEVING,
        AgenticStage.COMMUNITY_RETRIEVING, AgenticStage.QUERY_FUSING,
    },
    AgenticStage.DENSE_RETRIEVING: {
        AgenticStage.GRAPH_RETRIEVING, AgenticStage.COMMUNITY_RETRIEVING,
        AgenticStage.QUERY_FUSING,
    },
    AgenticStage.GRAPH_RETRIEVING: {
        AgenticStage.COMMUNITY_RETRIEVING, AgenticStage.QUERY_FUSING,
    },
    AgenticStage.COMMUNITY_RETRIEVING: {AgenticStage.QUERY_FUSING},
    AgenticStage.QUERY_FUSING: {AgenticStage.RERANKING},
    AgenticStage.RETRIEVING: {AgenticStage.RERANKING},
    AgenticStage.RERANKING: {AgenticStage.COVERAGE_CHECKING},
    AgenticStage.COVERAGE_CHECKING: {
        AgenticStage.RETRIEVAL_DISPATCHING,
        AgenticStage.RETRIEVING,
        AgenticStage.CONTEXT_BUILDING,
    },
    AgenticStage.CONTEXT_BUILDING: {
        AgenticStage.COMPACTING,
        AgenticStage.GENERATING,
        AgenticStage.PERSISTING,
    },
    AgenticStage.COMPACTING: {AgenticStage.GENERATING},
    AgenticStage.GENERATING: {AgenticStage.STRUCTURAL_VALIDATING},
    AgenticStage.STRUCTURAL_VALIDATING: {AgenticStage.SEMANTIC_VALIDATING},
    AgenticStage.SEMANTIC_VALIDATING: {
        AgenticStage.RETRIEVING,
        AgenticStage.REPAIRING,
        AgenticStage.PERSISTING,
    },
    AgenticStage.REPAIRING: {AgenticStage.STRUCTURAL_VALIDATING},
    AgenticStage.PERSISTING: {
        AgenticStage.COMPLETED,
        AgenticStage.REFUSED,
        AgenticStage.FAILED,
    },
    AgenticStage.COMPLETED: set(),
    AgenticStage.REFUSED: set(),
    AgenticStage.FAILED: set(),
}


class AgenticRunHarness:
    """强制执行状态迁移和预算，并生成统一 diagnostics。"""

    def __init__(
        self,
        policy: AgenticRunPolicy,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._started = clock()
        self._state = AgenticStage.INITIALIZED
        self._traces: list[StageTrace] = []
        self._retrieval_rounds_used = 0
        self._query_expansion_calls_used = 0
        self._generated_query_count = 0
        self._structure_repairs_used = 0
        self._answer_repairs_used = 0
        self._pending_termination: TerminationReason | None = None
        self._termination_reason: TerminationReason | None = None

    @property
    def state(self) -> AgenticStage:
        """返回当前有限状态机状态。"""

        return self._state

    @property
    def retrieval_rounds_used(self) -> int:
        """返回已消费的检索轮数。"""

        return self._retrieval_rounds_used

    def elapsed_ms(self) -> float:
        """返回 Run 从创建到当前的单调时钟耗时。"""

        return round((self._clock() - self._started) * 1000, 3)

    def deadline_exceeded(self) -> bool:
        """检查总时限；超限时记录预算终止原因。"""

        limit = self.policy.max_total_latency_ms
        exceeded = limit is not None and self.elapsed_ms() >= limit
        if exceeded:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
        return exceeded

    def begin_retrieval_round(self) -> int:
        """消费一次检索预算并返回从 1 开始的轮次。"""

        if self.deadline_exceeded():
            raise HarnessConstraintError("总运行时限已耗尽。")
        if self._retrieval_rounds_used >= self.policy.max_retrieval_rounds:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
            raise HarnessConstraintError("检索轮数预算已耗尽。")
        self._retrieval_rounds_used += 1
        return self._retrieval_rounds_used

    def begin_query_expansion(self) -> int:
        if self._query_expansion_calls_used >= self.policy.max_query_expansion_calls:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
            raise HarnessConstraintError("query expansion budget exhausted")
        self._query_expansion_calls_used += 1
        return self._query_expansion_calls_used

    def record_query_variants(self, count: int, *, focused: bool = False) -> None:
        maximum = (self.policy.max_focused_queries_per_round if focused
                   else self.policy.max_query_variants)
        if count < 1 or count > maximum:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
            raise HarnessConstraintError("query variant budget exceeded")
        self._generated_query_count += count

    def can_retrieve(self) -> bool:
        """判断是否仍允许执行补充检索。"""

        return (
            self._retrieval_rounds_used < self.policy.max_retrieval_rounds
            and not self.deadline_exceeded()
        )

    def record_structure_repairs(self, count: int) -> None:
        """记录 Provider 实际结构修复次数并执行上限校验。"""

        if count < 0 or count > self.policy.max_structure_repairs:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
            raise HarnessConstraintError("结构修复次数超过 Harness 上限。")
        self._structure_repairs_used += count

    def begin_answer_repair(self) -> int:
        """消费一次答案 Repair 预算并返回次数。"""

        if self.deadline_exceeded():
            raise HarnessConstraintError("总运行时限已耗尽。")
        if self._answer_repairs_used >= self.policy.max_answer_repairs:
            self._pending_termination = TerminationReason.BUDGET_EXHAUSTED
            raise HarnessConstraintError("答案 Repair 预算已耗尽。")
        self._answer_repairs_used += 1
        return self._answer_repairs_used

    def can_repair_answer(self) -> bool:
        """判断是否仍允许执行答案 Repair。"""

        return (
            self._answer_repairs_used < self.policy.max_answer_repairs
            and not self.deadline_exceeded()
        )

    @contextmanager
    def stage(
        self,
        stage: AgenticStage,
        *,
        attempt: int = 1,
        details: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """执行一次合法阶段，并在退出时追加安全 Trace。"""

        self._transition(stage)
        started = self._clock()
        mutable_details = dict(details or {})
        try:
            yield mutable_details
        except Exception as error:
            self._traces.append(
                StageTrace(
                    ordinal=len(self._traces) + 1,
                    stage=stage,
                    attempt=attempt,
                    status=StageStatus.FAILED,
                    elapsed_ms=round((self._clock() - started) * 1000, 3),
                    details=mutable_details,
                    error_type=type(error).__name__,
                )
            )
            self._state = AgenticStage.FAILED
            self._termination_reason = TerminationReason.INTERNAL_ERROR
            raise
        self._traces.append(
            StageTrace(
                ordinal=len(self._traces) + 1,
                stage=stage,
                attempt=attempt,
                status=StageStatus.SUCCEEDED,
                elapsed_ms=round((self._clock() - started) * 1000, 3),
                details=mutable_details,
            )
        )

    def finish(self, *, answerable: bool, reason: TerminationReason) -> None:
        """从持久化阶段进入唯一终态。"""

        effective_reason = self._pending_termination or reason
        target = AgenticStage.COMPLETED if answerable else AgenticStage.REFUSED
        self._transition(target)
        self._termination_reason = effective_reason

    def diagnostics(self) -> dict[str, Any]:
        """返回面试和生产诊断可直接消费的 Harness 快照。"""

        visible_termination = self._termination_reason or self._pending_termination
        return {
            "policy": self.policy.to_dict(),
            "state": self._state.value,
            "termination_reason": (
                visible_termination.value if visible_termination is not None else None
            ),
            "budget": {
                "retrieval_rounds_used": self._retrieval_rounds_used,
                "query_expansion_calls_used": self._query_expansion_calls_used,
                "generated_query_count": self._generated_query_count,
                "structure_repairs_used": self._structure_repairs_used,
                "answer_repairs_used": self._answer_repairs_used,
                "elapsed_ms": self.elapsed_ms(),
            },
            "trace": [item.to_dict() for item in self._traces],
        }

    def _transition(self, target: AgenticStage) -> None:
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if target not in allowed:
            raise HarnessConstraintError(
                f"非法 Harness 状态迁移：{self._state.value} -> {target.value}"
            )
        self._state = target
