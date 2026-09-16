"""CrewAI event bridge into the existing Scholar run trace projection."""

from __future__ import annotations

import contextvars
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Any, Iterator

from crewai.events import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


@dataclass(frozen=True, slots=True)
class CrewTraceEvent:
    name: str
    kind: str
    status: str
    timestamp: str
    elapsed_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_active_trace: contextvars.ContextVar["CrewAITraceAdapter | None"] = (
    contextvars.ContextVar("scholar_crewai_trace", default=None)
)
_handlers_installed = False


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(child) for child in value]
    if hasattr(value, "model_dump"):
        try:
            return _safe(value.model_dump(mode="json", warnings="none"))
        except Exception:
            return type(value).__name__
    return str(value)[:500]


def _event_metadata(event: Any) -> dict[str, Any]:
    fields = (
        "event_id",
        "task_id",
        "task_name",
        "agent_id",
        "agent_role",
        "tool_name",
        "flow_name",
        "method_name",
        "crew_name",
        "total_tokens",
        "call_type",
        "usage",
    )
    return {
        field: _safe(getattr(event, field))
        for field in fields
        if getattr(event, field, None) is not None
    }


def _timer_key(
    name: str,
    kind: str,
    metadata: dict[str, Any],
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Build a stable key shared by CrewAI start/completion event pairs.

    ``event_id`` is deliberately excluded because CrewAI assigns a different
    ID to the start and completion event. The remaining identifiers are the
    operation identity exposed by the corresponding event family.
    """

    normalized_name = str(name).casefold()
    for suffix in ("startedevent", "completedevent", "finishedevent", "failedevent"):
        if normalized_name.endswith(suffix):
            normalized_name = normalized_name[: -len(suffix)]
            break
    for suffix in ("_started", "_completed", "_finished", "_failed"):
        if normalized_name.endswith(suffix):
            normalized_name = normalized_name[: -len(suffix)]
            break

    normalized_kind = str(kind).casefold()
    fields_by_kind = {
        "crew": ("crew_name",),
        "task": ("task_id", "task_name"),
        "agent": ("agent_id", "agent_role"),
        "tool": ("tool_name",),
        "llm": ("task_id", "agent_id", "agent_role", "task_name"),
        "flow": ("flow_name", "method_name"),
    }
    identity = tuple(
        (field, str(metadata[field]))
        for field in fields_by_kind.get(normalized_kind, ())
        if metadata.get(field) is not None
    )
    if not identity:
        # Explicit Flow trace records use these lightweight identifiers rather
        # than CrewAI's typed event fields.
        identity = tuple(
            (field, str(metadata[field]))
            for field in ("tool_name", "agent", "task")
            if metadata.get(field) is not None
        )
    return normalized_kind, normalized_name, identity


class CrewAITraceAdapter:
    """Collect framework diagnostics under an existing run correlation ID.

    The adapter never invents evaluator events.  Every framework event below
    comes from CrewAI's global event bus; the Flow's explicit transitions are
    recorded alongside them as business events.
    """

    def __init__(
        self,
        *,
        project_id: str,
        session_id: str | None,
        run_id: str,
        trace_id: str,
        event_sink: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self.session_id = session_id
        self.run_id = run_id
        self.trace_id = trace_id
        self.events: list[CrewTraceEvent] = []
        self._record_lock = RLock()
        # CrewAI emits separate start and completion events. Keep a FIFO of
        # monotonic start times per logical operation so repeated calls (for
        # example multiple LLM calls in one Task) are paired in order.
        self._active_timers: dict[tuple[str, str, tuple[tuple[str, str], ...]], deque[float]] = defaultdict(deque)
        self.agent_calls = 0
        self.tool_calls = 0
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost: float | None = None
        # The sink is owned by Project Runtime (normally RunEventStore).  It
        # is intentionally a callback so this adapter remains usable by the
        # existing in-process tests and does not make CrewAI the durability
        # authority.
        self.event_sink = event_sink
        _install_handlers()

    @contextmanager
    def activate(self) -> Iterator["CrewAITraceAdapter"]:
        token = _active_trace.set(self)
        try:
            yield self
        finally:
            _active_trace.reset(token)

    def record(
        self,
        name: str,
        status: str,
        metadata: dict[str, Any] | None = None,
        *,
        kind: str = "step",
    ) -> None:
        # Research progress callbacks may originate from bounded coverage
        # workers when an operator explicitly enables parallel retrieval.
        # Keep the in-memory trace and timer pairing coherent as well as the
        # SQLite event projection.
        with self._record_lock:
            self._record_locked(name, status, metadata, kind=kind)

    def _record_locked(
        self,
        name: str,
        status: str,
        metadata: dict[str, Any] | None = None,
        *,
        kind: str = "step",
    ) -> None:
        # Capability calls are intentionally executed by the deterministic
        # Flow boundary rather than selected by an LLM.  They therefore do
        # not produce CrewAI ToolUsage events, but they are still real tool
        # invocations and must be included in the product trace accounting.
        if kind == "tool" and name == "capability_tool_call" and status == "RUNNING":
            self.tool_calls += 1
        normalized_status = str(status).upper()
        event_metadata = {
            "project_id": self.project_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            **dict(metadata or {}),
        }
        timer_key = _timer_key(name, kind, event_metadata)
        elapsed_ms: float | None = None
        now = perf_counter()
        if normalized_status == "RUNNING":
            self._active_timers[timer_key].append(now)
        elif normalized_status in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            starts = self._active_timers.get(timer_key)
            if starts:
                elapsed_ms = round(max(0.0, now - starts.popleft()) * 1000, 3)
                if not starts:
                    self._active_timers.pop(timer_key, None)
        event = CrewTraceEvent(
            name=name,
            kind=kind,
            status=normalized_status,
            timestamp=datetime.now(UTC).isoformat(),
            elapsed_ms=elapsed_ms,
            metadata=event_metadata,
        )
        self.events.append(event)
        if self.event_sink is not None:
            try:
                self.event_sink(asdict(event))
            except Exception:
                # A telemetry projection failure must not change the domain
                # result.  The durable result projection remains authoritative
                # and the caller can expose the failure through diagnostics.
                pass
    def transition(self, state: str, *, route: str | None = None) -> None:
        self.record(
            "flow_transition",
            "COMPLETED",
            {"state": state, **({"route": route} if route else {})},
            kind="flow",
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "crewai",
            "project_id": self.project_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "agent_call_count": self.agent_calls,
            "tool_call_count": self.tool_calls,
            "llm_call_count": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "cost": self.cost,
            "events": [asdict(event) for event in self.events],
            "trace": [asdict(event) for event in self.events],
        }


def _record_framework_event(event: Any, *, kind: str, status: str) -> None:
    adapter = _active_trace.get()
    if adapter is None:
        return
    event_name = type(event).__name__
    metadata = _event_metadata(event)
    if kind == "agent":
        if status == "RUNNING":
            adapter.agent_calls += 1
    if kind == "tool" and status == "RUNNING":
        adapter.tool_calls += 1
    if kind == "llm":
        if status == "RUNNING":
            adapter.llm_calls += 1
        usage = getattr(event, "usage", None)
        if isinstance(usage, dict):
            prompt_tokens = int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            adapter.prompt_tokens += prompt_tokens
            adapter.completion_tokens += completion_tokens
            adapter.total_tokens += int(
                usage.get("total_tokens") or prompt_tokens + completion_tokens
            )
            reported_cost = next(
                (
                    usage.get(field)
                    for field in ("cost", "total_cost", "total_cost_usd")
                    if usage.get(field) is not None
                ),
                None,
            )
            if reported_cost is not None:
                adapter.cost = (adapter.cost or 0.0) + float(reported_cost)
    if kind == "crew" and status == "COMPLETED" and not adapter.total_tokens:
        # CrewAI 1.15.21 also exposes the aggregate usage on the kickoff
        # completion event.  Use it only as a fallback when a provider did
        # not return per-call usage; otherwise summing both would double
        # count the same tokens.
        total_tokens = getattr(event, "total_tokens", 0)
        if isinstance(total_tokens, int) and total_tokens > 0:
            adapter.total_tokens = total_tokens
    adapter.record(event_name, status, metadata, kind=kind)


def _install_handlers() -> None:
    global _handlers_installed  # noqa: PLW0603
    if _handlers_installed:
        return
    _handlers_installed = True

    @crewai_event_bus.on(FlowStartedEvent)
    def _flow_started(_: Any, event: FlowStartedEvent) -> None:
        _record_framework_event(event, kind="flow", status="RUNNING")

    @crewai_event_bus.on(FlowFinishedEvent)
    def _flow_finished(_: Any, event: FlowFinishedEvent) -> None:
        _record_framework_event(event, kind="flow", status="COMPLETED")

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def _method_started(_: Any, event: MethodExecutionStartedEvent) -> None:
        _record_framework_event(event, kind="flow", status="RUNNING")

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _agent_started(_: Any, event: AgentExecutionStartedEvent) -> None:
        _record_framework_event(event, kind="agent", status="RUNNING")

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _agent_completed(_: Any, event: AgentExecutionCompletedEvent) -> None:
        _record_framework_event(event, kind="agent", status="COMPLETED")

    @crewai_event_bus.on(TaskStartedEvent)
    def _task_started(_: Any, event: TaskStartedEvent) -> None:
        _record_framework_event(event, kind="task", status="RUNNING")

    @crewai_event_bus.on(TaskCompletedEvent)
    def _task_completed(_: Any, event: TaskCompletedEvent) -> None:
        _record_framework_event(event, kind="task", status="COMPLETED")

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _tool_started(_: Any, event: ToolUsageStartedEvent) -> None:
        _record_framework_event(event, kind="tool", status="RUNNING")

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _tool_finished(_: Any, event: ToolUsageFinishedEvent) -> None:
        _record_framework_event(event, kind="tool", status="COMPLETED")

    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def _crew_started(_: Any, event: CrewKickoffStartedEvent) -> None:
        _record_framework_event(event, kind="crew", status="RUNNING")

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def _crew_completed(_: Any, event: CrewKickoffCompletedEvent) -> None:
        _record_framework_event(event, kind="crew", status="COMPLETED")

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _llm_started(_: Any, event: LLMCallStartedEvent) -> None:
        _record_framework_event(event, kind="llm", status="RUNNING")

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _llm_completed(_: Any, event: LLMCallCompletedEvent) -> None:
        _record_framework_event(event, kind="llm", status="COMPLETED")

    @crewai_event_bus.on(LLMCallFailedEvent)
    def _llm_failed(_: Any, event: LLMCallFailedEvent) -> None:
        _record_framework_event(event, kind="llm", status="FAILED")
