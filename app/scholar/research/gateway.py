"""Validated Tool Gateway for Scholar's external research capabilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

from app.scholar.research.budget import CapabilityBudget, RecoveryLevel

SideEffect = Literal["none", "read", "bounded_write", "external_write"]
ToolHandler = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    retryable_errors: tuple[str, ...] = ("TimeoutError", "ConnectionError")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    permission: str
    timeout_seconds: float
    idempotent: bool
    side_effect: SideEffect
    allowed_workflows: frozenset[str]
    retry_policy: RetryPolicy = RetryPolicy()


def _validate(schema: Mapping[str, Any], value: Mapping[str, Any], label: str) -> None:
    required = schema.get("required")
    if isinstance(required, list):
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{label} 缺少字段：{missing}")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return
    types = {"string": str, "integer": int, "boolean": bool, "array": list, "object": Mapping}
    for key, rule in properties.items():
        if key not in value or not isinstance(rule, Mapping) or "type" not in rule:
            continue
        expected = types.get(str(rule["type"]))
        if expected is not None and not isinstance(value[key], expected):
            raise ValueError(f"{label}.{key} 类型错误。")


class ToolGateway:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler | None = None) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool 已注册：{spec.name}")
        self._specs[spec.name] = spec
        if handler is not None:
            self._handlers[spec.name] = handler

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def invoke(self, name: str, arguments: Mapping[str, Any], *, context: Mapping[str, Any], budget: CapabilityBudget) -> Mapping[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"未注册 Tool：{name}")
        workflow = str(context.get("workflow") or "")
        permissions = set(context.get("permissions") or ())
        if workflow not in spec.allowed_workflows:
            raise PermissionError(f"Workflow {workflow} 不允许调用 {name}")
        if spec.permission not in permissions:
            raise PermissionError(f"缺少 Tool 权限：{spec.permission}")
        if "workspace_id" in arguments and str(arguments["workspace_id"]) != str(context.get("workspace_id")):
            raise PermissionError("Tool workspace_id 与 Run 不一致。")
        if "scope_version" in arguments and int(arguments["scope_version"]) != int(context.get("scope_version") or 0):
            raise PermissionError("Tool scope_version 与 Run 不一致。")
        handler = self._handlers.get(name)
        if handler is None:
            raise RuntimeError(f"Tool handler 未配置：{name}")
        _validate(spec.input_schema, arguments, "input")
        budget.consume("tool_calls")
        if name == "knowledge.retrieve":
            budget.consume("retrieval_rounds")
        if name == "literature.search":
            budget.consume("external_searches")
        started = perf_counter()
        for attempt in range(1, spec.retry_policy.max_attempts + 1):
            try:
                result = handler(dict(arguments), dict(context))
                elapsed = (perf_counter() - started) * 1000
                if elapsed > spec.timeout_seconds * 1000:
                    raise TimeoutError(f"Tool 超时：{name}")
                if not isinstance(result, Mapping):
                    raise TypeError("Tool 输出必须是对象。")
                _validate(spec.output_schema, result, "output")
                budget.record_tool(name, "succeeded", elapsed, {"attempt": attempt})
                return dict(result)
            except Exception as error:
                retryable = type(error).__name__ in spec.retry_policy.retryable_errors
                if retryable and attempt < spec.retry_policy.max_attempts:
                    budget.record_tool(name, "retrying", (perf_counter() - started) * 1000, {"attempt": attempt, "error_type": type(error).__name__, "recovery_level": int(RecoveryLevel.TRANSIENT_TOOL_RETRY)})
                    budget.transition("recovering")
                    budget.recover(RecoveryLevel.TRANSIENT_TOOL_RETRY, f"retry:{name}", outcome="retrying")
                    budget.transition("executing")
                    continue
                budget.record_tool(name, "failed", (perf_counter() - started) * 1000, {"attempt": attempt, "error_type": type(error).__name__})
                raise
        raise RuntimeError(f"Tool 未完成：{name}")

    async def ainvoke(self, name: str, arguments: Mapping[str, Any], *, context: Mapping[str, Any], budget: CapabilityBudget) -> Mapping[str, Any]:
        spec = self._specs[name]
        return await asyncio.wait_for(asyncio.to_thread(self.invoke, name, arguments, context=context, budget=budget), timeout=spec.timeout_seconds)


_ALL = frozenset({"scholar_research"})
_SCHOLAR = _ALL


def default_tool_specs() -> tuple[ToolSpec, ...]:
    object_schema = {"type": "object"}
    return (
        ToolSpec("literature.search", {"required": ["query"], "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "year_from": {"type": "integer"}, "year_to": {"type": "integer"}}}, object_schema, "literature.search", 30.0, True, "read", _SCHOLAR, RetryPolicy(2)),
        ToolSpec("literature.get_metadata", {"required": ["paper_id"]}, object_schema, "literature.read", 15.0, True, "read", _SCHOLAR, RetryPolicy(2)),
        ToolSpec("literature.resolve_publication_date", {"required": ["title"], "properties": {"title": {"type": "string"}}}, object_schema, "literature.read", 30.0, True, "read", _SCHOLAR, RetryPolicy(2)),
        ToolSpec("job.get_status", {"required": ["job_id"]}, object_schema, "job.read", 5.0, True, "read", _ALL),
    )


def build_default_gateway(handlers: Mapping[str, ToolHandler] | None = None) -> ToolGateway:
    gateway = ToolGateway()
    values = handlers or {}
    for spec in default_tool_specs():
        gateway.register(spec, values.get(spec.name))
    return gateway


__all__ = ["ToolGateway", "ToolHandler", "ToolSpec", "build_default_gateway"]
