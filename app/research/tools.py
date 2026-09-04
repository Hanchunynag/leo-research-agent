"""唯一 Tool Gateway：声明式注册、权限、预算、超时、重试和副作用控制。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

from app.research.harness import HarnessState, RecoveryLevel, ResearchRunHarness


SideEffect = Literal["none", "read", "bounded_write", "external_write"]


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


ToolHandler = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


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


class ToolGatewayRegistry:
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

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        harness: ResearchRunHarness,
    ) -> Mapping[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"未注册 Tool：{name}")
        workflow = str(context.get("workflow") or "")
        permissions = set(context.get("permissions") or ())
        if workflow not in spec.allowed_workflows:
            raise PermissionError(f"Workflow {workflow} 不允许调用 {name}")
        if spec.permission not in permissions:
            raise PermissionError(f"缺少 Tool 权限：{spec.permission}")
        if "workspace_id" in arguments and str(arguments["workspace_id"]) != str(
            context.get("workspace_id")
        ):
            raise PermissionError("Tool workspace_id 与 Run 不一致。")
        if "scope_version" in arguments and int(arguments["scope_version"]) != int(
            context.get("scope_version") or 0
        ):
            raise PermissionError("Tool scope_version 与 Run 不一致。")
        handler = self._handlers.get(name)
        if handler is None:
            raise RuntimeError(f"Tool handler 未配置：{name}")
        _validate(spec.input_schema, arguments, "input")
        harness.consume("tool_calls")
        if name == "knowledge.retrieve":
            harness.consume("retrieval_rounds")
        if name == "literature.search":
            harness.consume("external_searches")
        started = perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, spec.retry_policy.max_attempts + 1):
            try:
                result = handler(dict(arguments), dict(context))
                elapsed = (perf_counter() - started) * 1000
                if elapsed > spec.timeout_seconds * 1000:
                    raise TimeoutError(f"Tool 超时：{name}")
                if not isinstance(result, Mapping):
                    raise TypeError("Tool 输出必须是对象。")
                _validate(spec.output_schema, result, "output")
                harness.record_tool(name, "succeeded", elapsed, {"attempt": attempt})
                return dict(result)
            except Exception as error:
                last_error = error
                retryable = type(error).__name__ in spec.retry_policy.retryable_errors
                if retryable and attempt < spec.retry_policy.max_attempts:
                    harness.record_tool(
                        name,
                        "retrying",
                        (perf_counter() - started) * 1000,
                        {
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "recovery_level": int(RecoveryLevel.TRANSIENT_TOOL_RETRY),
                        },
                    )
                    harness.transition(HarnessState.RECOVERING)
                    harness.recover(
                        RecoveryLevel.TRANSIENT_TOOL_RETRY,
                        f"retry:{name}",
                        outcome="retrying",
                    )
                    harness.transition(HarnessState.EXECUTING)
                    continue
                if not retryable or attempt >= spec.retry_policy.max_attempts:
                    harness.record_tool(
                        name,
                        "failed",
                        (perf_counter() - started) * 1000,
                        {"attempt": attempt, "error_type": type(error).__name__},
                    )
                    raise
        assert last_error is not None
        raise last_error

    async def ainvoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        harness: ResearchRunHarness,
    ) -> Mapping[str, Any]:
        """短工具使用真正的异步截止时间；长工具 handler 只提交 Job。"""

        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"未注册 Tool：{name}")
        return await asyncio.wait_for(
            asyncio.to_thread(
                self.invoke,
                name,
                arguments,
                context=context,
                harness=harness,
            ),
            timeout=spec.timeout_seconds,
        )


_ALL = frozenset({"direct_qa", "relation_reasoning", "deep_research", "research_bootstrap"})
_DEEP_BOOTSTRAP = frozenset({"deep_research", "research_bootstrap"})
_BOOTSTRAP = frozenset({"research_bootstrap"})


def default_tool_specs() -> tuple[ToolSpec, ...]:
    object_schema = {"type": "object"}
    return (
        # Local BGE-M3 + Cross Encoder model loading can exceed 30 seconds on
        # CPU/MPS cold start.  The timeout covers one idempotent retrieval
        # attempt; subsequent requests reuse the providers in the process.
        ToolSpec("knowledge.retrieve", {"required": ["query", "workspace_id", "scope_version"], "properties": {"query": {"type": "string"}, "workspace_id": {"type": "string"}, "scope_version": {"type": "integer"}}}, object_schema, "knowledge.read", 120.0, True, "read", _ALL, RetryPolicy(2)),
        ToolSpec("workspace.read_scope", {"required": ["workspace_id", "scope_version"]}, object_schema, "workspace.read", 5.0, True, "read", _ALL),
        ToolSpec("workspace.update_scope", {"required": ["workspace_id", "scope_version", "document_ids"], "properties": {"workspace_id": {"type": "string"}, "scope_version": {"type": "integer"}, "document_ids": {"type": "array"}}}, object_schema, "workspace.write", 10.0, True, "bounded_write", _BOOTSTRAP),
        ToolSpec("literature.search", {"required": ["query"]}, object_schema, "literature.search", 30.0, True, "read", _DEEP_BOOTSTRAP, RetryPolicy(2)),
        ToolSpec("literature.get_metadata", {"required": ["paper_id"]}, object_schema, "literature.read", 15.0, True, "read", _DEEP_BOOTSTRAP, RetryPolicy(2)),
        ToolSpec("literature.resolve_publication_date", {"required": ["title"], "properties": {"title": {"type": "string"}}}, object_schema, "literature.read", 30.0, True, "read", _ALL, RetryPolicy(2)),
        ToolSpec("literature.download", {"required": ["paper_id"]}, object_schema, "literature.download", 60.0, True, "external_write", _BOOTSTRAP, RetryPolicy(2)),
        ToolSpec("document.parse", {"required": ["path", "mode"]}, object_schema, "document.parse", 300.0, True, "bounded_write", _BOOTSTRAP),
        ToolSpec("job.get_status", {"required": ["job_id"]}, object_schema, "job.read", 5.0, True, "read", _ALL),
    )


def build_default_gateway(handlers: Mapping[str, ToolHandler] | None = None) -> ToolGatewayRegistry:
    gateway = ToolGatewayRegistry()
    values = handlers or {}
    for spec in default_tool_specs():
        gateway.register(spec, values.get(spec.name))
    return gateway
