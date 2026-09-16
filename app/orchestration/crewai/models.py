"""CrewAI ``BaseLLM`` adapter for the project's provider abstraction."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, PrivateAttr

from crewai.llms.base_llm import BaseLLM
from crewai.events.types.llm_events import LLMCallType


def _message_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("role", "user")
        payload.setdefault("content", "")
        return payload
    role = str(getattr(value, "role", "user"))
    content = getattr(value, "content", value)
    return {
        "role": role,
        "content": content if isinstance(content, (str, list)) else str(content),
    }


def _tool_payload(value: Any) -> dict[str, Any] | None:
    """Convert the public CrewAI tool shape to OpenAI-compatible JSON."""

    if isinstance(value, dict):
        return value
    name = getattr(value, "name", None)
    if not name:
        return None
    schema = getattr(value, "args_schema", None)
    parameters = (
        schema.model_json_schema() if schema is not None else {"type": "object"}
    )
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(getattr(value, "description", "")),
            "parameters": parameters,
        },
    }


class CrewAIProviderAdapter(BaseLLM):
    """Wrap an existing ``chat_completion`` provider without provider coupling.

    The domain's OpenAI-compatible provider, a DeepSeek-compatible endpoint,
    Ollama gateway, or a deterministic test double can all implement the same
    small ``chat_completion`` surface.  No provider-specific HTTP is written
    in an Agent or Tool.
    """

    _completion_provider: Any = PrivateAttr()

    def __init__(
        self,
        completion_provider: Any,
        *,
        model: str | None = None,
        provider_name: str = "custom",
        **kwargs: Any,
    ) -> None:
        resolved_model = model or str(
            getattr(completion_provider, "model_name", "scholar-model")
        )
        super().__init__(model=resolved_model, provider=provider_name, **kwargs)
        self._completion_provider = completion_provider

    @classmethod
    def from_model(cls, model: Any | None) -> BaseLLM:
        if isinstance(model, BaseLLM):
            return model
        if model is None:
            return DeterministicContractLLM()
        provider = getattr(model, "provider", model)
        chat_completion = getattr(provider, "chat_completion", None)
        if callable(chat_completion):
            return cls(
                provider,
                model=str(
                    getattr(
                        model,
                        "model_name",
                        getattr(provider, "model_name", "scholar-model"),
                    )
                ),
                provider_name=str(getattr(provider, "provider_name", "custom")),
                max_tokens=getattr(provider, "config", None)
                and getattr(provider.config, "max_tokens", None),
            )
        # A small compatibility escape hatch for injected local model stubs.
        invoke = getattr(model, "invoke", None)
        if callable(invoke):
            return _InvokableModelAdapter(model)
        raise TypeError("CrewAI Provider 需要 BaseLLM、chat_completion 或 invoke。")

    def _call_provider(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> Any:
        call = self._completion_provider.chat_completion
        options: dict[str, Any] = {"max_tokens": max_tokens}
        if tools:
            options["tools"] = tools
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        try:
            return call(messages, **options)
        except TypeError as error:
            # Older deterministic test doubles intentionally expose only the
            # original two-argument contract.  They still work for structured
            # cognition; unsupported optional provider controls are dropped,
            # while the real OpenAI-compatible provider receives them.
            error_text = str(error)
            unsupported = (
                (tools and "tools" in error_text)
                or (reasoning_effort is not None and "reasoning_effort" in error_text)
                or "unexpected keyword" in error_text
            )
            if unsupported:
                fallback = {"max_tokens": max_tokens}
                return call(messages, **fallback)
            raise

    def supports_function_calling(self) -> bool:
        """Declare the capability expected by CrewAI's structured converter.

        ``crewai.llms.base_llm.BaseLLM`` in the pinned CrewAI 1.15 runtime
        does not expose this method, while the Agent/Task converter calls it
        for ``output_pydantic`` tasks.  The adapter already translates the
        public CrewAI tool schema to the provider's chat-completion shape, so
        function calling is a supported capability at this boundary.
        """

        return True

    @staticmethod
    def _usage(payload: Any) -> dict[str, Any] | None:
        """Normalize OpenAI-compatible usage before CrewAI event emission."""

        if not isinstance(payload, dict):
            return None
        raw = payload.get("usage")
        if not isinstance(raw, dict):
            return None
        usage = dict(raw)
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        if "prompt_tokens" not in usage:
            usage["prompt_tokens"] = prompt
        if "completion_tokens" not in usage:
            usage["completion_tokens"] = completion
        if "total_tokens" not in usage:
            usage["total_tokens"] = int(prompt or 0) + int(completion or 0)
        return usage

    @staticmethod
    def _finish_reason(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        return str(first.get("finish_reason")) if isinstance(first, dict) and first.get("finish_reason") else None

    @staticmethod
    def _response_id(payload: Any) -> str | None:
        if not isinstance(payload, dict) or not payload.get("id"):
            return None
        return str(payload["id"])

    @staticmethod
    def _content_and_tools(payload: Any) -> tuple[str, list[dict[str, Any]]]:
        if isinstance(payload, BaseModel):
            return payload.model_dump_json(), []
        if isinstance(payload, str):
            return payload, []
        if not isinstance(payload, dict):
            return str(payload), []
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = first.get("message") if isinstance(first, dict) else {}
        message = message if isinstance(message, dict) else {}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return "", [value for value in tool_calls if isinstance(value, dict)]
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        return str(content or ""), []

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        payload = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else [_message_payload(value) for value in messages]
        )
        normalized_tools = [
            item
            for value in (tools or ())
            if (item := _tool_payload(value)) is not None
        ]
        # Specialist structured contracts are intentionally small. The full
        # evidence/patch objects stay in the domain runtime and are projected
        # back after CrewAI validation; allowing an 8k-token completion here
        # makes a local Qwen model echo context until its JSON is truncated.
        configured_max_tokens = self.max_tokens
        if not isinstance(configured_max_tokens, int) or configured_max_tokens < 1:
            # Injected test doubles and lightweight compatibility providers do
            # not always expose a config.max_tokens value. Keep the adapter's
            # wire contract valid instead of passing None into CrewAI's
            # structured-output converter.
            configured_max_tokens = 2048
        call_max_tokens = (
            min(configured_max_tokens, 2048)
            if response_model is not None and not normalized_tools
            else configured_max_tokens
        )
        # Qwen3.5-compatible gateways may spend the whole small structured
        # completion budget on hidden reasoning and return an empty
        # ``message.content``.  The CrewAI contract is a short routing/result
        # JSON object, so disable hidden reasoning for this call.  This option
        # is only sent on the structured, no-tool path; normal generation and
        # native tool calls retain their configured behavior.
        reasoning_effort = (
            "none" if response_model is not None and not normalized_tools else None
        )
        self._emit_call_started_event(
            payload,
            normalized_tools or None,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            max_tokens=call_max_tokens,
        )
        try:
            response = self._call_provider(
                payload,
                normalized_tools or None,
                max_tokens=call_max_tokens,
                reasoning_effort=reasoning_effort,
            )
            usage = self._usage(response)
            if usage is not None:
                self._track_token_usage_internal(usage)
            content, tool_calls = self._content_and_tools(response)
            self._emit_call_completed_event(
                response=response,
                call_type=LLMCallType.TOOL_CALL if tool_calls else LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=payload,
                usage=usage,
                finish_reason=self._finish_reason(response),
                response_id=self._response_id(response),
            )
            if tool_calls and not available_functions:
                return tool_calls
            if tool_calls and available_functions:
                call = tool_calls[0]
                function = call.get("function", {}) if isinstance(call, dict) else {}
                function_name = function.get("name")
                arguments = function.get("arguments", "{}")
                if function_name in available_functions:
                    parsed = (
                        json.loads(arguments) if isinstance(arguments, str) else arguments
                    )
                    return available_functions[function_name](**parsed)
            if response_model is not None:
                return self._validate_structured_output(content, response_model)
            return content
        except Exception as error:
            self._emit_call_failed_event(str(error)[:500], from_task, from_agent)
            raise

    async def acall(self, messages: str | list[Any], **kwargs: Any) -> str | Any:
        return await asyncio.to_thread(self.call, messages, **kwargs)


class _InvokableModelAdapter(BaseLLM):
    _model_instance: Any = PrivateAttr()

    def __init__(self, model_instance: Any) -> None:
        super().__init__(
            model=str(getattr(model_instance, "model_name", "scholar-model")),
            provider="custom",
        )
        self._model_instance = model_instance

    def call(self, messages: str | list[Any], **_: Any) -> str:
        result = self._model_instance.invoke(messages)
        return str(getattr(result, "content", result))

    def supports_function_calling(self) -> bool:
        """Keep injected invoke-compatible models valid for CrewAI Agents."""

        return False


class DeterministicContractLLM(BaseLLM):
    """No-network fallback used by architecture tests and local dry-runs."""

    def __init__(self) -> None:
        super().__init__(model="deterministic-contract-model", provider="local")

    def supports_function_calling(self) -> bool:
        """Expose the CrewAI capability probe for offline contract runs."""

        return True

    def call(
        self,
        messages: str | list[Any],
        response_model: type[BaseModel] | None = None,
        **_: Any,
    ) -> str | Any:
        text = (
            messages
            if isinstance(messages, str)
            else "\n".join(
                str(_message_payload(value).get("content", "")) for value in messages
            )
        )
        marker = text.casefold()
        if "selected_route" in marker or "routing decision" in marker:
            route = "RESEARCH"
            # The Flow supplies the deterministic route hint.  Prefer it over
            # route names that may also occur in the generated JSON schema.
            for key in ("deterministic_route", "allowed_route"):
                marker_start = marker.find(key)
                if marker_start >= 0:
                    tail = marker[marker_start : marker_start + 100]
                    for candidate in (
                        "SUPPORT_CLAIM",
                        "WRITE_INTRODUCTION",
                        "WRITE_CONCLUSION",
                        "WRITE_ABSTRACT",
                        "REVIEW",
                        "RESEARCH",
                    ):
                        if candidate.casefold() in tail:
                            route = candidate
                            break
                    if route != "RESEARCH" or "research" in tail:
                        break
            payload: dict[str, Any] = {
                "status": "COMPLETED",
                "selected_route": route,
                "approval_required": False,
            }
        elif "writer agent" in marker:
            payload = {
                "status": "READY",
                "research_required": False,
                "missing_context": [],
                "used_evidence": [],
                "warnings": [],
            }
        elif "reviewer agent" in marker or (
            "decision" in marker and "review" in marker
        ):
            payload = {
                "decision": "PASS",
                "issues": [],
                "unsupported_claims": [],
                "citation_issues": [],
                "fact_conflicts": [],
                "revision_instructions": [],
            }
        elif "draft_patch" in marker:
            payload = {
                "status": "READY",
                "research_required": False,
                "missing_context": [],
                "used_evidence": [],
                "warnings": [],
            }
        else:
            payload = {
                "status": "COMPLETED",
                "research_summary": "Deterministic contract acknowledgement",
                "evidence": [],
                "citations": [],
                "unresolved_claims": [],
                "contradictions": [],
                "freshness_status": "UNKNOWN",
            }
        if response_model is not None:
            return response_model.model_validate(payload)
        return json.dumps(payload, ensure_ascii=False)
