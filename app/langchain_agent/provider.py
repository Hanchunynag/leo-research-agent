"""LangChain 1.x ChatModel adapter for the existing OpenAI-compatible provider.

The Scholar domain never receives this type.  It translates LangChain messages
and tool schemas at the framework boundary, while the existing HTTP provider
remains responsible for transport, authentication and response parsing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, PrivateAttr


def _content(value: Any) -> str | list[dict[str, Any]]:
    """Keep text blocks and multimodal blocks accepted by OpenAI-compatible APIs."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return str(value)


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    role = "user"
    if isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif message.type == "system":
        role = "system"
    payload: dict[str, Any] = {"role": role, "content": _content(message.content)}
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    if isinstance(message, AIMessage):
        tool_calls = message.tool_calls
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": str(call.get("id") or f"call_{index}"),
                    "type": "function",
                    "function": {
                        "name": str(call.get("name") or ""),
                        "arguments": call.get("args") or {},
                    },
                }
                for index, call in enumerate(tool_calls, 1)
            ]
    return payload


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, Any]] = []
    for index, value in enumerate(raw, 1):
        if not isinstance(value, dict):
            continue
        function = value.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(
            {
                "name": str(function["name"]),
                "args": arguments if isinstance(arguments, dict) else {},
                "id": str(value.get("id") or f"call_{index}"),
                "type": "tool_call",
            }
        )
    return calls


class OpenAICompatibleChatModel(BaseChatModel):
    """Thin LangChain 1.x adapter over ``chat_completion``.

    ``bind_tools`` returns another adapter carrying only serialized tool
    schemas.  It does not move ToolGateway or domain operations into LangChain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    provider: Any
    model_name: str
    max_tokens: int = 8192
    temperature: float = 0.0
    _bound_tools: tuple[dict[str, Any], ...] = PrivateAttr(default=())
    _tool_choice: str | dict[str, Any] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "openai_compatible_chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **_: Any,
    ) -> "OpenAICompatibleChatModel":
        bound = self.model_copy(deep=False)
        bound._bound_tools = tuple(convert_to_openai_tool(tool) for tool in tools)
        bound._tool_choice = tool_choice
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **_: Any,
    ) -> ChatResult:
        payload = [_message_payload(message) for message in messages]
        options: dict[str, Any] = {}
        if self._bound_tools:
            options["tools"] = list(self._bound_tools)
        if self._tool_choice is not None:
            options["tool_choice"] = self._tool_choice
        response = self.provider.chat_completion(
            payload,
            max_tokens=self.max_tokens,
            **options,
        )
        choices = response.get("choices") if isinstance(response, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else {}
        raw_message = first.get("message") if isinstance(first, dict) else {}
        raw_message = raw_message if isinstance(raw_message, dict) else {}
        usage = response.get("usage") if isinstance(response, dict) else None
        safe_metadata = {
            key: value
            for key, value in (usage or {}).items()
            if isinstance(usage, dict)
            and isinstance(key, str)
            and isinstance(value, (str, int, float, bool))
        }
        message = AIMessage(
            content=raw_message.get("content") or "",
            tool_calls=_tool_calls(raw_message),
            response_metadata=safe_metadata,
        )
        generation_info = {
            "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
            "usage": usage,
        }
        return ChatResult(
            generations=[ChatGeneration(message=message, generation_info=generation_info)]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._generate(messages, stop, **kwargs)
        generation = result.generations[0]
        message = generation.message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_calls=getattr(message, "tool_calls", []),
            )
        )


class ScholarChatModelAdapter(BaseChatModel):
    """Normalize an arbitrary LangChain 1.x model for the Scholar harness.

    Tests and deployments may already provide a first-party ``BaseChatModel``.
    Wrapping it gives Deep Agents one stable provider identity, which lets the
    harness profile remove unsafe builtin tools for both production and test
    models without touching the model's implementation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inner: BaseChatModel
    model_name: str = "scholar-model"
    _bound_inner: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "scholar_chat_model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "inner_type": type(self.inner).__name__}

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScholarChatModelAdapter":
        binder = getattr(self.inner, "bind_tools", None)
        if not callable(binder):
            raise TypeError("Deep Agents 所需的 BaseChatModel 不支持 bind_tools。")
        bound = self.model_copy(deep=False)
        bound._bound_inner = binder(tools, tool_choice=tool_choice, **kwargs)
        return bound

    def _active(self) -> Any:
        return self._bound_inner or self.inner

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        active = self._active()
        output = active.invoke(messages, stop=stop, **kwargs)
        if isinstance(output, ChatResult):
            return output
        message = output if isinstance(output, BaseMessage) else AIMessage(content=str(output))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        active = self._active()
        ainvoke = getattr(active, "ainvoke", None)
        if callable(ainvoke):
            output = await ainvoke(messages, stop=stop, **kwargs)
            if isinstance(output, ChatResult):
                return output
            message = output if isinstance(output, BaseMessage) else AIMessage(content=str(output))
            return ChatResult(generations=[ChatGeneration(message=message)])
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        active = self._active()
        stream = getattr(active, "stream", None)
        if callable(stream):
            for chunk in stream(messages, stop=stop, **kwargs):
                if isinstance(chunk, AIMessageChunk):
                    yield ChatGenerationChunk(message=chunk)
                elif isinstance(chunk, BaseMessage):
                    yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.content))
            return
        yield from super()._stream(messages, stop=stop, **kwargs)
