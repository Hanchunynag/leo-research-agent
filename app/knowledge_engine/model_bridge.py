"""现有本地 Embedding/LLM Provider 到 LightRAG 1.5.6 的窄桥接。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import numpy as np

from app.knowledge_engine.lightrag_engine import LightRAGClientConfig


@dataclass
class LightRAGUsage:
    llm_calls: int = 0
    llm_failures: int = 0
    llm_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_calls: int = 0
    embedded_texts: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "llm_retries": self.llm_retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "embedding_calls": self.embedding_calls,
            "embedded_texts": self.embedded_texts,
        }


def build_lightrag_client_config(
    embedding_provider: Any,
    completion_provider: Any,
    *,
    llm_model_name: str,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
    usage: LightRAGUsage | None = None,
    options: Mapping[str, Any] | None = None,
) -> tuple[LightRAGClientConfig, LightRAGUsage]:
    """复用已有 Provider，不让 LightRAG 配置泄漏到 Agent。"""

    from lightrag.utils import EmbeddingFunc  # type: ignore[import-untyped]

    metrics = usage or LightRAGUsage()
    if embedding_dimension is None:
        embedding_dimension = len(embedding_provider.embed_query("dimension probe"))
    model_name = embedding_model_name or str(getattr(embedding_provider, "model_name", "legacy-embedding"))

    async def embed(texts: list[str], context: str | None = None) -> np.ndarray:
        metrics.embedding_calls += 1
        metrics.embedded_texts += len(texts)

        def compute() -> list[list[float]]:
            if context == "query":
                return [embedding_provider.embed_query(value) for value in texts]
            return embedding_provider.embed_documents(texts)

        return np.asarray(await asyncio.to_thread(compute), dtype=np.float32)

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for value in history_messages or []:
            role = value.get("role")
            content = value.get("content")
            if isinstance(role, str) and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        import httpx

        payload: dict[str, Any] | None = None
        for attempt in range(3):
            metrics.llm_calls += 1
            try:
                payload = await asyncio.to_thread(completion_provider.chat_completion, messages)
                break
            except httpx.TransportError:
                metrics.llm_failures += 1
                if attempt == 2:
                    raise
                metrics.llm_retries += 1
                await asyncio.sleep(2**attempt)
        assert payload is not None
        raw_usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(raw_usage, dict):
            metrics.prompt_tokens += int(raw_usage.get("prompt_tokens") or 0)
            metrics.completion_tokens += int(raw_usage.get("completion_tokens") or 0)
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LightRAG LLM bridge 未收到 choices。")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("LightRAG LLM bridge 未收到文本 content。")
        return content

    return (
        LightRAGClientConfig(
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                func=embed,
                model_name=model_name,
                supports_asymmetric=True,
            ),
            llm_model_func=complete,
            llm_model_name=llm_model_name,
            options=options,
        ),
        metrics,
    )
