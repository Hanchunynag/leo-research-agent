from __future__ import annotations

import asyncio
from typing import Any

from app.knowledge_engine import build_lightrag_client_config


class Embeddings:
    model_name = "fixture-embedding"

    def embed_query(self, value: str) -> list[float]:
        return [1.0, 2.0]

    def embed_documents(self, values: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] for _ in values]


class Completion:
    def chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "structured result"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }


def test_model_bridge_tracks_embedding_and_llm_usage() -> None:
    config, usage = build_lightrag_client_config(
        Embeddings(),
        Completion(),
        llm_model_name="fixture-llm",
        embedding_dimension=2,
    )

    vectors = asyncio.run(config.embedding_func(["one", "two"], context="document"))
    content = asyncio.run(config.llm_model_func("prompt", system_prompt="system"))

    assert vectors.shape == (2, 2)
    assert content == "structured result"
    assert usage.to_dict() == {
        "llm_calls": 1,
        "llm_failures": 0,
        "llm_retries": 0,
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "embedding_calls": 1,
        "embedded_texts": 2,
    }
