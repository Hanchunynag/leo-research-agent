"""兼容 Ollama、LM Studio 与 vLLM 的 OpenAI Chat Completions 适配器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.context.models import ContextBundle
from app.generation.models import AnswerClaim, AnswerDraft


SYSTEM_PROMPT = """You are a research-paper question answering component.
Use only the supplied evidence. Do not use outside knowledge.
Return exactly one JSON object and no markdown.
If the evidence is insufficient, return:
{"answerable": false, "claims": [], "refusal_reason": "specific reason"}
Otherwise return:
{"answerable": true, "claims": [{"claim_id": "C1", "text": "one atomic factual claim", "source_ids": ["S1"]}], "refusal_reason": null}
Every claim must be atomic and supported by every listed source ID. Use only source IDs
present in the evidence. Never put citation markers in claim text; the application renders them.
"""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_tokens: int = 1200
    temperature: float = 0.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必须是有效的 http(s) URL。")
        if not self.model.strip():
            raise ValueError("model 不能为空。")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")
        if self.max_tokens < 1:
            raise ValueError("max_tokens 必须大于 0。")
        if self.temperature < 0:
            raise ValueError("temperature 不能小于 0。")


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _strip_json_fence(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def _parse_answer_draft(content: str) -> AnswerDraft:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as error:
        raise ValueError("回答模型未返回合法 JSON。") from error
    if not isinstance(payload, dict):
        raise ValueError("回答模型必须返回 JSON object。")
    answerable = payload.get("answerable")
    claims_payload = payload.get("claims")
    refusal_reason = payload.get("refusal_reason")
    if not isinstance(answerable, bool):
        raise ValueError("answerable 必须是布尔值。")
    if not isinstance(claims_payload, list):
        raise ValueError("claims 必须是数组。")
    if refusal_reason is not None and not isinstance(refusal_reason, str):
        raise ValueError("refusal_reason 必须是字符串或 null。")

    claims: list[AnswerClaim] = []
    for value in claims_payload:
        if not isinstance(value, dict):
            raise ValueError("每个 claim 必须是 JSON object。")
        claim_id = value.get("claim_id")
        text = value.get("text")
        source_ids = value.get("source_ids")
        if not isinstance(claim_id, str) or not isinstance(text, str):
            raise ValueError("claim_id 和 text 必须是字符串。")
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("source_ids 必须是字符串数组。")
        claims.append(AnswerClaim(claim_id, text, source_ids))
    return AnswerDraft(answerable, claims, refusal_reason)


class OpenAICompatibleAnswerProvider:
    """调用单个 OpenAI-compatible `/v1/chat/completions` 服务。"""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model_name = config.model
        self.endpoint = _chat_completions_url(config.base_url)
        headers = (
            {"Authorization": f"Bearer {config.api_key}"}
            if config.api_key
            else None
        )
        self._client = client or httpx.Client(
            timeout=config.timeout_seconds,
            headers=headers,
        )

    def generate(self, query: str, context: ContextBundle) -> AnswerDraft:
        user_prompt = (
            f"Question:\n{query.strip()}\n\n"
            f"Evidence bundle:\n{context.context_text}\n\n"
            "Return the required JSON object."
        )
        response = self._client.post(
            self.endpoint,
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Chat Completions 响应缺少 message.content。") from error
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat Completions 返回了空 content。")
        return _parse_answer_draft(content)
