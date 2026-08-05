"""兼容 Ollama、LM Studio 与 vLLM 的 OpenAI Chat Completions 适配器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from app.context.models import ContextBundle
from app.generation.models import AnswerClaim, AnswerDraft
from app.indexing.tokenization import token_count


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

PromptLayout = Literal["query_first", "context_first"]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 120.0
    max_tokens: int = 8192
    temperature: float = 0.0
    prompt_layout: PromptLayout = "query_first"
    json_mode: bool = True

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
        if self.prompt_layout not in {"query_first", "context_first"}:
            raise ValueError("prompt_layout 必须是 query_first 或 context_first。")
        if not isinstance(self.json_mode, bool):
            raise ValueError("json_mode 必须是布尔值。")


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


def parse_json_object(content: str) -> dict[str, Any]:
    """解析唯一 JSON object，并容忍常见的 Markdown 或简短前后缀。"""

    cleaned = _strip_json_fence(content)
    decoder = json.JSONDecoder()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        payload = None
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            raise ValueError(
                "回答模型未返回可解析的 JSON object（"
                f"line={original_error.lineno}, column={original_error.colno}）。"
            ) from original_error
    if not isinstance(payload, dict):
        raise ValueError("回答模型必须返回 JSON object。")
    return payload


def _parse_answer_draft(content: str) -> AnswerDraft:
    payload = parse_json_object(content)
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
        category = value.get("category")
        evidence_ids = value.get("evidence_ids", [])
        if not isinstance(claim_id, str) or not isinstance(text, str):
            raise ValueError("claim_id 和 text 必须是字符串。")
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("source_ids 必须是字符串数组。")
        if category is not None and not isinstance(category, str):
            raise ValueError("category 必须是字符串或 null。")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(evidence_id, str) for evidence_id in evidence_ids
        ):
            raise ValueError("evidence_ids 必须是字符串数组。")
        claims.append(
            AnswerClaim(claim_id, text, source_ids, category, evidence_ids)
        )
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

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """供结构化 Agentic 阶段复用同一安全 HTTP 客户端。"""

        request_payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if self.config.json_mode:
            request_payload["response_format"] = {"type": "json_object"}
        response = self._client.post(
            self.endpoint,
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Chat Completions 必须返回 JSON object。")
        return payload

    def generate(self, query: str, context: ContextBundle) -> AnswerDraft:
        cleaned_query = query.strip()
        if self.config.prompt_layout == "context_first":
            user_prompt = (
                f"Evidence bundle:\n{context.context_text}\n\n"
                f"Question:\n{cleaned_query}\n\n"
                "Return the required JSON object."
            )
        else:
            user_prompt = (
                f"Question:\n{cleaned_query}\n\n"
                f"Evidence bundle:\n{context.context_text}\n\n"
                "Return the required JSON object."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        payload = self.chat_completion(messages)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Chat Completions 响应缺少 message.content。") from error
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat Completions 返回了空 content。")
        metadata: dict[str, Any] = {}
        response_model = payload.get("model") if isinstance(payload, dict) else None
        if isinstance(response_model, str) and response_model.strip():
            metadata["response_model"] = response_model
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            safe_usage = {
                str(key): value
                for key, value in usage.items()
                if isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            if safe_usage:
                metadata["usage"] = safe_usage
                hit_tokens = safe_usage.get("prompt_cache_hit_tokens")
                miss_tokens = safe_usage.get("prompt_cache_miss_tokens")
                if isinstance(hit_tokens, (int, float)) and isinstance(
                    miss_tokens,
                    (int, float),
                ):
                    eligible_tokens = hit_tokens + miss_tokens
                    if eligible_tokens > 0:
                        metadata["cache_diagnostics"] = {
                            "hit_tokens": hit_tokens,
                            "miss_tokens": miss_tokens,
                            "eligible_prompt_tokens": eligible_tokens,
                            "hit_rate": round(hit_tokens / eligible_tokens, 6),
                        }
        prompt_serialized = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stable_prefix = (
            SYSTEM_PROMPT
            + (
                f"Evidence bundle:\n{context.context_text}\n\nQuestion:\n"
                if self.config.prompt_layout == "context_first"
                else "Question:\n"
            )
        )
        metadata["prompt_diagnostics"] = {
            "layout": self.config.prompt_layout,
            "fingerprint": hashlib.sha256(
                prompt_serialized.encode("utf-8")
            ).hexdigest(),
            "stable_prefix_fingerprint": hashlib.sha256(
                stable_prefix.encode("utf-8")
            ).hexdigest(),
            "stable_prefix_approx_tokens": token_count(stable_prefix),
            "system_approx_tokens": token_count(SYSTEM_PROMPT),
            "query_approx_tokens": token_count(cleaned_query),
            "context_approx_tokens": token_count(context.context_text),
            "total_approx_tokens": token_count(SYSTEM_PROMPT + user_prompt),
            "token_counter": "deterministic_approximation_v1",
        }
        return replace(
            _parse_answer_draft(content),
            provider_metadata=metadata,
        )
