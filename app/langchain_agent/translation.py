"""LangChain Tool：把用户问题规范为可并行检索的中英文查询。"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.generation.openai_compatible import parse_json_object


def detect_output_language(text: str) -> str:
    """Return the language requested by the user, without changing retrieval variants."""

    # Chinese is deliberately detected from the original input rather than from the
    # translation result.  This prevents an English translation from accidentally
    # becoming the answer language for a Chinese question.
    return "zh" if any("\u4e00" <= character <= "\u9fff" for character in text) else "en"


class TranslationToolInput(BaseModel):
    """翻译 Tool 的公开输入契约。"""

    query: str = Field(min_length=1, description="用户的原始科研问题")


class BilingualQuery(BaseModel):
    """翻译模型返回的受限结构。"""

    original_query: str = Field(min_length=1)
    zh_query: str = Field(min_length=1)
    en_query: str = Field(min_length=1)

    def retrieval_queries(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value.strip()
                for value in (self.original_query, self.zh_query, self.en_query)
                if value.strip()
            )
        )


class BilingualQueryTranslator:
    """使用现有 OpenAI-compatible 客户端的确定性 LangChain Tool 实现。

    翻译失败不能让已有单语检索失效，因此始终回退到原 query；调用状态会
    返回给追踪信息，避免把模型故障误报为“证据不足”。
    """

    def __init__(self, reasoning_provider: Any) -> None:
        provider = getattr(reasoning_provider, "provider", None)
        if provider is None or not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("翻译 Tool 需要具备 chat_completion 的回答模型 Provider。")
        self.provider = provider

    def translate(self, query: str) -> dict[str, Any]:
        original = query.strip()
        if not original:
            raise ValueError("query 不能为空。")
        output_language = detect_output_language(original)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a deterministic bilingual query translation tool for scientific "
                    "literature retrieval. Preserve every technical entity, qualifier, "
                    "measurement, relationship, acronym, and requested output. Return exactly "
                    "one JSON object and no markdown with these non-empty string fields: "
                    "original_query, zh_query, en_query. zh_query must be a faithful simplified "
                    "Chinese query; en_query must be a faithful English query. Do not answer the "
                    "question, expand the scope, or add facts."
                ),
            },
            {"role": "user", "content": original},
        ]
        try:
            response = self.provider.chat_completion(messages, max_tokens=800)
            choices = response.get("choices") if isinstance(response, Mapping) else None
            first = choices[0] if isinstance(choices, list) and choices else {}
            message = first.get("message") if isinstance(first, Mapping) else {}
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("翻译模型未返回 message.content。")
            parsed = parse_json_object(content)
            value = BilingualQuery.model_validate(
                {
                    "original_query": original,
                    "zh_query": parsed.get("zh_query"),
                    "en_query": parsed.get("en_query"),
                }
            )
            return {
                **value.model_dump(),
                "retrieval_queries": list(value.retrieval_queries()),
                "translation_status": "translated",
                "output_language": output_language,
                "usage": _usage(response),
            }
        except Exception as error:
            # 失败时既不虚构翻译，也不阻塞已有的 RAG 回答能力。
            fallback = BilingualQuery(
                original_query=original,
                zh_query=original,
                en_query=original,
            )
            return {
                **fallback.model_dump(),
                "retrieval_queries": list(fallback.retrieval_queries()),
                "translation_status": "fallback",
                "translation_failure_kind": type(error).__name__,
                "output_language": output_language,
                "usage": {},
            }


def _usage(response: Any) -> dict[str, Any]:
    """Keep provider usage fields small, numeric, and safe for Web diagnostics."""

    raw = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    usage = {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if "total_tokens" not in usage:
        usage["total_tokens"] = int(usage.get("prompt_tokens") or 0) + int(
            usage.get("completion_tokens") or 0
        )
    return usage


def build_translation_tool(translator: BilingualQueryTranslator) -> StructuredTool:
    """创建显式命名的 LangChain Tool，确保翻译在召回前可审计。"""

    return StructuredTool.from_function(
        func=translator.translate,
        name="translate_query",
        description=(
            "Translate a scientific query into faithful Chinese and English variants before "
            "retrieval. This tool must run before any literature retrieval."
        ),
        args_schema=TranslationToolInput,
    )
