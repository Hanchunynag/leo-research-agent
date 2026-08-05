"""Agentic 各 LLM 阶段的结构化 Provider 与一次修复重试。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.agentic.models import (
    AgenticAnswerDraft,
    CoverageReport,
    QueryPlan,
    RoutingLLMDecision,
    SemanticValidationReport,
)
from app.generation.openai_compatible import OpenAICompatibleAnswerProvider
from app.generation.openai_compatible import parse_json_object


ROUTER_SYSTEM_PROMPT = """You are a deterministic scientific conversation topic router.
Return only JSON matching the supplied schema. Decide same_topic, related_subtopic, or
new_topic. Resolve pronouns in standalone_query. Never answer the research question."""

COVERAGE_SYSTEM_PROMPT = """You are a scientific evidence coverage checker.
Return only JSON matching the supplied schema. Evidence is sufficient only when it directly
supports each subquestion. Background relevance is not direct support. Produce focused
followup_queries for missing evidence. For a cross-paper synthesis or evolution question,
coverage is compositional: direct evidence describing each paper's own method, contribution,
extension, or result may jointly support an evolution stage. Do not require one source to state
the complete cross-paper evolution. Do not use outside knowledge."""

AGENTIC_ANSWER_SYSTEM_PROMPT = """You are a session-aware scientific RAG answer generator.
The following event stream is append-only and deterministic. Use only evidence_added events
and current source mappings. Return only JSON. Each atomic claim needs category, source_ids,
and stable evidence_ids. Direct answers may contain only the QueryPlan target_category.
Never classify predicted ephemerides, SGP4 propagation, priors, states, or parameters as
measurements. Do not merge conditions from different papers. Refuse when evidence coverage
is insufficient. For cross-paper synthesis, each paper-level method statement must cite its own
evidence; a comparison or progression claim must cite the evidence for every compared stage.
Return at most five concise atomic claims. Prior answerable=false events are failures or refusals,
not output examples to imitate. Only evidence IDs present in the latest current_sources mapping
may be cited; older evidence_added events remain history but are forbidden for the current answer.
Output exactly:
{"answerable":true,"claims":[{"claim_id":"C1","text":"atomic fact","category":"measurement","source_ids":["S1"],"evidence_ids":["E001"]}],"refusal_reason":null}
or {"answerable":false,"claims":[],"refusal_reason":"specific evidence limitation"}."""

SEMANTIC_VALIDATION_SYSTEM_PROMPT = """You are a strict claim-citation entailment judge.
Return only JSON. For every claim decide entailed, partially_entailed, not_entailed, or
contradicted. Check query alignment, category correctness, direct citation support, scope,
experimental conditions, and conflicts. Topic relevance alone is not entailment. Predicted
ephemerides are inputs or priors, not measurements. For cross-paper synthesis, a comparison or
evolution claim may be entailed compositionally when its citations directly support each side;
no individual paper needs to state the final cross-paper synthesis. Use only supplied evidence."""

REPAIR_SYSTEM_PROMPT = """You repair a structured scientific answer exactly once.
Return only JSON matching the answer schema. Apply validation actions: keep, narrow/rewrite,
drop, or refuse. Never add facts or citations absent from supplied evidence."""

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """不包含模型原文的可诊断结构化输出错误。"""

    def __init__(
        self,
        stage: str,
        cause: Exception,
        response_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.failure_kind = self._failure_kind(cause)
        self.validation_issues = self._validation_issues(cause)
        self.response_diagnostics = response_diagnostics or {}
        detail = (
            f"{stage} 未在结构修复预算内返回合法 JSON"
            f"（{self.failure_kind}）。"
        )
        super().__init__(detail)

    @staticmethod
    def _failure_kind(cause: Exception) -> str:
        if isinstance(cause, ValidationError):
            return "schema_validation"
        message = str(cause).casefold()
        if "content" in message:
            return "missing_content"
        if "json" in message:
            return "invalid_json"
        return "response_shape"

    @staticmethod
    def _validation_issues(cause: Exception) -> list[dict[str, str]]:
        if not isinstance(cause, ValidationError):
            return []
        issues: list[dict[str, str]] = []
        for value in cause.errors(include_url=False, include_input=False)[:8]:
            location = ".".join(str(item) for item in value.get("loc", ()))
            issues.append(
                {
                    "location": location or "$",
                    "type": str(value.get("type") or "validation_error"),
                    "message": str(value.get("msg") or "schema mismatch"),
                }
            )
        return issues

    def to_diagnostics(self) -> dict[str, Any]:
        """返回不含 Prompt、原始响应或密钥的安全诊断。"""

        return {
            "stage": self.stage,
            "failure_kind": self.failure_kind,
            "validation_issues": self.validation_issues,
            "response": self.response_diagnostics,
        }


class AgenticReasoningProvider(Protocol):
    """可注入 Fake 的 Agentic 结构化推理契约。"""

    model_name: str

    def resolve_route(
        self,
        standalone_query: str,
        topic_summary: str,
        signals: dict[str, float],
    ) -> RoutingLLMDecision: ...

    def check_coverage(
        self,
        plan: QueryPlan,
        evidence: Sequence[dict[str, Any]],
    ) -> CoverageReport: ...

    def generate_answer(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[AgenticAnswerDraft, dict[str, Any]]: ...

    def validate_semantic(
        self,
        query: str,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        evidence: Sequence[dict[str, Any]],
    ) -> SemanticValidationReport: ...

    def repair_answer(
        self,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        validation: SemanticValidationReport,
        evidence: Sequence[dict[str, Any]],
    ) -> AgenticAnswerDraft: ...


def _schema_payload(model_type: type[BaseModel]) -> dict[str, Any]:
    return model_type.model_json_schema()


def _safe_evidence(evidence: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "source_id",
        "evidence_id",
        "chunk_id",
        "work_id",
        "document_id",
        "title",
        "authors",
        "year",
        "doi",
        "section_path",
        "page_start",
        "page_end",
        "block_ids",
        "content",
        "origin",
        "directness_grade",
    )
    return [
        {field: item.get(field) for field in fields if field in item}
        for item in evidence
    ]


class OpenAIAgenticReasoningProvider:
    """使用现有 OpenAI-compatible 客户端执行全部受限结构化阶段。"""

    def __init__(
        self,
        provider: OpenAICompatibleAnswerProvider,
        *,
        max_structure_repairs: int = 1,
    ) -> None:
        if max_structure_repairs not in {0, 1}:
            raise ValueError("max_structure_repairs 只能是 0 或 1。")
        self.provider = provider
        self.model_name = provider.model_name
        self.max_structure_repairs = max_structure_repairs
        self.last_stage_diagnostics: dict[str, dict[str, Any]] = {}

    def _complete(
        self,
        stage: str,
        messages: list[dict[str, str]],
        model_type: type[ModelT],
        *,
        max_tokens: int | None = None,
    ) -> tuple[ModelT, dict[str, Any]]:
        working = [dict(message) for message in messages]
        last_error: Exception | None = None
        response_diagnostics: dict[str, Any] = {}
        requested_max_tokens = max_tokens
        configured_max_tokens = getattr(
            getattr(self.provider, "config", None),
            "max_tokens",
            1200,
        )
        initial_max_tokens = max_tokens or int(configured_max_tokens)
        for attempt in range(self.max_structure_repairs + 1):
            payload = (
                self.provider.chat_completion(working)
                if requested_max_tokens is None
                else self.provider.chat_completion(
                    working,
                    max_tokens=requested_max_tokens,
                )
            )
            response_diagnostics = self._response_diagnostics(payload)
            try:
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("message.content 不是字符串。")
                result = model_type.model_validate(parse_json_object(content))
                diagnostics = self._metadata(payload, messages, attempt)
                self.last_stage_diagnostics[stage] = diagnostics
                return result, diagnostics
            except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
                last_error = error
                if attempt < self.max_structure_repairs:
                    if response_diagnostics.get("truncated_before_content"):
                        requested_max_tokens = min(
                            max(initial_max_tokens * 2, 4096),
                            16384,
                        )
                    choices = payload.get("choices")
                    first_choice = (
                        choices[0]
                        if isinstance(choices, list)
                        and choices
                        and isinstance(choices[0], dict)
                        else {}
                    )
                    working.extend(
                        [
                            {
                                "role": "assistant",
                                "content": str(
                                    first_choice.get("message", {})
                                    .get("content", "")
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was invalid. Return exactly one "
                                    "JSON object matching this schema, with no markdown:\n"
                                    + json.dumps(
                                        _schema_payload(model_type),
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                ),
                            },
                        ]
                    )
        assert last_error is not None
        raise StructuredOutputError(
            stage,
            last_error,
            response_diagnostics,
        ) from last_error

    @staticmethod
    def _response_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
        """只保留截断判断所需的安全响应元数据。"""

        choices = payload.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else {}
        )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        usage = payload.get("usage")
        safe_usage = (
            {
                str(key): value
                for key, value in usage.items()
                if isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            if isinstance(usage, dict)
            else {}
        )
        finish_reason = choice.get("finish_reason")
        content_length = len(content) if isinstance(content, str) else None
        return {
            "finish_reason": (
                finish_reason if isinstance(finish_reason, str) else None
            ),
            "content_length": content_length,
            "usage": safe_usage,
            "truncated_before_content": bool(
                finish_reason == "length" and not content
            ),
        }

    @staticmethod
    def _metadata(
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        repair_attempt: int,
    ) -> dict[str, Any]:
        usage = payload.get("usage")
        safe_usage = (
            {
                str(key): value
                for key, value in usage.items()
                if isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            if isinstance(usage, dict)
            else {}
        )
        serialized = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "response_model": payload.get("model"),
            "usage": safe_usage,
            "prompt_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "message_count": len(messages),
            "structure_repair_attempts": repair_attempt,
        }

    @staticmethod
    def _request_messages(
        system_prompt: str,
        model_type: type[BaseModel],
        payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        request = {
            "input": payload,
            "output_schema": _schema_payload(model_type),
        }
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]

    def resolve_route(
        self,
        standalone_query: str,
        topic_summary: str,
        signals: dict[str, float],
    ) -> RoutingLLMDecision:
        """对模糊组合分数执行一次结构化 Topic 判定。"""

        result, _ = self._complete(
            "topic_router",
            self._request_messages(
                ROUTER_SYSTEM_PROMPT,
                RoutingLLMDecision,
                {
                    "standalone_query": standalone_query,
                    "topic_summary": topic_summary,
                    "signals": signals,
                },
            ),
            RoutingLLMDecision,
            max_tokens=min(self.provider.config.max_tokens, 1200),
        )
        return result

    def check_coverage(
        self,
        plan: QueryPlan,
        evidence: Sequence[dict[str, Any]],
    ) -> CoverageReport:
        """让模型逐个子问题检查直接证据覆盖。"""

        result, _ = self._complete(
            "coverage",
            self._request_messages(
                COVERAGE_SYSTEM_PROMPT,
                CoverageReport,
                {"query_plan": plan.model_dump(mode="json"), "evidence": _safe_evidence(evidence)},
            ),
            CoverageReport,
            max_tokens=min(self.provider.config.max_tokens, 2048),
        )
        return result

    def generate_answer(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[AgenticAnswerDraft, dict[str, Any]]:
        """从 append-only Topic 消息生成结构化原子 Claim。"""

        return self._complete(
            "answer",
            messages,
            AgenticAnswerDraft,
            max_tokens=max(self.provider.config.max_tokens, 8192),
        )

    def validate_semantic(
        self,
        query: str,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        evidence: Sequence[dict[str, Any]],
    ) -> SemanticValidationReport:
        """执行逐 Claim 的 citation entailment 判断。"""

        result, _ = self._complete(
            "semantic_validation",
            self._request_messages(
                SEMANTIC_VALIDATION_SYSTEM_PROMPT,
                SemanticValidationReport,
                {
                    "query": query,
                    "query_plan": plan.model_dump(mode="json"),
                    "answer": draft.model_dump(mode="json"),
                    "evidence": _safe_evidence(evidence),
                },
            ),
            SemanticValidationReport,
            max_tokens=min(self.provider.config.max_tokens, 4096),
        )
        return result

    def repair_answer(
        self,
        plan: QueryPlan,
        draft: AgenticAnswerDraft,
        validation: SemanticValidationReport,
        evidence: Sequence[dict[str, Any]],
    ) -> AgenticAnswerDraft:
        """依据语义验证动作执行至多一次结构化修复。"""

        result, _ = self._complete(
            "answer_repair",
            self._request_messages(
                REPAIR_SYSTEM_PROMPT,
                AgenticAnswerDraft,
                {
                    "query_plan": plan.model_dump(mode="json"),
                    "answer": draft.model_dump(mode="json"),
                    "validation": validation.model_dump(mode="json"),
                    "evidence": _safe_evidence(evidence),
                },
            ),
            AgenticAnswerDraft,
            max_tokens=min(self.provider.config.max_tokens, 4096),
        )
        return result
