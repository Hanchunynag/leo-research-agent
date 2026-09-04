"""应用内 LangChain Skill：固定执行的双语查询翻译。"""

from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter
from typing import Any, Mapping, Sequence, cast

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.langchain_agent.translation import (
    BilingualQueryTranslator,
    build_translation_tool,
)
from app.langchain_agent.tools import (
    BilingualRetrievalTool,
    build_bilingual_gateway_handler,
    build_bilingual_retrieval_tool,
)
from app.research.context import PhaseContextPack
from app.research.validation import TieredClaimEvidenceValidator
from app.generation.openai_compatible import parse_json_object


class ChatCompletionHighRiskJudge:
    """用同一个 OpenAI-compatible ChatModel 做一次高风险 Claim 判定。

    该 Judge 只返回支持关系，不生成答案，也不能增加或修改 Evidence。
    只有生产 Provider（带 config 的 HTTP Provider）才会注入它；轻量测试
    Provider 仍使用原有离线校验路径。
    """

    def __init__(self, provider: Any) -> None:
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("高风险语义 Judge 需要 chat_completion Provider。")
        self.provider = provider
        self.last_usage: dict[str, int] = {}
        self.last_called = False

    def judge(
        self,
        claim: str,
        evidence: Sequence[str],
        risk_types: Sequence[str],
    ) -> dict[str, Any]:
        self.last_called = True
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict claim-evidence entailment judge. Return exactly one JSON "
                    "object with label, confidence, reason. label must be one of supports, "
                    "partially_supports, contradicts, unrelated. Judge only the supplied claim "
                    "against the supplied evidence. Do not add facts or citations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "claim": claim,
                        "evidence": list(evidence),
                        "risk_types": list(risk_types),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        configured = int(getattr(getattr(self.provider, "config", None), "max_tokens", 800))
        response = self.provider.chat_completion(messages, max_tokens=min(configured, 800))
        choices = response.get("choices") if isinstance(response, Mapping) else None
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = first.get("message") if isinstance(first, Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("语义 Judge 未返回 message.content。")
        payload = parse_json_object(content)
        label = str(payload.get("label") or "")
        if label not in {"supports", "partially_supports", "contradicts", "unrelated"}:
            raise ValueError("语义 Judge label 无效。")
        confidence = float(payload.get("confidence") or 0.0)
        if not 0 <= confidence <= 1:
            raise ValueError("语义 Judge confidence 无效。")
        raw_usage = response.get("usage") if isinstance(response, Mapping) else None
        usage = {
            "input_tokens": int((raw_usage or {}).get("input_tokens") or (raw_usage or {}).get("prompt_tokens") or 0),
            "output_tokens": int((raw_usage or {}).get("output_tokens") or (raw_usage or {}).get("completion_tokens") or 0),
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        self.last_usage = usage
        return {
            "label": label,
            "confidence": confidence,
            "reason": str(payload.get("reason") or "高风险结构化判定。"),
            "usage": usage,
        }


class TranslationSkill:
    """在 RAG 召回前固定调用 LLM 的 LangChain Skill。

    此 Skill 不负责选择是否翻译。它一旦被放入 Chain 就必定执行，保证中英文
    query 的生成和检索路径完全可审计；选择策略不交由回答模型隐式判断。
    """

    def __init__(self, reasoning_provider: Any) -> None:
        self.tool = build_translation_tool(BilingualQueryTranslator(reasoning_provider))
        self.model_name = str(getattr(reasoning_provider, "model_name", ""))
        self.runnable: Runnable[Mapping[str, Any], Mapping[str, Any]] = (
            RunnableLambda(self._invoke_tool, name="TranslationSkill")
            .with_config(
                run_name="translation_skill",
                tags=["langchain", "skill", "translation", "required_before_retrieval"],
            )
        )

    def _invoke_tool(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Mapping[str, Any]:
        query = str(state["query"])
        callback = state.get("progress_callback")
        if callable(callback):
            try:
                callback("translation", "正在调用 TranslationSkill 的 LLM Tool。", 0.23)
            except Exception:
                pass
        started = perf_counter()
        # StructuredTool 将 config 中的 callbacks/tags/metadata 继续向下传播，
        # 因此外部 LangChain tracer 能同时看到 Skill 和 translate_query Tool。
        result = self.tool.invoke({"query": query}, config=config)
        if not isinstance(result, Mapping):
            raise TypeError("TranslationSkill 必须返回结构化对象。")
        output = {
            **dict(result),
            "llm_execution": {
                "tool": "translate_query",
                "model": self.model_name,
                "max_tokens": 800,
                "usage": dict(result.get("usage") or {}),
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            },
        }
        if callable(callback):
            try:
                callback(
                    "translation_completed",
                    "TranslationSkill 完成，已生成中英文检索 query。",
                    0.30,
                    {"translation_status": output.get("translation_status")},
                )
            except Exception:
                pass
        return output


class BilingualRetrievalSkill:
    """调用 legacy RAG、融合中英文 query 且只输出 Selected Evidence。"""

    def __init__(self, knowledge: Any) -> None:
        self.tool = build_bilingual_retrieval_tool(BilingualRetrievalTool(knowledge))
        self.name = "BilingualRetrievalSkill"

    def gateway_handler(self):
        """返回受现有 Scope、权限与预算 Gateway 管理的 Tool handler。"""

        return build_bilingual_gateway_handler(self.tool)


class ScopeReadInput(BaseModel):
    workspace_id: str = Field(min_length=1)
    scope_version: int = Field(ge=1)


class ScopeReadSkill:
    """读取不可变 Workspace Scope，禁止检索越过当前文献集合。"""

    name = "ScopeReadSkill"

    def __init__(self, workspaces: Any) -> None:
        self.workspaces = workspaces
        self.tool = StructuredTool.from_function(
            func=self.read_scope,
            name="read_workspace_scope",
            description="Read the immutable workspace scope before literature retrieval.",
            args_schema=ScopeReadInput,
        )

    def read_scope(self, workspace_id: str, scope_version: int) -> dict[str, Any]:
        scope = self.workspaces.require_scope(workspace_id, scope_version)
        return {**asdict(scope), "constraints": []}

    def gateway_handler(self):
        def read_scope(
            arguments: Mapping[str, Any], context: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            workspace_id = str(arguments["workspace_id"])
            scope_version = int(arguments["scope_version"])
            if workspace_id != str(context.get("workspace_id")) or scope_version != int(
                context.get("scope_version") or 0
            ):
                raise PermissionError("Tool 参数与 Run scope 不一致。")
            result = self.tool.invoke(
                {"workspace_id": workspace_id, "scope_version": scope_version}
            )
            if not isinstance(result, Mapping):
                raise TypeError("ScopeReadSkill 必须返回结构化对象。")
            return dict(result)

        return read_scope


class AnswerGenerationSkill:
    """保持原回答 Prompt 和结构化契约的 LangChain LLM Skill。"""

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self.runnable: Runnable[PhaseContextPack, Mapping[str, Any]] = (
            RunnableLambda(self._invoke_generator, name="AnswerGenerationSkill")
            .with_config(
                run_name="answer_generation_skill",
                tags=["langchain", "skill", "llm", "selected_evidence_only"],
            )
        )

    def _invoke_generator(
        self,
        context: PhaseContextPack,
        _config: RunnableConfig | None = None,
    ) -> Mapping[str, Any]:
        return self.generator.generate(context)

    def generate(self, context: PhaseContextPack) -> Mapping[str, Any]:
        """兼容现有 Workflow 的 Generator 协议。"""

        result = self.runnable.invoke(context)
        if not isinstance(result, Mapping):
            raise TypeError("AnswerGenerationSkill 必须返回结构化回答草稿。")
        return dict(result)


class ClaimValidationSkill:
    """把 Claim-Evidence 校验放入 LangChain Runnable，保持保守验证逻辑。"""

    def __init__(
        self,
        validator: Any | None = None,
        *,
        high_risk_judge: Any | None = None,
    ) -> None:
        self.validator = validator or TieredClaimEvidenceValidator(
            high_risk_judge=high_risk_judge
        )
        self.runnable: Runnable[Mapping[str, Any], Any] = cast(
            Runnable[Mapping[str, Any], Any],
            RunnableLambda(self._invoke_validator, name="ClaimValidationSkill").with_config(
                run_name="claim_validation_skill",
                tags=["langchain", "skill", "validation", "selected_evidence_only"],
            ),
        )

    def _invoke_validator(
        self,
        state: Mapping[str, Any],
        _config: RunnableConfig | None = None,
    ) -> Any:
        judge = getattr(self.validator, "high_risk_judge", None)
        if judge is not None:
            # The skill instance is reused by the Web runtime; diagnostics must
            # describe this run only, never leak a previous run's usage.
            judge.last_usage = {}
            judge.last_called = False
        return self.validator.validate(
            state["draft"],
            state["evidence"],
            scope_constraints=state.get("scope_constraints", ()),
            conflicts=state.get("conflicts", ()),
        )

    def validate(
        self,
        draft: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        *,
        scope_constraints: Sequence[str] = (),
        conflicts: Sequence[Mapping[str, Any]] = (),
    ) -> Any:
        return self.runnable.invoke(
            {
                "draft": draft,
                "evidence": evidence,
                "scope_constraints": tuple(scope_constraints),
                "conflicts": tuple(conflicts),
            }
        )

    def deterministic_repair(self, draft: Mapping[str, Any], report: Any) -> dict[str, Any]:
        return self.validator.deterministic_repair(draft, report)

    @property
    def last_judge_usage(self) -> Mapping[str, Any]:
        judge = getattr(self.validator, "high_risk_judge", None)
        usage = getattr(judge, "last_usage", {})
        return dict(usage) if isinstance(usage, Mapping) else {}

    @property
    def last_judge_called(self) -> bool:
        judge = getattr(self.validator, "high_risk_judge", None)
        return bool(getattr(judge, "last_called", False))
