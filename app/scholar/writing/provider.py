"""可注入的 Introduction Writer 和可选语义 Reviewer Provider。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.generation.openai_compatible import parse_json_object
from app.scholar.models import ReviewIssue
from app.scholar.writing.models import SectionDraft, WritingContext
from app.scholar.writing.runtime import SkillExecutionContext


class ChatCompletionIntroductionWriter:
    """把现有 chat_completion Provider 适配为结构化 Introduction Writer。"""

    def __init__(self, provider: Any) -> None:
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("Introduction Writer 需要 chat_completion Provider。")
        self.provider = provider

    def _call(self, context: WritingContext) -> SectionDraft:
        payload = context.public_mapping()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative scientific Introduction editor. Return only JSON. "
                    "Use only supplied verified evidence for external claims. Never invent facts, "
                    "contributions, evidence IDs, or citation keys. Preserve unrelated existing text. "
                    "Return fields: content, claim_ids, evidence_ids, citation_keys, citation_binding_ids, contribution_ids, "
                    "change_summary, warnings."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        response = self.provider.chat_completion(messages)
        choices = response.get("choices", []) if isinstance(response, Mapping) else []
        content = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Introduction Writer 返回空内容。")
        value = parse_json_object(content)
        allowed_claims = context.claim_plan.claim_ids
        allowed_evidence = context.claim_plan.evidence_ids
        allowed_contributions = {item.contribution_id for item in context.contributions}
        claim_ids = tuple(str(item) for item in value.get("claim_ids", []) if isinstance(item, str))
        evidence_ids = tuple(str(item) for item in value.get("evidence_ids", []) if isinstance(item, str))
        citation_keys = tuple(str(item) for item in value.get("citation_keys", []) if isinstance(item, str))
        contribution_ids = tuple(str(item) for item in value.get("contribution_ids", []) if isinstance(item, str))
        citation_binding_ids = tuple(str(item) for item in value.get("citation_binding_ids", []) if isinstance(item, str))
        if not set(claim_ids) <= allowed_claims:
            raise ValueError("Introduction Writer 返回了未知 claim_id。")
        if not set(evidence_ids) <= allowed_evidence:
            raise ValueError("Introduction Writer 返回了未知 evidence_id。")
        if not set(contribution_ids) <= allowed_contributions:
            raise ValueError("Introduction Writer 返回了未知 contribution_id。")
        return SectionDraft(
            target_section="introduction",
            base_hash=context.current_hash,
            content=str(value.get("content") or ""),
            claim_ids=claim_ids,
            evidence_ids=evidence_ids,
            citation_keys=citation_keys,
            citation_requirements=context.citation_requirements,
            contribution_ids=contribution_ids,
            change_summary=str(value.get("change_summary") or ""),
            warnings=tuple(str(item) for item in value.get("warnings", []) if isinstance(item, str)),
            original_content=context.current_introduction,
            citation_binding_ids=citation_binding_ids,
        )

    def generate(self, context: WritingContext) -> SectionDraft:
        return self._call(context)

    def revise(self, context: WritingContext, _: Any) -> SectionDraft:
        return self._call(context)


class ChatCompletionSemanticReviewJudge:
    """可选的 LLM Semantic Review；只返回 ReviewIssue，不修改任何状态。"""

    def __init__(self, provider: Any) -> None:
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("Semantic Review Judge 需要 chat_completion Provider。")
        self.provider = provider

    def review(self, payload: Mapping[str, Any]) -> Sequence[ReviewIssue]:
        response = self.provider.chat_completion([
            {"role": "system", "content": "Return only JSON: {\"issues\":[{\"code\":str,\"severity\":\"BLOCKER\"|\"HIGH\"|\"MEDIUM\"|\"LOW\",\"message\":str,\"claim_id\":str|null}]} . Judge entailment using only supplied evidence."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ])
        choices = response.get("choices", []) if isinstance(response, Mapping) else []
        text = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], Mapping) else None
        value = parse_json_object(text) if isinstance(text, str) else {}
        output: list[ReviewIssue] = []
        for item in value.get("issues", []) if isinstance(value.get("issues"), list) else []:
            if not isinstance(item, Mapping):
                continue
            severity = str(item.get("severity") or "HIGH")
            if severity not in {"BLOCKER", "HIGH", "MEDIUM", "LOW"}:
                severity = "HIGH"
            output.append(ReviewIssue(str(item.get("code") or "SEMANTIC_REVIEW"), severity, str(item.get("message") or "Semantic review issue."), str(item.get("claim_id")) if item.get("claim_id") else None))  # type: ignore[arg-type]
        return tuple(output)


class ChatCompletionSynthesisWriter:
    """Conservative provider adapter shared by Conclusion and Abstract."""

    def __init__(self, provider: Any, target_section: str) -> None:
        if target_section not in {"conclusion", "abstract"}:
            raise ValueError("Synthesis Writer 只支持 conclusion/abstract。")
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("Synthesis Writer 需要 chat_completion Provider。")
        self.provider = provider
        self.target_section = target_section

    def _call(self, context: SkillExecutionContext) -> SectionDraft:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a conservative scientific {self.target_section} editor. Return only JSON. "
                    "Use only the supplied manuscript snapshot, facts and confirmed contributions. "
                    "Do not use literature, citations, new numbers, new methods, or new contributions. "
                    "Return fields: content, claim_ids, contribution_ids, change_summary, warnings."
                ),
            },
            {"role": "user", "content": json.dumps(context.public_mapping(), ensure_ascii=False, default=str)},
        ]
        response = self.provider.chat_completion(messages)
        choices = response.get("choices", []) if isinstance(response, Mapping) else []
        content = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Synthesis Writer 返回空内容。")
        value = parse_json_object(content)
        allowed_contributions = {item.contribution_id for item in context.contributions}
        contribution_ids = tuple(str(item) for item in value.get("contribution_ids", []) if isinstance(item, str))
        if not set(contribution_ids) <= allowed_contributions:
            raise ValueError("Synthesis Writer 返回了未知 contribution_id。")
        return SectionDraft(
            target_section=self.target_section,
            base_hash=str(context.current_hash or ""),
            content=str(value.get("content") or ""),
            claim_ids=tuple(str(item) for item in value.get("claim_ids", []) if isinstance(item, str)),
            evidence_ids=(),
            citation_keys=(),
            contribution_ids=contribution_ids,
            change_summary=str(value.get("change_summary") or ""),
            warnings=tuple(str(item) for item in value.get("warnings", []) if isinstance(item, str)),
            original_content=context.current_section,
        )

    def generate(self, context: SkillExecutionContext) -> SectionDraft:
        return self._call(context)

    def revise(self, context: SkillExecutionContext, _: Any) -> SectionDraft:
        return self._call(context)
