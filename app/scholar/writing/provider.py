"""可注入的 Introduction Writer 和可选语义 Reviewer Provider。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.generation.openai_compatible import parse_json_object, structured_chat_completion
from app.generation.security import redact_sensitive_text
from app.scholar.models import ReviewIssue
from app.scholar.writing.models import SectionDraft, WritingContext
from app.scholar.writing.runtime import SkillExecutionContext


class ChatCompletionIntroductionWriter:
    """把现有 chat_completion Provider 适配为结构化 Introduction Writer。"""

    def __init__(self, provider: Any) -> None:
        if not callable(getattr(provider, "chat_completion", None)):
            raise TypeError("Introduction Writer 需要 chat_completion Provider。")
        self.provider = provider
        self._trace: Any | None = None

    def set_trace(self, trace: Any | None) -> None:
        """Attach the current Run trace for direct (non-CrewAI) LLM calls."""

        self._trace = trace

    def _structured_call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stage: str,
    ) -> Mapping[str, Any]:
        trace = self._trace
        metadata = {
            "agent_role": "Writer Agent",
            "provider_stage": stage,
            "max_tokens": max_tokens,
        }
        if trace is not None:
            trace.record("writer_provider_call", "RUNNING", metadata, kind="llm")
        try:
            response = structured_chat_completion(
                self.provider,
                messages,
                max_tokens=max_tokens,
            )
        except Exception as error:
            if trace is not None:
                trace.record(
                    "writer_provider_call",
                    "FAILED",
                    {**metadata, "error_type": type(error).__name__},
                    kind="llm",
                )
            raise
        if trace is not None:
            usage = response.get("usage") if isinstance(response, Mapping) else None
            trace.record(
                "writer_provider_call",
                "COMPLETED",
                {
                    **metadata,
                    **({"usage": usage} if isinstance(usage, Mapping) else {}),
                },
                kind="llm",
            )
        return response

    def _call(self, context: WritingContext) -> SectionDraft:
        # Keep complete evidence in the domain runtime, but send only the
        # bounded writer projection over the wire.  The old public mapping
        # included repeated chunks and full metadata and could exceed the
        # gateway context window before a draft was generated.
        payload = context.writer_mapping()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative scientific Introduction editor. Return only JSON. "
                    "Use only supplied verified evidence for external claims. Never invent facts, "
                    "contributions, evidence IDs, or citation keys. Preserve unrelated existing text. "
                    "This is a literature-grounded Introduction: cite at least 5 DISTINCT papers. "
                    "Count papers by stable paper identity, never by chunks or duplicate BibKeys. "
                    "If the supplied evidence cannot support 5 distinct papers, explain the insufficiency "
                    "in warnings and do not pretend that fewer sources satisfy the requirement. "
                    "Return exactly one JSON object matching this shape: "
                    '{"content":"...","claim_ids":[],"evidence_ids":[],"citation_keys":[],'
                    '"citation_binding_ids":[],"contribution_ids":[],"change_summary":"...","warnings":[]}. '
                    "Do not write any text before or after the JSON object."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        # Structured Writer output does not need hidden chain-of-thought.  On
        # the local Qwen-compatible gateway, allowing it to consume the full
        # 8k default budget made the call look hung and often left no JSON
        # content.  Keep the response bounded and explicitly disable hidden
        # reasoning where the concrete provider supports it.
        response = self._structured_call(
            messages,
            max_tokens=4096,
            stage="draft_generation",
        )
        choices = response.get("choices", []) if isinstance(response, Mapping) else []
        content = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Introduction Writer 返回空内容。")
        try:
            value = parse_json_object(content)
        except ValueError as original_error:
            # Some OpenAI-compatible gateways honor JSON mode for short
            # prompts but return a useful plain-text draft for a long writing
            # prompt. Give the same model one bounded normalization pass. The
            # repair prompt contains only the draft and allow-lists; it cannot
            # invent IDs and the deterministic reviewer remains authoritative.
            repair_payload = {
                "draft_text": content[:9000],
                "allowed_claim_ids": sorted(context.claim_plan.claim_ids),
                "allowed_evidence_ids": sorted(context.claim_plan.evidence_ids),
                "citation_catalog": dict(context.citation_catalog),
                "allowed_contribution_ids": sorted(
                    item.contribution_id for item in context.contributions
                ),
            }
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON normalizer. Return exactly one JSON object and "
                        "nothing else. Wrap the supplied draft text into content and select "
                        "only IDs from the allow-lists. Select citation_keys only from the "
                        "citation_catalog. Do not add Markdown fences, explanations, or new "
                        "facts. Required shape: "
                        '{"content":"...","claim_ids":[],"evidence_ids":[],"citation_keys":[],'
                        '"citation_binding_ids":[],"contribution_ids":[],"change_summary":"...","warnings":[]}. '
                        "Keep content concise enough to fit the response and preserve the supplied draft meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        repair_payload, ensure_ascii=False, default=str
                    ),
                },
            ]
            try:
                repaired = self._structured_call(
                    repair_messages,
                    max_tokens=4096,
                    stage="json_repair",
                )
                repair_choices = (
                    repaired.get("choices", [])
                    if isinstance(repaired, Mapping)
                    else []
                )
                repair_content = (
                    repair_choices[0].get("message", {}).get("content")
                    if repair_choices and isinstance(repair_choices[0], Mapping)
                    else None
                )
                if not isinstance(repair_content, str) or not repair_content.strip():
                    raise ValueError("JSON 修复调用返回空内容。")
                value = parse_json_object(repair_content)
            except Exception as repair_error:
                # Keep the provider failure actionable without persisting the
                # full model response or any prompt content to Run events.
                preview = redact_sensitive_text(content.strip()[:500])
                repair_preview = ""
                if "repair_content" in locals() and isinstance(repair_content, str):
                    repair_preview = redact_sensitive_text(repair_content.strip()[:500])
                raise ValueError(
                    f"{original_error}; JSON_REPAIR_FAILED={type(repair_error).__name__}: "
                    f"{repair_error}; raw_content_preview={preview!r}; "
                    f"repair_content_preview={repair_preview!r}"
                ) from repair_error
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
