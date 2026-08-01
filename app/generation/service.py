"""连接检索、上下文组装、回答生成与引用校验。"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from app.context.models import ContextBundle
from app.generation.base import AnswerProvider
from app.generation.models import (
    AnswerClaim,
    CitationValidationIssue,
    CitationValidationReport,
    GroundedAnswer,
)
from app.generation.refusal import (
    EMPTY_CONTEXT_REFUSAL,
    INVALID_CONTEXT_REFUSAL,
    INVALID_DRAFT_REFUSAL,
    PROVIDER_ERROR_REFUSAL,
)
from app.generation.validation import validate_answer_draft, validate_context_bundle
from app.runtime.retrieval import RetrievalMode, RetrievalRuntime


def _render_claims(claims: list[AnswerClaim]) -> str:
    lines = []
    for claim in claims:
        citations = "".join(f"[{source_id}]" for source_id in claim.source_ids)
        lines.append(f"{claim.text.strip()} {citations}")
    return "\n".join(lines)


class GroundedAnswerService:
    """只输出通过上下文完整性和逐条引用校验的回答。"""

    def __init__(
        self,
        retrieval_runtime: RetrievalRuntime,
        answer_provider: AnswerProvider,
    ) -> None:
        self.retrieval_runtime = retrieval_runtime
        self.answer_provider = answer_provider

    def _refusal(
        self,
        context: ContextBundle,
        reason: str,
        validation: CitationValidationReport,
        diagnostics: dict[str, Any],
    ) -> GroundedAnswer:
        return GroundedAnswer(
            query=context.query,
            answerable=False,
            answer="",
            claims=[],
            citations=[],
            refusal_reason=reason,
            validation=validation,
            context=context,
            diagnostics=diagnostics,
        )

    def answer_from_context(self, context: ContextBundle) -> GroundedAnswer:
        """对已组装证据包生成回答，便于 API 复用和独立测试。"""

        diagnostics: dict[str, Any] = {
            "provider": type(self.answer_provider).__name__,
            "model": getattr(self.answer_provider, "model_name", None),
        }
        context_issues = validate_context_bundle(context)
        if context_issues:
            validation = CitationValidationReport(False, context_issues, [])
            reason = (
                EMPTY_CONTEXT_REFUSAL
                if any(issue.code == "empty_context" for issue in context_issues)
                else INVALID_CONTEXT_REFUSAL
            )
            diagnostics["generation_skipped"] = True
            return self._refusal(context, reason, validation, diagnostics)

        started = perf_counter()
        try:
            draft = self.answer_provider.generate(context.query, context)
        except Exception as error:
            diagnostics.update(
                {
                    "generation_elapsed_ms": round(
                        (perf_counter() - started) * 1000,
                        3,
                    ),
                    "provider_error": f"{type(error).__name__}: {error}",
                }
            )
            validation = CitationValidationReport(
                False,
                [
                    CitationValidationIssue(
                        "provider_error",
                        "回答模型调用失败或返回格式不合法。",
                    )
                ],
                [],
            )
            return self._refusal(
                context,
                PROVIDER_ERROR_REFUSAL,
                validation,
                diagnostics,
            )
        diagnostics["generation_elapsed_ms"] = round(
            (perf_counter() - started) * 1000,
            3,
        )
        diagnostics.update(draft.provider_metadata)

        validation = validate_answer_draft(draft, context)
        if not validation.valid:
            return self._refusal(
                context,
                INVALID_DRAFT_REFUSAL,
                validation,
                diagnostics,
            )
        if not draft.answerable:
            assert draft.refusal_reason is not None
            return self._refusal(
                context,
                draft.refusal_reason.strip(),
                validation,
                diagnostics,
            )
        return GroundedAnswer(
            query=context.query,
            answerable=True,
            answer=_render_claims(draft.claims),
            claims=draft.claims,
            citations=validation.citations,
            refusal_reason=None,
            validation=validation,
            context=context,
            diagnostics=diagnostics,
        )

    def answer(
        self,
        query: str,
        *,
        mode: RetrievalMode = "fast",
        retrieval_limit: int = 10,
        token_budget: int = 6000,
        max_evidence: int = 8,
        max_evidence_per_work: int = 2,
        work_id: str | None = None,
        document_id: str | None = None,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ) -> GroundedAnswer:
        """检索、组装并生成一个带可追溯引用的回答。"""

        started = perf_counter()
        context = self.retrieval_runtime.build_context(
            query=query,
            mode=mode,
            retrieval_limit=retrieval_limit,
            token_budget=token_budget,
            max_evidence=max_evidence,
            max_evidence_per_work=max_evidence_per_work,
            work_id=work_id,
            document_id=document_id,
            candidate_limit=candidate_limit,
            rrf_k=rrf_k,
        )
        result = self.answer_from_context(context)
        diagnostics = dict(result.diagnostics)
        diagnostics["retrieval_and_context_elapsed_ms"] = round(
            (perf_counter() - started) * 1000
            - float(diagnostics.get("generation_elapsed_ms", 0.0)),
            3,
        )
        return GroundedAnswer(
            query=result.query,
            answerable=result.answerable,
            answer=result.answer,
            claims=result.claims,
            citations=result.citations,
            refusal_reason=result.refusal_reason,
            validation=result.validation,
            context=result.context,
            diagnostics=diagnostics,
        )
