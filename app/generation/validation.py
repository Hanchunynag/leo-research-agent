"""证据包完整性与 claim 级引用校验。"""

from __future__ import annotations

import re

from app.context.models import ContextBundle, EvidenceItem
from app.generation.models import (
    AnswerDraft,
    CitationRecord,
    CitationValidationIssue,
    CitationValidationReport,
)
from app.indexing.tokenization import token_count


SOURCE_ID_PATTERN = re.compile(r"S[1-9][0-9]*")


def _issue(
    code: str,
    message: str,
    *,
    claim_id: str | None = None,
    source_id: str | None = None,
) -> CitationValidationIssue:
    return CitationValidationIssue(code, message, claim_id, source_id)


def _valid_evidence_metadata(item: EvidenceItem) -> bool:
    return bool(
        SOURCE_ID_PATTERN.fullmatch(item.source_id)
        and item.chunk_id.strip()
        and item.work_id.strip()
        and item.document_id.strip()
        and item.title.strip()
        and item.content.strip()
        and item.block_ids
        and all(block_id.strip() for block_id in item.block_ids)
        and item.page_start >= 1
        and item.page_end >= item.page_start
    )


def validate_context_bundle(context: ContextBundle) -> list[CitationValidationIssue]:
    """生成前验证来源身份、预算和序列化文本是否一致。"""

    issues: list[CitationValidationIssue] = []
    if not context.query.strip():
        issues.append(_issue("empty_query", "证据包查询为空。"))
    if context.retrieval_mode not in {"fast", "accurate"}:
        issues.append(_issue("invalid_retrieval_mode", "证据包检索模式不合法。"))
    if not context.evidence:
        issues.append(_issue("empty_context", "证据包中没有可用证据。"))
    if not context.context_text.strip():
        issues.append(_issue("empty_context_text", "证据包文本为空。"))
    if (
        context.token_budget < 100
        or context.token_budget > 100_000
        or context.token_count < 0
        or context.token_count > context.token_budget
    ):
        issues.append(_issue("invalid_token_budget", "证据包超出 token budget。"))
    actual_token_count = token_count(context.context_text)
    if context.token_count != actual_token_count:
        issues.append(
            _issue("token_count_mismatch", "证据包记录的 token 数与文本不一致。")
        )

    source_ids = [item.source_id for item in context.evidence]
    if len(source_ids) != len(set(source_ids)):
        issues.append(_issue("duplicate_source_id", "证据包包含重复 source ID。"))
    for item in context.evidence:
        if not _valid_evidence_metadata(item):
            issues.append(
                _issue(
                    "invalid_source_metadata",
                    "来源元数据不完整或不合法。",
                    source_id=item.source_id or None,
                )
            )
        if f"[{item.source_id}]" not in context.context_text:
            issues.append(
                _issue(
                    "source_missing_from_context_text",
                    "来源未出现在证据包文本中。",
                    source_id=item.source_id,
                )
            )
    return issues


def validate_answer_draft(
    draft: AnswerDraft,
    context: ContextBundle,
) -> CitationValidationReport:
    """只接受能映射到当前证据包的、逐条有引用的结构化回答。"""

    issues: list[CitationValidationIssue] = []
    citations: list[CitationRecord] = []
    evidence_by_source = {item.source_id: item for item in context.evidence}

    if not draft.answerable:
        if draft.claims:
            issues.append(_issue("refusal_has_claims", "拒答不能同时包含 claims。"))
        if not isinstance(draft.refusal_reason, str) or not draft.refusal_reason.strip():
            issues.append(_issue("missing_refusal_reason", "拒答必须说明原因。"))
        return CitationValidationReport(not issues, issues, citations)

    if isinstance(draft.refusal_reason, str) and draft.refusal_reason.strip():
        issues.append(_issue("answer_has_refusal_reason", "可回答结果不能包含拒答原因。"))
    if not draft.claims:
        issues.append(_issue("missing_claims", "可回答结果必须至少包含一个 claim。"))

    seen_claim_ids: set[str] = set()
    for claim in draft.claims:
        if not isinstance(claim.claim_id, str) or not isinstance(claim.text, str):
            issues.append(_issue("invalid_claim", "claim ID 和文本必须是字符串。"))
            continue
        claim_id = claim.claim_id.strip()
        if not claim_id:
            issues.append(_issue("empty_claim_id", "claim ID 不能为空。"))
        elif claim_id in seen_claim_ids:
            issues.append(
                _issue("duplicate_claim_id", "claim ID 不能重复。", claim_id=claim_id)
            )
        else:
            seen_claim_ids.add(claim_id)
        if not claim.text.strip():
            issues.append(
                _issue("empty_claim_text", "claim 文本不能为空。", claim_id=claim_id or None)
            )
        if not isinstance(claim.source_ids, list) or not all(
            isinstance(source_id, str) for source_id in claim.source_ids
        ):
            issues.append(
                _issue(
                    "invalid_source_ids",
                    "claim 的 source IDs 必须是字符串数组。",
                    claim_id=claim_id or None,
                )
            )
            continue
        if not claim.source_ids:
            issues.append(
                _issue(
                    "claim_without_citation",
                    "每个 claim 必须至少引用一个来源。",
                    claim_id=claim_id or None,
                )
            )
            continue
        if len(claim.source_ids) != len(set(claim.source_ids)):
            issues.append(
                _issue(
                    "duplicate_claim_source",
                    "同一 claim 不能重复引用同一来源。",
                    claim_id=claim_id or None,
                )
            )
        for source_id_value in claim.source_ids:
            source_id = source_id_value.strip()
            evidence = evidence_by_source.get(source_id)
            if not source_id:
                issues.append(
                    _issue(
                        "empty_source_id",
                        "引用 source ID 不能为空。",
                        claim_id=claim_id or None,
                    )
                )
                continue
            if evidence is None:
                issues.append(
                    _issue(
                        "unknown_source_id",
                        "引用不属于当前证据包。",
                        claim_id=claim_id or None,
                        source_id=source_id,
                    )
                )
                continue
            citations.append(
                CitationRecord(
                    claim_id=claim_id,
                    source_id=source_id,
                    chunk_id=evidence.chunk_id,
                    work_id=evidence.work_id,
                    document_id=evidence.document_id,
                    title=evidence.title,
                    section_path=evidence.section_path,
                    page_start=evidence.page_start,
                    page_end=evidence.page_end,
                    block_ids=evidence.block_ids,
                    evidence_id=evidence.evidence_id,
                )
            )

    if issues:
        citations = []
    return CitationValidationReport(not issues, issues, citations)
