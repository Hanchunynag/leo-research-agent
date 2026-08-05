"""Coverage 与语义验证的保守确定性回退实现。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agentic.models import (
    AgenticAnswerDraft,
    AgenticClaim,
    CoverageItem,
    CoverageReport,
    EntailmentLabel,
    EvidenceStatus,
    QueryPlan,
    RepairAction,
    SemanticClaimResult,
    SemanticValidationReport,
)
from app.indexing.tokenization import normalize_search_text


_SYNTHESIS_MARKERS: dict[str, tuple[str, ...]] = {
    "SQ1": (
        "propose",
        "develop",
        "presented",
        "framework",
        "scheme",
        "approach",
        "method",
        "estimate",
        "tracking",
        "correction",
        "compensation",
    ),
    "SQ2": (
        "measurement",
        "observable",
        "pseudorange",
        "carrier phase",
        "doppler",
        "model",
        "state",
        "estimate",
        "filter",
        "ekf",
        "nls",
    ),
    "SQ3": (
        "joint",
        "clock",
        "prediction",
        "neural",
        "experiment",
        "validation",
        "navigation",
        "application",
        "pnt",
    ),
}


def _synthesis_coverage(
    plan: QueryPlan,
    evidence: Sequence[dict[str, Any]],
) -> CoverageReport:
    """以多篇论文各自的直接方法证据组合评估跨文献综述。"""

    direct = [
        item for item in evidence if int(item.get("directness_grade") or 0) >= 2
    ]
    direct_work_ids = {
        str(item.get("work_id") or item.get("document_id") or "")
        for item in direct
        if item.get("work_id") or item.get("document_id")
    }
    has_cross_paper_basis = len(direct_work_ids) >= 2
    items: list[CoverageItem] = []
    followups: list[str] = []
    for subquestion in plan.subquestions:
        markers = _SYNTHESIS_MARKERS.get(subquestion.id, ())
        matched = [
            item
            for item in direct
            if not markers
            or any(
                marker
                in normalize_search_text(
                    " ".join(
                        (
                            str(item.get("title") or ""),
                            " ".join(item.get("section_path") or []),
                            str(item.get("content") or ""),
                        )
                    )
                )
                for marker in markers
            )
        ]
        # 按 work 去重，避免同一篇论文的多个 Chunk 伪装成跨文献覆盖。
        supporting: list[str] = []
        seen_works: set[str] = set()
        for item in matched:
            work_id = str(item.get("work_id") or item.get("document_id") or "")
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id or work_id in seen_works:
                continue
            supporting.append(evidence_id)
            seen_works.add(work_id)
        sufficient = has_cross_paper_basis and bool(supporting)
        if sufficient:
            status: EvidenceStatus = "sufficient"
            missing = ""
        elif direct:
            status = "partial"
            missing = "已找到单篇论文的方法证据，但缺少跨论文的组合覆盖。"
            followups.append(subquestion.question + " proposed method extension")
        else:
            status = "missing"
            missing = "没有找到直接描述论文方法、贡献或扩展的证据。"
            followups.append(subquestion.question + " abstract method conclusion")
        items.append(
            CoverageItem(
                subquestion_id=subquestion.id,
                status=status,
                supporting_evidence_ids=supporting,
                missing_information=missing,
            )
        )
    return CoverageReport(
        overall_sufficient=all(item.status == "sufficient" for item in items),
        coverage=items,
        followup_queries=list(dict.fromkeys(followups)),
    )


def deterministic_coverage(
    plan: QueryPlan,
    evidence: Sequence[dict[str, Any]],
) -> CoverageReport:
    """只把 directness >= 2 且包含子问题关键概念的证据视为覆盖。"""

    if plan.intent == "synthesis" and plan.target_category == "method":
        return _synthesis_coverage(plan, evidence)

    items: list[CoverageItem] = []
    followups: list[str] = []
    for subquestion in plan.subquestions:
        normalized_question = normalize_search_text(subquestion.question)
        requires_clock = any(
            value in normalized_question for value in ("时钟", "钟漂", "clock")
        )
        requires_ephemeris = any(
            value in normalized_question for value in ("星历", "ephemeris")
        )
        supporting: list[str] = []
        partial = False
        for item in evidence:
            text = normalize_search_text(str(item.get("content") or ""))
            grade = int(item.get("directness_grade") or 0)
            concept_match = (
                (not requires_clock or any(v in text for v in ("clock", "时钟", "drift")))
                and (
                    not requires_ephemeris
                    or any(v in text for v in ("ephemer", "星历", "orbit error"))
                )
            )
            if concept_match and grade >= 2:
                evidence_id = str(item.get("evidence_id") or "")
                if evidence_id:
                    supporting.append(evidence_id)
            elif concept_match or grade >= 1:
                partial = True
        if supporting:
            status: EvidenceStatus = "sufficient"
            missing = ""
        elif partial:
            status = "partial"
            missing = "现有证据只有背景或间接支持，缺少直接表述。"
            followups.append(subquestion.question + " direct evidence")
        else:
            status = "missing"
            missing = "没有找到直接支持该子问题的证据。"
            followups.append(subquestion.question + " measurement estimate")
        items.append(
            CoverageItem(
                subquestion_id=subquestion.id,
                status=status,
                supporting_evidence_ids=list(dict.fromkeys(supporting)),
                missing_information=missing,
            )
        )
    return CoverageReport(
        overall_sufficient=all(item.status == "sufficient" for item in items),
        coverage=items,
        followup_queries=list(dict.fromkeys(followups)),
    )


def deterministic_semantic_validation(
    query: str,
    plan: QueryPlan,
    draft: AgenticAnswerDraft,
    evidence: Sequence[dict[str, Any]],
    *,
    structural_valid: bool,
) -> SemanticValidationReport:
    """在 LLM Validator 不可用时保守识别类别混淆和间接引用。"""

    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence
        if item.get("evidence_id")
    }
    results: list[SemanticClaimResult] = []
    issues: list[str] = []
    for claim in draft.claims:
        normalized = normalize_search_text(claim.text)
        cited = [evidence_by_id.get(value) for value in claim.evidence_ids]
        known = [item for item in cited if item is not None]
        predicted_as_measurement = (
            claim.category in {"measurement", "observable"}
            and any(value in normalized for value in ("预测星历", "predicted ephemer", "sgp4"))
        )
        category_correct = (
            claim.category == plan.target_category and not predicted_as_measurement
        )
        all_citations_known = bool(cited) and len(known) == len(cited)
        citation_direct = all_citations_known and all(
            int(item.get("directness_grade") or 0) >= 2 for item in known
        )
        query_aligned = category_correct
        if not all_citations_known:
            entailment: EntailmentLabel = "not_entailed"
            action: RepairAction = "drop"
            reason = "Claim 包含无法解析的稳定 evidence_id。"
        elif predicted_as_measurement:
            entailment = "partially_entailed"
            action = "drop"
            reason = "证据把预测星历描述为输入或先验，而不是观测量。"
        elif not category_correct:
            entailment = "partially_entailed"
            action = "rewrite"
            reason = "Claim 类别与用户要求的 target_category 不一致。"
        elif not citation_direct:
            entailment = "partially_entailed"
            action = "retrieve_more"
            reason = "引用只有主题相关性，缺少直接支持。"
        else:
            entailment = "entailed"
            action = "keep"
            reason = "类别、问题方向和直接引用均通过保守检查。"
        result = SemanticClaimResult(
            claim_id=claim.claim_id,
            entailment=entailment,
            query_aligned=query_aligned,
            category_correct=category_correct,
            citation_direct=citation_direct,
            reason=reason,
            repair_action=action,
            revised_claim=None,
        )
        results.append(result)
        if action != "keep":
            issues.append(f"{claim.claim_id}: {reason}")
    semantic_valid = bool(results) and all(
        item.entailment == "entailed"
        and item.query_aligned
        and item.category_correct
        and item.citation_direct
        for item in results
    )
    return SemanticValidationReport(
        valid=structural_valid and semantic_valid,
        issues=issues,
        structural_valid=structural_valid,
        semantic_valid=semantic_valid,
        claim_results=results,
        requires_retrieval=any(
            result.repair_action == "retrieve_more" for result in results
        ),
        followup_queries=[query] if any(
            result.repair_action == "retrieve_more" for result in results
        ) else [],
    )


def deterministic_repair(
    draft: AgenticAnswerDraft,
    validation: SemanticValidationReport,
) -> AgenticAnswerDraft:
    """最多一次地保留、缩写或删除 Claim，不引入新事实。"""

    results = {item.claim_id: item for item in validation.claim_results}
    claims: list[AgenticClaim] = []
    for claim in draft.claims:
        result = results.get(claim.claim_id)
        if result is None or result.repair_action in {"drop", "retrieve_more"}:
            continue
        text = result.revised_claim or claim.text
        claims.append(claim.model_copy(update={"text": text}))
    if not claims:
        return AgenticAnswerDraft(
            answerable=False,
            claims=[],
            refusal_reason="现有论文证据不足以形成通过语义验证的回答。",
        )
    return AgenticAnswerDraft(answerable=True, claims=claims, refusal_reason=None)
