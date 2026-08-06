"""Claim—Evidence 确定性验证与无新增事实修复。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.indexing.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class ClaimIssue:
    claim_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ClaimValidationReport:
    valid: bool
    issues: tuple[ClaimIssue, ...]
    supported_claim_ids: tuple[str, ...]


class ClaimEvidenceValidator:
    def validate(
        self,
        draft: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        *,
        scope_constraints: Sequence[str] = (),
        conflicts: Sequence[Mapping[str, Any]] = (),
    ) -> ClaimValidationReport:
        evidence_by_id = {
            str(value.get("evidence_id")): value
            for value in evidence
            if (value.get("evidence_state") or value.get("state")) == "selected"
        }
        raw_claims = draft.get("claims")
        claims = raw_claims if isinstance(raw_claims, list) else []
        issues: list[ClaimIssue] = []
        supported: list[str] = []
        for index, value in enumerate(claims, 1):
            if not isinstance(value, Mapping):
                issues.append(ClaimIssue(f"C{index}", "invalid_claim", "Claim 必须是对象。"))
                continue
            claim_id = str(value.get("claim_id") or f"C{index}")
            text = str(value.get("text") or "").strip()
            ids = value.get("evidence_ids")
            cited = [str(item) for item in ids] if isinstance(ids, list) else []
            if not text:
                issues.append(ClaimIssue(claim_id, "empty_claim", "事实 Claim 不能为空。"))
                continue
            if not cited:
                issues.append(ClaimIssue(claim_id, "missing_evidence", "事实 Claim 必须引用 Selected Evidence。"))
                continue
            unknown = [item for item in cited if item not in evidence_by_id]
            if unknown:
                issues.append(ClaimIssue(claim_id, "candidate_or_unknown_evidence", "Claim 引用了非 Selected Evidence。"))
                continue
            cited_values = [evidence_by_id[item] for item in cited]
            # 先阻断高风险语义误述，再执行通用的词面支持代理检查。
            # 否则 graph inference/analogy 可能只显示为泛化的“不支持”，
            # 不利于选择正确的降级措辞与恢复动作。
            if any(value.get("evidence_grade") == "graph_inference" for value in cited_values):
                if not re.search(r"可能|提示|推断|may|might|suggest|inferred", text, re.IGNORECASE):
                    issues.append(ClaimIssue(claim_id, "graph_inference_as_fact", "图推断被写成了直接事实。"))
                    continue
            if any(value.get("evidence_grade") == "analogy" for value in cited_values):
                if not re.search(r"类比|analog|可能|suggest", text, re.IGNORECASE):
                    issues.append(ClaimIssue(claim_id, "analogy_as_domain_fact", "跨领域类比被写成本领域结论。"))
                    continue
            claim_tokens = {value for value in tokenize(text) if len(value) > 2}
            evidence_tokens = {
                token
                for cited_value in cited_values
                for token in tokenize(str(cited_value.get("content") or ""))
                if len(token) > 2
            }
            if claim_tokens and not claim_tokens.intersection(evidence_tokens):
                issues.append(ClaimIssue(claim_id, "not_supported_by_evidence", "引用证据与 Claim 没有可验证的内容重合。"))
                continue
            if conflicts and not bool(draft.get("conflicts_acknowledged")):
                issues.append(ClaimIssue(claim_id, "conflict_ignored", "答案未说明冲突证据。"))
                continue
            if scope_constraints and bool(value.get("outside_scope")):
                issues.append(ClaimIssue(claim_id, "outside_scope", "Claim 超出 Scope。"))
                continue
            supported.append(claim_id)
        if bool(draft.get("answerable")) and not claims:
            issues.append(ClaimIssue("$", "answer_without_claims", "确定性答案必须包含可验证 Claim。"))
        if not evidence and bool(draft.get("answerable")):
            issues.append(ClaimIssue("$", "answer_without_evidence", "无证据时不得生成确定性科研结论。"))
        return ClaimValidationReport(not issues, tuple(issues), tuple(supported))

    def deterministic_repair(
        self,
        draft: Mapping[str, Any],
        report: ClaimValidationReport,
    ) -> dict[str, Any]:
        invalid = {value.claim_id for value in report.issues}
        claims = [
            dict(value)
            for value in draft.get("claims", [])
            if isinstance(value, Mapping) and str(value.get("claim_id")) not in invalid
        ]
        if not claims:
            return {
                "answerable": False,
                "claims": [],
                "refusal_reason": "现有 Selected Evidence 不足以支持确定性科研结论。",
                "recovery": "unsupported_claims_removed",
            }
        return {
            **dict(draft),
            "claims": claims,
            "recovery": "unsupported_claims_removed",
        }
