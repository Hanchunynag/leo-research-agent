"""Claim—Evidence 确定性验证与无新增事实修复。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from app.indexing.tokenization import tokenize


_CROSS_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "观测量": ("observation", "observations", "observable", "observables", "measurement", "measurements"),
    "伪距": ("pseudorange", "range"),
    "伪距率": ("pseudorange rate", "range rate"),
    "载波相位": ("carrier phase", "carrier", "phase"),
    "多普勒": ("doppler", "frequency"),
    "频率": ("frequency",),
    "星历": ("ephemeris", "orbit"),
    "时钟": ("clock", "timing"),
    "钟漂": ("clock drift", "drift"),
    "误差": ("error", "errors"),
    "估计": ("estimate", "estimated", "estimation", "estimates"),
    "低轨卫星": ("leo", "satellite", "space vehicle"),
    "接收机": ("receiver",),
    "使用": ("use", "used", "using", "utilize", "utilizing"),
    "采用": ("use", "used", "using", "utilize", "utilizing"),
}


def _semantic_tokens(value: str) -> set[str]:
    """保留词面校验，同时为中英技术术语补充确定性别名。"""

    normalized = value.casefold()
    tokens = {token for token in tokenize(value) if len(token) > 2}
    for alias, equivalents in _CROSS_LANGUAGE_ALIASES.items():
        if alias in normalized:
            tokens.update(
                token
                for equivalent in equivalents
                for token in tokenize(equivalent)
                if len(token) > 2
            )
    return tokens


def _negation_anchors(value: str) -> set[str]:
    """只提取否定谓词本身，不把 ``no effect`` 等条件短语当成矛盾。"""

    anchors: set[str] = set()
    normalized = value.casefold()
    for match in re.finditer(
        r"\b(?:not|cannot|can't|never|doesn't|isn't|aren't)\s+([a-z0-9]+)",
        normalized,
    ):
        anchors.update(_semantic_tokens(match.group(1)))
    for verb in ("使用", "采用", "支持", "估计", "包括", "需要", "存在"):
        if f"不{verb}" in normalized:
            anchors.update(_semantic_tokens(verb))
    return anchors


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
    semantic_results: tuple["ClaimSemanticResult", ...] = ()
    judge_call_count: int = 0
    semantic_input_tokens: int = 0
    semantic_output_tokens: int = 0


SemanticLabel = Literal[
    "supports", "partially_supports", "contradicts", "unrelated"
]


@dataclass(frozen=True, slots=True)
class ClaimSemanticResult:
    claim_id: str
    label: SemanticLabel
    confidence: float
    layer: Literal["local", "judge"]
    reason: str


class LocalSemanticValidator(Protocol):
    def validate(
        self, claim: str, evidence: Sequence[str]
    ) -> tuple[SemanticLabel, float, str]: ...


class HighRiskSemanticJudge(Protocol):
    def judge(
        self, claim: str, evidence: Sequence[str], risk_types: Sequence[str]
    ) -> Mapping[str, Any]: ...


class HeuristicLocalSemanticValidator:
    """离线轻量语义回退；生产可注入本地 NLI/Cross-Encoder。"""

    def validate(
        self, claim: str, evidence: Sequence[str]
    ) -> tuple[SemanticLabel, float, str]:
        claim_tokens = _semantic_tokens(claim)
        evidence_text = " ".join(evidence)
        evidence_tokens = _semantic_tokens(evidence_text)
        if not claim_tokens or not evidence_tokens:
            return "unrelated", 0.99, "Claim 或 Evidence 缺少可判定语义单元。"
        overlap = len(claim_tokens & evidence_tokens)
        ratio = overlap / max(1, min(len(claim_tokens), len(evidence_tokens)))
        claim_negations = _negation_anchors(claim)
        evidence_negations = _negation_anchors(evidence_text)
        negation_mismatch = bool(
            (claim_negations & evidence_tokens and not evidence_negations)
            or (evidence_negations & claim_tokens and not claim_negations)
        )
        if negation_mismatch and overlap >= 2:
            return "contradicts", 0.85, "Claim 与 Evidence 的否定极性不一致。"
        if overlap >= 2 and ratio >= 0.25:
            return "supports", min(0.95, 0.75 + ratio / 4), "轻量语义锚点一致。"
        if overlap >= 1:
            return "partially_supports", 0.60, "仅部分语义锚点一致。"
        return "unrelated", 0.90, "未找到可对齐的语义锚点。"


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
            relevant_conflicts = [
                conflict
                for conflict in conflicts
                if str(conflict.get("supporting_evidence_id") or "") in cited
                or str(conflict.get("opposing_evidence_id") or "") in cited
            ]
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
            claim_tokens = _semantic_tokens(text)
            evidence_tokens = {
                token
                for cited_value in cited_values
                for token in _semantic_tokens(str(cited_value.get("content") or ""))
            }
            if claim_tokens and not claim_tokens.intersection(evidence_tokens):
                issues.append(ClaimIssue(claim_id, "not_supported_by_evidence", "引用证据与 Claim 没有可验证的内容重合。"))
                continue
            if relevant_conflicts and not bool(draft.get("conflicts_acknowledged")):
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


class TieredClaimEvidenceValidator:
    """确定性 → 本地语义 → 单次高风险 Judge 的保守验证链。"""

    _CAUSAL = re.compile(
        r"导致|引起|提高|改善|因果|cause|causes|lead(?:s)? to|improve(?:s|d)?|result(?:s)? in",
        re.IGNORECASE,
    )
    _NOVELTY = re.compile(
        r"首次|创新|研究空白|novel|first|research gap|state of the art",
        re.IGNORECASE,
    )
    _SYNTHESIS = re.compile(
        r"多篇|跨文献|跨论文|综合|演进|演化|比较|对比|"
        r"across papers?|cross[- ]paper|synthesi[sz]|evolution|progression|"
        r"compar(?:e|ison|ed)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        local_validator: LocalSemanticValidator | None = None,
        high_risk_judge: HighRiskSemanticJudge | None = None,
        low_confidence_threshold: float = 0.65,
    ) -> None:
        self.deterministic = ClaimEvidenceValidator()
        self.local_validator = local_validator or HeuristicLocalSemanticValidator()
        self.high_risk_judge = high_risk_judge
        self.low_confidence_threshold = low_confidence_threshold

    @staticmethod
    def _selected_by_id(
        evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        return {
            str(value.get("evidence_id")): value
            for value in evidence
            if (value.get("evidence_state") or value.get("state")) == "selected"
        }

    def _risks(
        self,
        claim: Mapping[str, Any],
        cited: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]],
        confidence: float,
    ) -> tuple[str, ...]:
        text = str(claim.get("text") or "")
        risks: list[str] = []
        if self._CAUSAL.search(text):
            risks.append("causal_claim")
        if self._NOVELTY.search(text):
            risks.append("novelty_or_research_gap")
        grades = {str(value.get("evidence_grade") or "") for value in cited}
        if "graph_inference" in grades:
            risks.append("graph_inference")
        if "analogy" in grades:
            risks.append("analogy")
        documents = {str(value.get("document_id") or "") for value in cited}
        if len(documents - {""}) > 1 and (
            str(claim.get("category") or "").casefold()
            in {"comparison", "evolution", "synthesis", "综合", "比较"}
            or self._SYNTHESIS.search(text)
        ):
            risks.append("multi_paper_synthesis")
        if conflicts:
            risks.append("conflicting_evidence")
        if confidence < self.low_confidence_threshold:
            risks.append("low_local_confidence")
        return tuple(dict.fromkeys(risks))

    @staticmethod
    def _judge_result(
        claim_id: str, payload: Mapping[str, Any]
    ) -> ClaimSemanticResult:
        label = str(payload.get("label") or "")
        allowed = {
            "supports",
            "partially_supports",
            "contradicts",
            "unrelated",
        }
        if label not in allowed:
            raise ValueError("Semantic Judge label 无效。")
        confidence = float(payload.get("confidence") or 0.0)
        if not 0 <= confidence <= 1:
            raise ValueError("Semantic Judge confidence 无效。")
        return ClaimSemanticResult(
            claim_id=claim_id,
            label=label,  # type: ignore[arg-type]
            confidence=confidence,
            layer="judge",
            reason=str(payload.get("reason") or "高风险结构化判定。"),
        )

    def validate(
        self,
        draft: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        *,
        scope_constraints: Sequence[str] = (),
        conflicts: Sequence[Mapping[str, Any]] = (),
    ) -> ClaimValidationReport:
        layer_one = self.deterministic.validate(
            draft,
            evidence,
            scope_constraints=scope_constraints,
            conflicts=conflicts,
        )
        if not layer_one.valid:
            return layer_one
        evidence_by_id = self._selected_by_id(evidence)
        claims = [
            value for value in draft.get("claims", []) if isinstance(value, Mapping)
        ]
        issues: list[ClaimIssue] = []
        results: list[ClaimSemanticResult] = []
        judge_used = False
        semantic_input_tokens = 0
        semantic_output_tokens = 0
        supported: list[str] = []
        for index, claim in enumerate(claims, 1):
            claim_id = str(claim.get("claim_id") or f"C{index}")
            text = str(claim.get("text") or "")
            ids = claim.get("evidence_ids")
            evidence_ids = [str(value) for value in ids] if isinstance(ids, list) else []
            cited = [evidence_by_id[value] for value in evidence_ids]
            snippets = [str(value.get("content") or "") for value in cited[:4]]
            semantic_input_tokens += len(tokenize(text)) + sum(
                len(tokenize(value)) for value in snippets
            )
            label, confidence, reason = self.local_validator.validate(text, snippets)
            local = ClaimSemanticResult(
                claim_id, label, confidence, "local", reason
            )
            results.append(local)
            relevant_conflicts = [
                conflict
                for conflict in conflicts
                if str(conflict.get("supporting_evidence_id") or "") in evidence_ids
                or str(conflict.get("opposing_evidence_id") or "") in evidence_ids
            ]
            risks = self._risks(claim, cited, relevant_conflicts, confidence)
            final = local
            if risks:
                if self.high_risk_judge is not None and not judge_used:
                    payload = self.high_risk_judge.judge(text, snippets, risks)
                    judge_used = True
                    usage = payload.get("usage")
                    if isinstance(usage, Mapping):
                        semantic_input_tokens += int(usage.get("input_tokens") or 0)
                        semantic_output_tokens += int(usage.get("output_tokens") or 0)
                    final = self._judge_result(claim_id, payload)
                    results.append(final)
                elif label not in {"contradicts", "unrelated"}:
                    issues.append(
                        ClaimIssue(
                            claim_id,
                            "high_risk_semantic_judge_unavailable",
                            "高风险 Claim 缺少可用的单次语义 Judge。",
                        )
                    )
                    continue
            if final.label == "supports":
                supported.append(claim_id)
            elif final.label == "partially_supports":
                issues.append(
                    ClaimIssue(
                        claim_id,
                        "semantic_partial_support",
                        "Evidence 仅部分支持 Claim，必须降低表述力度。",
                    )
                )
            elif final.label == "contradicts":
                issues.append(
                    ClaimIssue(
                        claim_id,
                        "semantic_contradiction",
                        "Evidence 与 Claim 矛盾。",
                    )
                )
            else:
                issues.append(
                    ClaimIssue(
                        claim_id,
                        "semantic_unrelated",
                        "Evidence 与 Claim 在语义上无关。",
                    )
                )
        return ClaimValidationReport(
            valid=not issues,
            issues=tuple(issues),
            supported_claim_ids=tuple(supported),
            semantic_results=tuple(results),
            judge_call_count=int(judge_used),
            semantic_input_tokens=semantic_input_tokens,
            semantic_output_tokens=semantic_output_tokens,
        )

    def deterministic_repair(
        self, draft: Mapping[str, Any], report: ClaimValidationReport
    ) -> dict[str, Any]:
        issue_by_claim = {value.claim_id: value.code for value in report.issues}
        repaired: list[dict[str, Any]] = []
        for value in draft.get("claims", []):
            if not isinstance(value, Mapping):
                continue
            claim = dict(value)
            code = issue_by_claim.get(str(claim.get("claim_id") or ""))
            if code in {"semantic_contradiction", "semantic_unrelated"}:
                continue
            if code == "semantic_partial_support":
                text = str(claim.get("text") or "")
                claim["text"] = "现有证据提示，" + re.sub(
                    r"证明|证实|demonstrates?|proves?", "提示", text, flags=re.IGNORECASE
                )
                claim["validation_status"] = "downgraded"
            elif code == "high_risk_semantic_judge_unavailable":
                claim["text"] = "待验证假设：" + str(claim.get("text") or "")
                claim["validation_status"] = "hypothesis"
            elif code is not None:
                continue
            repaired.append(claim)
        if not repaired:
            return {
                "answerable": False,
                "claims": [],
                "refusal_reason": "语义验证后仍无足够证据形成确定性科研结论。",
                "recovery": "semantic_claims_rejected",
            }
        return {
            **dict(draft),
            "claims": repaired,
            "recovery": "semantic_claims_downgraded",
        }
