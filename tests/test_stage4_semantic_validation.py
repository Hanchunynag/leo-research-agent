from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.research import TieredClaimEvidenceValidator


def _evidence(
    evidence_id: str,
    content: str,
    *,
    document_id: str = "D_1",
    grade: str = "primary",
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "evidence_state": "selected",
        "workspace_id": "default",
        "scope_version": 1,
        "document_id": document_id,
        "chunk_id": f"{document_id}_c1",
        "content": content,
        "evidence_grade": grade,
        "directness": "direct",
    }


def _draft(text: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "answerable": True,
        "claims": [
            {"claim_id": "C1", "text": text, "evidence_ids": evidence_ids}
        ],
    }


class LocalResult:
    def __init__(self, label: str, confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def validate(
        self, claim: str, evidence: Sequence[str]
    ) -> tuple[Any, float, str]:
        self.calls.append((claim, tuple(evidence)))
        return self.label, self.confidence, "fixture local semantic result"


class Judge:
    def __init__(self, label: str = "supports") -> None:
        self.label = label
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def judge(
        self, claim: str, evidence: Sequence[str], risk_types: Sequence[str]
    ) -> Mapping[str, Any]:
        self.calls.append((claim, tuple(evidence), tuple(risk_types)))
        return {
            "label": self.label,
            "confidence": 0.91,
            "reason": "structured judgment",
            "rewritten_answer": "This field must be ignored.",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }


def test_local_semantic_support_uses_only_claim_and_cited_evidence() -> None:
    local = LocalResult("supports")
    validator = TieredClaimEvidenceValidator(local_validator=local)
    evidence = [_evidence("E1", "Carrier phase estimates ephemeris error.")]

    report = validator.validate(
        _draft("Carrier phase estimates ephemeris error.", ["E1"]), evidence
    )

    assert report.valid is True
    assert report.judge_call_count == 0
    assert local.calls == [
        (
            "Carrier phase estimates ephemeris error.",
            ("Carrier phase estimates ephemeris error.",),
        )
    ]
    assert report.semantic_results[0].label == "supports"


def test_local_contradiction_removes_claim_without_extra_generation() -> None:
    validator = TieredClaimEvidenceValidator(
        local_validator=LocalResult("contradicts")
    )
    draft = _draft(
        "Fixed weights improve positioning accuracy.",
        ["E1"],
    )
    report = validator.validate(
        draft,
        [_evidence("E1", "Fixed weights do not improve positioning accuracy.")],
    )
    repaired = validator.deterministic_repair(draft, report)

    assert report.valid is False
    assert report.issues[0].code == "semantic_contradiction"
    assert repaired["answerable"] is False
    assert repaired["claims"] == []


def test_high_risk_causal_claim_invokes_structured_judge_at_most_once() -> None:
    local = LocalResult("supports")
    judge = Judge()
    validator = TieredClaimEvidenceValidator(
        local_validator=local, high_risk_judge=judge
    )
    draft = {
        "answerable": True,
        "claims": [
            {
                "claim_id": "C1",
                "text": "Adaptive weighting improves positioning accuracy.",
                "evidence_ids": ["E1"],
            },
            {
                "claim_id": "C2",
                "text": "Robust weighting is associated with lower positioning error.",
                "evidence_ids": ["E2"],
            },
        ],
    }
    evidence = [
        _evidence("E1", "Adaptive weighting improves positioning accuracy."),
        _evidence(
            "E2", "Robust weighting is associated with lower positioning error."
        ),
    ]

    report = validator.validate(draft, evidence)

    assert report.valid is True
    assert report.judge_call_count == 1
    assert len(judge.calls) == 1
    assert "causal_claim" in judge.calls[0][2]
    assert report.semantic_input_tokens >= 12
    assert report.semantic_output_tokens == 4
    assert all(
        not hasattr(value, "rewritten_answer") for value in report.semantic_results
    )


def test_partial_high_risk_claim_without_judge_becomes_hypothesis() -> None:
    validator = TieredClaimEvidenceValidator(
        local_validator=LocalResult("partially_supports", confidence=0.5)
    )
    draft = _draft(
        "Adaptive noise may improve positioning accuracy.", ["E1"]
    )
    evidence = [
        _evidence("E1", "Adaptive noise relates to positioning accuracy.")
    ]

    report = validator.validate(draft, evidence)
    repaired = validator.deterministic_repair(draft, report)

    assert report.valid is False
    assert report.issues[0].code == "high_risk_semantic_judge_unavailable"
    assert repaired["claims"][0]["validation_status"] == "hypothesis"
    assert repaired["claims"][0]["text"].startswith("待验证假设：")


def test_graph_inference_still_fails_deterministic_layer_before_semantic() -> None:
    local = LocalResult("supports")
    validator = TieredClaimEvidenceValidator(local_validator=local)

    report = validator.validate(
        _draft("Graph relation proves the mechanism.", ["E1"]),
        [
            _evidence(
                "E1", "Graph relation suggests the mechanism.", grade="graph_inference"
            )
        ],
    )

    assert report.valid is False
    assert report.issues[0].code == "graph_inference_as_fact"
    assert local.calls == []
