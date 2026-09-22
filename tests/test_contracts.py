from __future__ import annotations

from typing import Any

import pytest

from app.contracts import EvidenceMapper, EvidenceRequest


def request(**overrides: Any) -> EvidenceRequest:
    values: dict[str, Any] = {
        "request_id": "REQ-001",
        "workspace_id": "WS-LEO",
        "scope_version": 3,
        "query": "Which inertial measurements aid the state prediction?",
        "top_k": 2,
    }
    values.update(overrides)
    return EvidenceRequest(**values)


def test_workspace_and_scope_are_required_by_evidence_request() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        request(workspace_id="")
    with pytest.raises(ValueError, match="scope_version"):
        request(scope_version=0)


def test_evidence_mapper_rejects_untraceable_candidate() -> None:
    mapper = EvidenceMapper()
    req = request()
    valid = mapper.candidate_from_mapping(
        req,
        {
            "chunk_id": "C-1",
            "work_id": "W-1",
            "document_id": "D-1",
            "page_start": 2,
            "page_end": 2,
            "block_ids": ["B-1"],
            "content": "Traceable evidence.",
        },
        fallback_rank=1,
    )
    invalid = mapper.candidate_from_mapping(
        req,
        {"chunk_id": "C-2", "content": "No canonical locator."},
        fallback_rank=2,
    )

    bundle = mapper.bundle(req, [valid, invalid])

    assert [item.chunk_id for item in bundle.evidence] == ["C-1"]
    assert bundle.rejected_candidate_ids == ("C-2",)
    assert bundle.evidence[0].verification_method == "canonical_locator_fields"
