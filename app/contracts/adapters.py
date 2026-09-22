"""Adapters between retrieval projections and canonical evidence contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from app.contracts.domain import (
    CandidateEvidence,
    EvidenceRequest,
    VerifiedEvidence,
    VerifiedEvidenceBundle,
)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


class EvidenceMapper:
    """Map retrieval dictionaries into traceable evidence contracts."""

    def candidate_from_mapping(
        self,
        request: EvidenceRequest,
        value: Mapping[str, Any],
        *,
        fallback_rank: int,
    ) -> CandidateEvidence:
        chunk_id = str(value.get("chunk_id") or "")
        candidate_id = str(value.get("evidence_id") or chunk_id or f"candidate-{fallback_rank}")
        raw_score = value.get("score")
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
            else 0.0
        )
        return CandidateEvidence(
            candidate_id=candidate_id,
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            content=str(value.get("content") or ""),
            score=score,
            retrieval_source=str(value.get("retrieval_source") or "local_hybrid"),
            work_id=str(value.get("work_id") or "") or None,
            document_id=str(value.get("document_id") or "") or None,
            chunk_id=chunk_id or None,
            section_id=str(value.get("section_id") or "") or None,
            page_start=(int(value["page_start"]) if value.get("page_start") else None),
            page_end=(int(value["page_end"]) if value.get("page_end") else None),
            block_ids=_strings(value.get("block_ids")),
            metadata=dict(value),
        )

    def verified_from_candidate(self, candidate: CandidateEvidence) -> VerifiedEvidence | None:
        if not all(
            (
                candidate.work_id,
                candidate.document_id,
                candidate.chunk_id,
                candidate.content,
                candidate.page_start,
                candidate.page_end,
                candidate.block_ids,
            )
        ):
            return None
        assert candidate.work_id is not None
        assert candidate.document_id is not None
        assert candidate.chunk_id is not None
        assert candidate.page_start is not None
        assert candidate.page_end is not None
        return VerifiedEvidence(
            evidence_id=str(candidate.metadata.get("evidence_id") or candidate.candidate_id),
            candidate_id=candidate.candidate_id,
            request_id=candidate.request_id,
            workspace_id=candidate.workspace_id,
            scope_version=candidate.scope_version,
            content=candidate.content,
            work_id=candidate.work_id,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            block_ids=candidate.block_ids,
            verification_method="canonical_locator_fields",
            content_hash=hashlib.sha256(candidate.content.encode("utf-8")).hexdigest(),
            metadata=candidate.metadata,
        )

    def bundle(
        self,
        request: EvidenceRequest,
        candidates: Sequence[CandidateEvidence],
    ) -> VerifiedEvidenceBundle:
        verified: list[VerifiedEvidence] = []
        rejected: list[str] = []
        for candidate in candidates:
            value = self.verified_from_candidate(candidate)
            if value is None:
                rejected.append(candidate.candidate_id)
            else:
                verified.append(value)
        issues = (("evidence missing canonical locator fields",) if rejected else ())
        return VerifiedEvidenceBundle(
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            evidence=tuple(verified),
            rejected_candidate_ids=tuple(rejected),
            verification_issues=issues,
        )
