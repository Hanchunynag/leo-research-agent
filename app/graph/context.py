"""Graph-aware context rendering with claims and original source chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.graph.models import EvidenceCandidate


def build_graph_context(
    candidates: Sequence[EvidenceCandidate], chunks_by_key: dict[str, dict[str, Any]],
) -> str:
    sections: list[str] = []
    rendered_claims: set[str] = set()
    rendered_chunks: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_type == "graph_path":
            sections.append(
                f"[GRAPH PATH {candidate.evidence_id}]\n"
                f"Mode: {candidate.metadata.get('mode', 'inferred_path')}\n{candidate.text}\n"
                f"Edges: {', '.join(candidate.source_claim_ids)}"
            )
        claims = candidate.metadata.get("edge_claims") or [candidate.metadata.get("claim")]
        for claim in claims:
            if not isinstance(claim, dict) or not claim.get("claim_id"):
                continue
            claim_id = str(claim["claim_id"])
            if claim_id in rendered_claims:
                continue
            rendered_claims.add(claim_id)
            sections.append(
                f"[RELATION CLAIM {claim_id}]\nPredicate: {claim.get('predicate')}\n"
                f"Qualifiers: {claim.get('qualifiers_json') or '{}'}\n"
                f"Polarity: {claim.get('polarity')}\nEvidence: {claim.get('evidence_text')}\n"
                f"Source chunk: {claim.get('chunk_key')}"
            )
        for chunk_key in candidate.source_chunk_keys:
            if chunk_key in rendered_chunks or chunk_key not in chunks_by_key:
                continue
            rendered_chunks.add(chunk_key)
            chunk = chunks_by_key[chunk_key]
            sections.append(
                f"[SOURCE {chunk_key}]\nWork: {chunk.get('work_id')}\n"
                f"Document: {chunk.get('document_id')}\nTitle: {chunk.get('title')}\n"
                f"Section: {' > '.join(chunk.get('section_path') or [])}\n"
                f"Pages: {chunk.get('page_start')}-{chunk.get('page_end')}\n"
                f"Chunk ID: {chunk.get('chunk_id')}\n"
                f"Block IDs: {json.dumps(chunk.get('block_ids') or [], ensure_ascii=False)}\n"
                f"Content:\n{chunk.get('content') or ''}"
            )
    return "\n\n".join(sections)
