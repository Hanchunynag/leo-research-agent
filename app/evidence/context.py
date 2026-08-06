"""只接受 selected 证据的 ContextBuilder。"""

from __future__ import annotations

import secrets
from collections.abc import Sequence

from app.contracts import ContextPack, EvidenceRequest, SelectedEvidence


class SelectedEvidenceContextBuilder:
    def build(self, request: EvidenceRequest, evidence: Sequence[SelectedEvidence], *, token_budget: int) -> ContextPack:
        if any(value.evidence.state != "selected" for value in evidence):
            raise ValueError("只有 selected evidence 可以进入 Context Builder。")
        lines = [
            f"[E{index}] {value.evidence.document_id} pp. {value.evidence.page_start}-{value.evidence.page_end}\n{value.evidence.content}"
            for index, value in enumerate(evidence, 1)
        ]
        tokens = sum(value.token_count for value in evidence)
        return ContextPack(
            context_id=f"CTX_{secrets.token_hex(8)}",
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            selected_evidence=tuple(evidence),
            rendered_context="\n\n".join(lines),
            token_count=tokens,
            token_budget=token_budget,
            diagnostics={"selected_only": True},
        )
