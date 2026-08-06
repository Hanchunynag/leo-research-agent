"""按 Router/Generator 阶段最小化上下文，禁止候选证据和工具日志泄漏。"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from app.indexing.tokenization import token_count
from app.research.harness import ResearchRunHarness


ContextAudience = Literal["router", "generator"]


@dataclass(frozen=True, slots=True)
class PhaseContextPack:
    context_id: str
    audience: ContextAudience
    query: str
    workspace_id: str
    scope_version: int
    workspace_summary: Mapping[str, Any] = field(default_factory=dict)
    recent_conversation: tuple[Mapping[str, str], ...] = ()
    scope_constraints: tuple[str, ...] = ()
    selected_evidence: tuple[Mapping[str, Any], ...] = ()
    conflicts: tuple[Mapping[str, Any], ...] = ()
    output_format: Mapping[str, Any] = field(default_factory=dict)
    token_count: int = 0


_EVIDENCE_FIELDS = (
    "source_id",
    "evidence_id",
    "document_id",
    "chunk_id",
    "page_start",
    "page_end",
    "block_ids",
    "content",
    "evidence_grade",
    "directness",
    "graph_inference_disclaimer",
)


class ResearchContextManager:
    def router_pack(
        self,
        *,
        query: str,
        workspace_id: str,
        scope_version: int,
        workspace_summary: Mapping[str, Any],
        recent_conversation: Sequence[Mapping[str, str]] = (),
        harness: ResearchRunHarness | None = None,
    ) -> PhaseContextPack:
        recent = tuple(
            {"role": str(value.get("role") or ""), "content": str(value.get("content") or "")[:800]}
            for value in recent_conversation[-4:]
        )
        rendered = f"{query}\n{dict(workspace_summary)}\n{recent}"
        count = token_count(rendered)
        if harness is not None:
            harness.consume("context_tokens", count)
            harness.consume("total_tokens", count)
        return PhaseContextPack(
            context_id=f"RC_{secrets.token_hex(8)}",
            audience="router",
            query=query,
            workspace_id=workspace_id,
            scope_version=scope_version,
            workspace_summary=dict(workspace_summary),
            recent_conversation=recent,
            token_count=count,
        )

    def generator_pack(
        self,
        *,
        query: str,
        workspace_id: str,
        scope_version: int,
        scope_constraints: Sequence[str],
        evidence: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]],
        output_format: Mapping[str, Any],
        harness: ResearchRunHarness | None = None,
    ) -> PhaseContextPack:
        selected: list[dict[str, Any]] = []
        for index, value in enumerate(evidence, 1):
            state = value.get("evidence_state") or value.get("state")
            if state != "selected":
                raise ValueError("Generator Context 只能包含 selected Evidence。")
            safe = {key: value.get(key) for key in _EVIDENCE_FIELDS if value.get(key) is not None}
            safe.setdefault("source_id", f"S{index}")
            safe.setdefault("evidence_id", str(value.get("evidence_id") or safe["source_id"]))
            selected.append(safe)
        rendered = f"{query}\n{tuple(scope_constraints)}\n{selected}\n{list(conflicts)}\n{dict(output_format)}"
        count = token_count(rendered)
        if harness is not None:
            harness.consume("context_tokens", count)
            harness.consume("total_tokens", count)
        return PhaseContextPack(
            context_id=f"GC_{secrets.token_hex(8)}",
            audience="generator",
            query=query,
            workspace_id=workspace_id,
            scope_version=scope_version,
            scope_constraints=tuple(scope_constraints),
            selected_evidence=tuple(selected),
            conflicts=tuple(dict(value) for value in conflicts),
            output_format=dict(output_format),
            token_count=count,
        )
