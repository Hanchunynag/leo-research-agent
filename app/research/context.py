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
    "paper_id",
    "document_id",
    "chunk_id",
    "title",
    "authors",
    "year",
    "section",
    "section_path",
    "page_start",
    "page_end",
    "block_ids",
    "content",
    "evidence_grade",
    "directness",
    "graph_inference_disclaimer",
)


def _truncate_to_tokens(value: str, maximum: int) -> str:
    """Keep evidence bounded while preserving both its beginning and ending."""

    text = value.strip()
    if maximum < 1 or token_count(text) <= maximum:
        return text
    marker = "\n[… evidence truncated …]\n"
    # Character slicing is only the search for a candidate boundary; the final
    # token_count check makes the limit deterministic for both English and CJK.
    target_chars = max(32, int(len(text) * maximum / max(token_count(text), 1)))
    while target_chars > 32:
        head_chars = max(16, int(target_chars * 0.70))
        tail_chars = max(16, target_chars - head_chars)
        candidate = text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()
        if token_count(candidate) <= maximum:
            return candidate
        target_chars = int(target_chars * 0.90)
    return text[: max(1, target_chars)]


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
            harness.record_context("router", count)
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
        max_evidence: int = 8,
        max_evidence_per_paper: int = 2,
        max_content_tokens: int = 220,
    ) -> PhaseContextPack:
        if max_evidence < 1 or max_evidence_per_paper < 1 or max_content_tokens < 1:
            raise ValueError("Generator Context 压缩参数必须大于 0。")
        selected: list[dict[str, Any]] = []
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        group_order: list[str] = []
        for value in evidence:
            state = value.get("evidence_state") or value.get("state")
            if state != "selected":
                raise ValueError("Generator Context 只能包含 selected Evidence。")
            group = str(
                value.get("paper_id")
                or value.get("document_id")
                or value.get("work_id")
                or "unknown-paper"
            )
            if group not in grouped:
                grouped[group] = []
                group_order.append(group)
            if len(grouped[group]) < max_evidence_per_paper:
                grouped[group].append(value)
        # Round-robin across papers prevents a high-scoring paper from
        # consuming the complete generation context in timeline/compare tasks.
        balanced: list[Mapping[str, Any]] = []
        for ordinal in range(max_evidence_per_paper):
            for group in group_order:
                values = grouped[group]
                if ordinal < len(values):
                    balanced.append(values[ordinal])
                    if len(balanced) >= max_evidence:
                        break
            if len(balanced) >= max_evidence:
                break
        for index, value in enumerate(balanced, 1):
            safe = {key: value.get(key) for key in _EVIDENCE_FIELDS if value.get(key) is not None}
            safe.setdefault("source_id", f"S{index}")
            safe.setdefault("evidence_id", str(value.get("evidence_id") or safe["source_id"]))
            safe["content"] = _truncate_to_tokens(str(safe.get("content") or ""), max_content_tokens)
            selected.append(safe)
        rendered = f"{query}\n{tuple(scope_constraints)}\n{selected}\n{list(conflicts)}\n{dict(output_format)}"
        count = token_count(rendered)
        if harness is not None:
            harness.record_context("generator", count)
            selected_tokens = token_count(
                "\n".join(str(value.get("content") or "") for value in selected)
            )
            harness.context_usage["selected_evidence"] = (
                harness.context_usage.get("selected_evidence", 0) + selected_tokens
            )
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
