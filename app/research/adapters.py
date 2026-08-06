"""现有结构化回答 Provider 到阶段三最小 Generator Context 的适配。"""

from __future__ import annotations

import json
from typing import Any

from app.research.context import PhaseContextPack


class AgenticReasoningGeneratorAdapter:
    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def generate(self, context: PhaseContextPack) -> dict[str, Any]:
        if context.audience != "generator":
            raise ValueError("Answer Generator 只能接收 generator ContextPack。")
        payload = {
            "query": context.query,
            "workspace_id": context.workspace_id,
            "scope_version": context.scope_version,
            "scope_constraints": list(context.scope_constraints),
            "selected_evidence": list(context.selected_evidence),
            "conflicts": list(context.conflicts),
            "output_format": dict(context.output_format),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Use only selected_evidence. Return structured atomic claims. "
                    "Every claim must cite source_ids and evidence_ids. Do not state graph "
                    "inference or analogy as a directly proven fact. If evidence is insufficient, refuse."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        draft, diagnostics = self.provider.generate_answer(messages)
        result = draft.model_dump(mode="json")
        raw_usage = diagnostics.get("usage") if isinstance(diagnostics, dict) else None
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        if "total_tokens" not in usage:
            usage["total_tokens"] = int(usage.get("prompt_tokens") or 0) + int(
                usage.get("completion_tokens") or 0
            )
        result["usage"] = usage
        return result
