"""Context budgets for Deep Agents Harness audiences.

The budget is a runtime concern.  Skill Markdown may describe context needs,
but it cannot enlarge these code-level limits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScholarContextBudget:
    supervisor: int = 4_000
    research: int = 4_000
    reviewer: int = 8_000
    total: int = 16_000

    def __post_init__(self) -> None:
        if min(self.supervisor, self.research, self.reviewer, self.total) < 1:
            raise ValueError("Scholar context budget 必须大于 0。")
        if max(self.supervisor, self.research, self.reviewer) > self.total:
            raise ValueError("单个 Scholar context budget 不能超过 total。")

    def limit_for(self, audience: str) -> int:
        return {
            "supervisor": self.supervisor,
            "research_subagent": self.research,
            "reviewer_subagent": self.reviewer,
        }.get(audience, self.total)
