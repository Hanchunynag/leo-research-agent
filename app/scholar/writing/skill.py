"""Introduction Skill：学术 SOP 和 Research Need 拆分。"""

from __future__ import annotations

from pathlib import Path

from app.scholar.writing.models import ResearchNeed, WritingRequest


class IntroductionSkill:
    name = "write-introduction"

    def __init__(self, skill_path: Path | None = None) -> None:
        self.skill_path = skill_path or Path("skills/write-introduction/SKILL.md")

    @property
    def text(self) -> str:
        return self.skill_path.read_text(encoding="utf-8") if self.skill_path.is_file() else ""

    def build_research_needs(self, request: WritingRequest) -> tuple[ResearchNeed, ...]:
        focus = (request.focus or request.instruction).strip()
        normalized = f"{request.instruction} {request.focus or ''}".casefold()
        needs: list[ResearchNeed] = []

        def add(need_id: str, move: str, purpose: str, query: str, claim: str) -> None:
            needs.append(ResearchNeed(need_id, move, query, (claim,), purpose))

        explicit_prior = any(token in normalized for token in ("prior", "existing", "已有", "相关工作", "现有工作"))
        explicit_problem = any(token in normalized for token in ("problem", "challenge", "问题", "挑战", "背景"))
        if explicit_problem or not explicit_prior:
            add(
                "background_problem",
                "technical_problem",
                "background",
                f"{focus}: scientific and technical problem",
                f"The scientific and technical problem relevant to {focus}.",
            )
        add(
            "prior_work",
            "existing_approaches",
            "prior_work",
            f"{focus}: existing approaches and prior work",
            f"Existing approaches relevant to {focus}.",
        )
        add(
            "limitations",
            "limitations",
            "prior_work",
            f"{focus}: limitations and unresolved research gaps",
            f"Limitations or unresolved issues in existing approaches for {focus}.",
        )
        return tuple(needs)
