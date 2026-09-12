"""Shared, framework-agnostic Skill Runtime and task routing contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from app.scholar.writing.models import (
    CapabilityProfile,
    CapabilitySet,
    TaskType,
)


class SkillRuntimeError(RuntimeError):
    """A stable error at the Skill boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapabilityDenied(SkillRuntimeError):
    def __init__(self, skill_name: str, capability: str) -> None:
        super().__init__("CAPABILITY_DENIED", f"{skill_name} 不允许 {capability}。")


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    task_type: TaskType
    required_context: tuple[str, ...]
    capabilities: CapabilityProfile
    completion_criteria: tuple[str, ...]
    skill_path: Path

    @property
    def text(self) -> str:
        return self.skill_path.read_text(encoding="utf-8") if self.skill_path.is_file() else ""


class SkillRegistry:
    """只登记 Skill 元数据；不执行 Research、Draft 或数据库操作。"""

    def __init__(self, definitions: tuple[SkillDefinition, ...] = ()) -> None:
        self._definitions = {value.task_type: value for value in definitions}

    def register(self, definition: SkillDefinition) -> None:
        if definition.task_type in self._definitions:
            raise ValueError(f"SKILL_ALREADY_REGISTERED: {definition.task_type}")
        self._definitions[definition.task_type] = definition

    def get(self, task_type: TaskType) -> SkillDefinition:
        try:
            return self._definitions[task_type]
        except KeyError as error:
            raise SkillRuntimeError("SKILL_NOT_AVAILABLE", f"Skill 不可用：{task_type}") from error

    def list(self) -> tuple[SkillDefinition, ...]:
        return tuple(self._definitions.values())

    @classmethod
    def default(cls, project_root: Path) -> "SkillRegistry":
        root = project_root.expanduser().resolve()
        read = {
            "workspace.read",
            "manuscript.read",
            "facts.read",
            "contributions.read",
        }
        intro = CapabilityProfile(
            skill_name="write-introduction",
            allowed=frozenset(
                read
                | {
                    "research.local",
                    "research.web",
                    "evidence.read",
                    "citation.resolve",
                    "bibliography.read",
                    "bibliography.propose",
                    "reviewer.run",
                    "patch.create",
                }
            ),
        )
        support = CapabilityProfile(
            skill_name="support-claim",
            allowed=frozenset({"research.local", "research.web", "evidence.read", "citation.resolve", "bibliography.read", "bibliography.propose"}),
        )
        synthesis = {
            "manuscript.read",
            "facts.read",
            "contributions.read",
            "reviewer.run",
            "patch.create",
        }
        conclusion = CapabilityProfile(skill_name="write-conclusion", allowed=frozenset(synthesis))
        abstract = CapabilityProfile(skill_name="write-abstract", allowed=frozenset(synthesis))
        return cls(
            (
                SkillDefinition(
                    "write-introduction",
                    "WRITE_INTRODUCTION",
                    ("current_section", "facts", "confirmed_contributions", "research_result"),
                    intro,
                    ("current base hash", "ClaimPlan", "ReviewReport", "unapplied DraftPatch"),
                    root / "skills" / "write-introduction" / "SKILL.md",
                ),
                SkillDefinition(
                    "support-claim",
                    "SUPPORT_CLAIM",
                    ("user_claim", "research_result"),
                    support,
                    ("verified locators", "claim/evidence relation", "unresolved when insufficient"),
                    root / "skills" / "support-claim" / "SKILL.md",
                ),
                SkillDefinition(
                    "write-conclusion",
                    "WRITE_CONCLUSION",
                    ("manuscript_snapshot", "facts", "confirmed_contributions"),
                    conclusion,
                    ("latest upstream sections", "review report", "unapplied DraftPatch"),
                    root / "skills" / "write-conclusion" / "SKILL.md",
                ),
                SkillDefinition(
                    "write-abstract",
                    "WRITE_ABSTRACT",
                    ("manuscript_snapshot", "abstract_fact_set", "facts", "confirmed_contributions"),
                    abstract,
                    ("latest results", "AbstractFactSet", "review report", "unapplied DraftPatch"),
                    root / "skills" / "write-abstract" / "SKILL.md",
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class SkillExecutionContext:
    project_id: str
    task_type: TaskType
    target_section: str | None = None
    session_id: str | None = None
    user_instruction: str = ""
    current_section: str = ""
    current_hash: str | None = None
    manuscript_sections: Mapping[str, str] = field(default_factory=dict)
    facts: tuple[Any, ...] = ()
    contributions: tuple[Any, ...] = ()
    research_result: Any | None = None
    abstract_fact_set: Any | None = None
    skill_metadata: Mapping[str, Any] = field(default_factory=dict)
    capabilities: CapabilityProfile = field(default_factory=CapabilitySet)

    def require(self, capability: str) -> None:
        if not self.capabilities.allows(capability):
            raise CapabilityDenied(self.capabilities.skill_name, capability)

    def public_mapping(self) -> dict[str, Any]:
        """Expose only the generic context; no Store, Service or RAG object."""

        output: dict[str, Any] = {
            "project_id": self.project_id,
            "task_type": self.task_type,
            "target_section": self.target_section,
            "session_id": self.session_id,
            "user_instruction": self.user_instruction,
            "current_section": self.current_section,
            "current_hash": self.current_hash,
            "manuscript_sections": dict(self.manuscript_sections),
            "facts": self.facts,
            "confirmed_contributions": self.contributions,
            "skill_metadata": dict(self.skill_metadata),
            "capabilities": self.capabilities,
        }
        if self.research_result is not None:
            output["research_result"] = self.research_result
        if self.abstract_fact_set is not None:
            output["abstract_fact_set"] = self.abstract_fact_set
        return output


@dataclass(frozen=True, slots=True)
class SkillResult:
    task_type: TaskType
    skill_name: str
    status: str
    value: Any | None = None
    error_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    task_type: TaskType | None
    confidence: float
    needs_user_choice: bool = False


class TaskRouter:
    """只选择 Skill；不产生 Research direction、Contribution 或副作用。"""

    _KEYWORDS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
        ("SUPPORT_CLAIM", ("support claim", "证明", "能否支持", "找文献")),
        ("WRITE_CONCLUSION", ("conclusion", "结论", "总结结论")),
        ("WRITE_ABSTRACT", ("abstract", "摘要")),
        ("WRITE_INTRODUCTION", ("introduction", "引言", "intro")),
    )

    def route(self, *, task_type: str | None = None, instruction: str = "") -> RoutingDecision:
        if task_type is not None:
            allowed = {"WRITE_INTRODUCTION", "SUPPORT_CLAIM", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}
            if task_type not in allowed:
                raise SkillRuntimeError("UNSUPPORTED_TASK", f"不支持的 task_type：{task_type}")
            return RoutingDecision(task_type, 1.0)  # type: ignore[arg-type]
        normalized = instruction.casefold()
        matches = [value for value, words in self._KEYWORDS if any(word.casefold() in normalized for word in words)]
        if len(matches) == 1:
            return RoutingDecision(matches[0], 0.75)
        return RoutingDecision(None, 0.0, needs_user_choice=True)
