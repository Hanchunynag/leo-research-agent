"""Thin CrewAI Tools over existing Scholar application capabilities.

Tools are an authorization boundary, not a place for retrieval, citation,
writing or approval logic.  The gateway supplied by the backend owns those
domain calls and this module only enforces the agent capability matrix and a
per-run tool budget.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings="none")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(child) for child in value]
    return value


class CapabilityMatrix:
    """Code-visible permission matrix; prompts are never the authority."""

    SUPERVISOR = frozenset()
    RESEARCH = frozenset(
        {"LOCAL_RESEARCH", "WEB_RESEARCH", "READ_EVIDENCE", "RESOLVE_CITATION"}
    )
    WRITER = frozenset(
        {
            "READ_MANUSCRIPT",
            "READ_FACTS",
            "READ_CONTRIBUTIONS",
            "READ_EVIDENCE",
            "RESOLVE_CITATION",
            "CREATE_DRAFT_PATCH",
        }
    )
    REVIEWER = frozenset(
        {
            "READ_MANUSCRIPT",
            "READ_FACTS",
            "READ_CONTRIBUTIONS",
            "READ_EVIDENCE",
            "RESOLVE_CITATION",
            "REVIEW",
        }
    )

    @classmethod
    def as_dict(cls) -> dict[str, list[str]]:
        return {
            "supervisor": sorted(cls.SUPERVISOR),
            "research": sorted(cls.RESEARCH),
            "writer": sorted(cls.WRITER),
            "reviewer": sorted(cls.REVIEWER),
            "human_approval": ["APPROVE_PATCH", "SAFE_APPLY"],
        }


class _ResearchArgs(BaseModel):
    query: str
    purpose: str = "background"
    target_claim: str | None = None
    freshness_mode: str = "LOCAL_ONLY"
    requested_from: str | None = None
    requested_to: str | None = None
    explicit_latest: bool = False


class _WriterArgs(BaseModel):
    instruction: str


class _ReviewerArgs(BaseModel):
    patch_id: str


class ResearchCapabilityTool(BaseTool):
    name: str = "research_capability"
    description: str = "Run bounded local/web research through ResearchCapabilityService and return verified evidence references."
    args_schema: type[BaseModel] = _ResearchArgs
    _gateway: Any = PrivateAttr()

    def __init__(self, gateway: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gateway = gateway

    def _run(
        self,
        query: str,
        purpose: str = "background",
        target_claim: str | None = None,
        freshness_mode: str = "LOCAL_ONLY",
        requested_from: str | None = None,
        requested_to: str | None = None,
        explicit_latest: bool = False,
    ) -> str:
        result = self._gateway.tool_research(
            query=query,
            purpose=purpose,
            target_claim=target_claim,
            freshness_mode=freshness_mode,
            requested_from=requested_from,
            requested_to=requested_to,
            explicit_latest=explicit_latest,
        )
        return json.dumps(_jsonable(result), ensure_ascii=False, default=str)

    def execute(self, **kwargs: Any) -> Any:
        return self._gateway.tool_research(**kwargs)


class WriterCapabilityTool(BaseTool):
    name: str = "writer_capability"
    description: str = "Create an immutable DraftPatch through the existing Scholar Writing and Skill Runtime; never applies it."
    args_schema: type[BaseModel] = _WriterArgs
    _gateway: Any = PrivateAttr()

    def __init__(self, gateway: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gateway = gateway

    def _run(self, instruction: str) -> str:
        result = self._gateway.tool_write(instruction=instruction)
        return json.dumps(_jsonable(result), ensure_ascii=False, default=str)

    def execute(self, **kwargs: Any) -> Any:
        return self._gateway.tool_write(**kwargs)


class ReviewerCapabilityTool(BaseTool):
    name: str = "review_capability"
    description: str = "Validate a DraftPatch with the existing deterministic reviewer; it cannot write, approve, or apply."
    args_schema: type[BaseModel] = _ReviewerArgs
    _gateway: Any = PrivateAttr()

    def __init__(self, gateway: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gateway = gateway

    def _run(self, patch_id: str) -> str:
        result = self._gateway.tool_review(patch_id=patch_id)
        return json.dumps(_jsonable(result), ensure_ascii=False, default=str)

    def execute(self, **kwargs: Any) -> Any:
        return self._gateway.tool_review(**kwargs)
