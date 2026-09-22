"""Shared Scholar Skill Runtime; routing only selects an existing Skill path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.scholar.project import ScholarProjectStore
from app.scholar.writing.models import WritingRequest
from app.scholar.writing.runtime import (
    SkillRegistry,
    SkillResult,
    SkillRuntimeError,
    TaskRouter,
)
from app.scholar.writing.service import ScholarWritingService
from app.scholar.writing.support import ClaimSupportResult, ClaimSupportService, SupportClaimRequest
from app.scholar.writing.synthesis import SynthesisWritingService, SynthesisWriter


class ScholarSkillRuntime:
    """One runtime boundary for Introduction, Support Claim, Conclusion and Abstract."""

    def __init__(
        self,
        project_root: Path,
        *,
        registry: SkillRegistry | None = None,
        project_store: ScholarProjectStore | None = None,
        introduction: ScholarWritingService | None = None,
        support_claim: ClaimSupportService | None = None,
        synthesis: SynthesisWritingService | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)
        self.registry = registry or SkillRegistry.default(self.project_root)
        self.router = TaskRouter()
        self.introduction = introduction
        self.support_claim_service = support_claim
        self.synthesis = synthesis

    def route(self, *, task_type: str | None = None, instruction: str = ""):
        return self.router.route(task_type=task_type, instruction=instruction)

    def write_introduction(self, request: WritingRequest, writer: Any) -> SkillResult:
        definition = self.registry.get("WRITE_INTRODUCTION")
        service = self.introduction
        if service is None:
            raise SkillRuntimeError("SKILL_NOT_AVAILABLE", "Introduction Runtime 未注入。")
        service.capabilities = definition.capabilities  # type: ignore[assignment]
        result = service.write_introduction(request, writer=writer)
        return SkillResult("WRITE_INTRODUCTION", definition.name, result.status, result, result.error_codes, result.warnings)

    def support_claim(self, request: SupportClaimRequest) -> SkillResult:
        definition = self.registry.get("SUPPORT_CLAIM")
        service = self.support_claim_service
        if service is None:
            raise SkillRuntimeError("SKILL_NOT_AVAILABLE", "Support Claim Runtime 未注入。")
        try:
            for capability in ("LOCAL_RESEARCH", "READ_EVIDENCE", "RESOLVE_CITATION"):
                definition.capabilities.require(capability)
        except PermissionError:
            return SkillResult("SUPPORT_CLAIM", definition.name, "FAILED", error_codes=("CAPABILITY_DENIED",))
        result = service.support_claim(request)
        if not isinstance(result, ClaimSupportResult):
            raise TypeError("Support Claim Runtime 必须返回 ClaimSupportResult。")
        return SkillResult("SUPPORT_CLAIM", definition.name, "READY", result)

    def write_synthesis(self, request: WritingRequest, writer: SynthesisWriter) -> SkillResult:
        if request.task_type not in {"WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            raise SkillRuntimeError("UNSUPPORTED_TASK", "Synthesis Runtime 只支持 Conclusion/Abstract。")
        definition = self.registry.get(request.task_type)
        if self.synthesis is None:
            raise SkillRuntimeError("SKILL_NOT_AVAILABLE", "Synthesis Runtime 未注入。")
        result = self.synthesis.write(request, writer, definition)
        return SkillResult(request.task_type, definition.name, result.status, result, result.error_codes, result.warnings)

    def execute(self, request: Any, *, writer: Any | None = None) -> SkillResult:
        if isinstance(request, SupportClaimRequest):
            return self.support_claim(request)
        if not isinstance(request, WritingRequest):
            raise SkillRuntimeError("UNSUPPORTED_TASK", "未知 Scholar Skill Request。")
        if request.task_type == "WRITE_INTRODUCTION":
            if writer is None:
                raise SkillRuntimeError("SKILL_NOT_AVAILABLE", "Introduction Writer 未提供。")
            return self.write_introduction(request, writer)
        if request.task_type in {"WRITE_CONCLUSION", "WRITE_ABSTRACT"}:
            if writer is None:
                raise SkillRuntimeError("SKILL_NOT_AVAILABLE", "Synthesis Writer 未提供。")
            return self.write_synthesis(request, writer)
        raise SkillRuntimeError("UNSUPPORTED_TASK", f"不支持的 task_type：{request.task_type}")
