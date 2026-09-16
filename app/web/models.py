"""Web API 的结构化请求与任务契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerRequest(WebModel):
    """一次 Agentic 问答请求，Session ID 为空时由核心服务创建。"""

    query: str = Field(min_length=1, max_length=8_000)
    session_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)
    force_new_topic: bool = False
    include_context: bool = True


class ResumeRequest(WebModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_input: str = Field(min_length=1, max_length=8_000)


class ParseOptions(WebModel):
    """PDF 上传后的 MinerU 解析选项。"""

    method: Literal["auto", "txt", "ocr"] = "auto"
    backend: Literal[
        "pipeline",
        "vlm-engine",
        "hybrid-engine",
        "vlm-http-client",
        "hybrid-http-client",
    ] = "pipeline"
    formula_enabled: bool = True
    table_enabled: bool = True
    force_mineru: bool = False


class JobEvent(WebModel):
    sequence: int
    stage: str
    message: str
    progress: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class JobSnapshot(WebModel):
    job_id: str
    kind: Literal["answer", "parse"]
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    events: list[JobEvent]
    result: dict[str, Any] | None = None
    error: str | None = None


class JobCreated(WebModel):
    job_id: str
    status: Literal["queued"] = "queued"


class PatchDecisionRequest(WebModel):
    project_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    expected_base_hash: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=256)


class ScholarTaskRequest(WebModel):
    """User-level Scholar request; execution is owned by ScholarHarnessService."""

    instruction: str = Field(min_length=1, max_length=16_000)
    project_id: str = Field(min_length=1, max_length=128)
    task_type: Literal[
        "RESEARCH",
        "WRITE_INTRODUCTION",
        "SUPPORT_CLAIM",
        "WRITE_CONCLUSION",
        "WRITE_ABSTRACT",
        "REVIEW",
    ] | None = None
    session_id: str | None = Field(default=None, max_length=128)
    thread_id: str | None = Field(default=None, max_length=128)


class ScholarResumeRequest(WebModel):
    """Framework checkpoint resume data, never Patch Approval data."""

    project_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)
    instruction: str = Field(min_length=1, max_length=16_000)
    resume_value: Any
    task_type: Literal[
        "RESEARCH",
        "WRITE_INTRODUCTION",
        "SUPPORT_CLAIM",
        "WRITE_CONCLUSION",
        "WRITE_ABSTRACT",
        "REVIEW",
    ] | None = None
    session_id: str | None = Field(default=None, max_length=128)


class ScholarRunCreateRequest(WebModel):
    """Asynchronous Scholar Run request; the body contains no secrets."""

    instruction: str = Field(min_length=1, max_length=16_000)
    project_id: str = Field(min_length=1, max_length=128)
    task_type: Literal[
        "RESEARCH",
        "WRITE_INTRODUCTION",
        "SUPPORT_CLAIM",
        "WRITE_CONCLUSION",
        "WRITE_ABSTRACT",
        "REVIEW",
    ] | None = None
    session_id: str | None = Field(default=None, max_length=128)
    thread_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=256)


class ScholarRunResumeRequest(WebModel):
    """Queue a durable approval/checkpoint reconciliation."""

    resume_value: Any = None
    idempotency_key: str | None = Field(default=None, max_length=256)


class BuildReportRequest(WebModel):
    build_id: str = Field(min_length=1, max_length=128)
    status: Literal["SUCCESS", "FAILED", "UNAVAILABLE", "UNKNOWN", "BUILD_TRIGGERED"]
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    completed_at: str | None = None
    message: str = ""
