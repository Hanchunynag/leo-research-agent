"""Web API 的结构化请求与任务契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    kind: Literal["parse"]
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
