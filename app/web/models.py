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
    force_new_topic: bool = False
    include_context: bool = True


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
    status: Literal["queued", "running", "succeeded", "failed"]
    events: list[JobEvent]
    result: dict[str, Any] | None = None
    error: str | None = None


class JobCreated(WebModel):
    job_id: str
    status: Literal["queued"] = "queued"
