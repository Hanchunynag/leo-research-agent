"""只从本地环境与项目 `.env` 读取回答模型配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalLLMSettings(BaseSettings):
    """环境变量优先于 `.env`，密钥以 SecretStr 保存。"""

    model_config = SettingsConfigDict(
        env_prefix="LEO_LLM_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LEO_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    timeout_seconds: float = 120.0
    max_tokens: int = 1200
    prompt_layout: Literal["query_first", "context_first"] | None = None


def load_local_llm_settings(project_root: Path) -> LocalLLMSettings:
    """读取项目根目录 `.env`；文件不存在时只读取进程环境变量。"""

    return LocalLLMSettings(_env_file=project_root / ".env")  # type: ignore[call-arg]
