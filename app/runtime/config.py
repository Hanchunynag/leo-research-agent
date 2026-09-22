"""Configuration owned by the CrewAI Scholar production runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RuntimeMode = Literal["production", "test", "local-fast"]


@dataclass(frozen=True, slots=True)
class ScholarRuntimeConfig:
    """Small, production-only configuration surface for Scholar Runtime."""

    runtime_mode: RuntimeMode = "production"
    scholar_max_manager_steps: int = 12

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "ScholarRuntimeConfig":
        """Read the production ``LEO_SCHOLAR_*`` settings."""

        values: dict[str, str] = {}
        if project_root is not None:
            env_file = project_root.expanduser().resolve() / ".env"
            if env_file.is_file():
                try:
                    for line in env_file.read_text(encoding="utf-8").splitlines():
                        cleaned = line.strip()
                        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
                            continue
                        key, _, value = cleaned.partition("=")
                        if key.strip().startswith("LEO_SCHOLAR_"):
                            values[key.strip()] = value.strip().strip("'\"")
                except OSError:
                    pass

        def raw(name: str) -> str | None:
            return os.getenv(f"LEO_SCHOLAR_{name}", values.get(f"LEO_SCHOLAR_{name}"))

        mode = raw("RUNTIME_MODE") or "production"
        steps = int(raw("MAX_MANAGER_STEPS") or 12)
        return cls(runtime_mode=mode, scholar_max_manager_steps=steps)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.runtime_mode not in {"production", "test", "local-fast"}:
            raise ValueError("runtime_mode 必须是 production、test 或 local-fast。")
        if not 3 <= self.scholar_max_manager_steps <= 32:
            raise ValueError("scholar_max_manager_steps 必须在 3 到 32 之间。")

    @property
    def scholar_runtime_mode(self) -> RuntimeMode:
        return self.runtime_mode
