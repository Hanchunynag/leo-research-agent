"""Official/Shadow 知识引擎配置、验收门禁和可审计切换。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from app.knowledge_engine.generations import IndexGenerationRepository
from app.storage import write_json_atomic


EngineName = Literal["legacy", "lightrag", "none"]
SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class KnowledgeServingConfig:
    official_engine: Literal["legacy", "lightrag"] = "legacy"
    official_generation_id: str | None = None
    shadow_engine: EngineName = "lightrag"
    shadow_generation_id: str | None = None
    revision: int = 1
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.official_engine == "lightrag" and not self.official_generation_id:
            raise ValueError("LightRAG Official 必须显式绑定 generation_id。")
        if self.official_engine == "legacy" and self.official_generation_id is not None:
            raise ValueError("Legacy Official 不接受 generation_id。")
        if self.shadow_engine == "lightrag" and not self.shadow_generation_id:
            # 初始兼容配置允许尚未组装 Shadow；持久化前由 repository 校验。
            if self.updated_at is not None:
                raise ValueError("LightRAG Shadow 必须显式绑定 generation_id。")
        if self.shadow_engine != "lightrag" and self.shadow_generation_id is not None:
            raise ValueError("只有 LightRAG Shadow 接受 generation_id。")
        if self.revision < 1:
            raise ValueError("Serving config revision 必须为正数。")


class KnowledgeServingConfigRepository:
    """以原子 JSON 保存服务快照，并以 JSONL 追加审计记录。"""

    def __init__(self, project_root: Path, *, path: Path | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.path = path or self.project_root / "data" / "knowledge" / "serving_config.json"
        self.audit_path = self.path.with_name("serving_audit.jsonl")

    def load(self) -> KnowledgeServingConfig:
        if not self.path.is_file():
            return KnowledgeServingConfig(shadow_engine="none")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Knowledge serving config schema 不受支持。")
        config = payload.get("knowledge")
        if not isinstance(config, Mapping):
            raise ValueError("Knowledge serving config 缺少 knowledge。")
        return KnowledgeServingConfig(**dict(config))

    def save(
        self,
        config: KnowledgeServingConfig,
        *,
        operation: str,
        actor: str,
        previous: KnowledgeServingConfig | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> KnowledgeServingConfig:
        current = previous or self.load()
        stored = replace(config, revision=current.revision + 1, updated_at=_now())
        # 触发 dataclass 的 fail-closed 校验后再原子替换。
        write_json_atomic(
            self.path,
            {"schema_version": SCHEMA_VERSION, "knowledge": asdict(stored)},
        )
        audit = {
            "timestamp": stored.updated_at,
            "operation": operation,
            "actor": actor,
            "from": asdict(current),
            "to": asdict(stored),
            "details": dict(details or {}),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
        return stored

    def audit_records(self) -> tuple[Mapping[str, Any], ...]:
        if not self.audit_path.is_file():
            return ()
        return tuple(
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


class EngineCutoverService:
    """唯一可写切换入口；未通过 Acceptance 时拒绝 LightRAG Official。"""

    def __init__(
        self,
        repository: KnowledgeServingConfigRepository,
        generations: IndexGenerationRepository,
    ) -> None:
        self.repository = repository
        self.generations = generations

    def _require_servable_generation(self, generation_id: str) -> Any:
        generation = self.generations.get(generation_id)
        if generation is None:
            raise KeyError(f"IndexGeneration 不存在：{generation_id}")
        if generation.state not in {"active", "retired"}:
            raise ValueError(
                f"Generation {generation_id} 状态为 {generation.state}，不能用于服务。"
            )
        return generation

    @staticmethod
    def _require_acceptance(acceptance: Mapping[str, Any]) -> None:
        if acceptance.get("passed") is not True or acceptance.get(
            "official_cutover_approved"
        ) is not True:
            failures = acceptance.get("failures")
            raise PermissionError(f"LightRAG Cutover Acceptance 未通过：{failures!r}")

    def switch_to_lightrag(
        self,
        generation_id: str,
        acceptance: Mapping[str, Any],
        *,
        actor: str,
    ) -> KnowledgeServingConfig:
        self._require_acceptance(acceptance)
        generation = self._require_servable_generation(generation_id)
        current = self.repository.load()
        target = replace(
            current,
            official_engine="lightrag",
            official_generation_id=generation_id,
        )
        return self.repository.save(
            target,
            operation="cutover",
            actor=actor,
            previous=current,
            details={"profile_id": generation.index_profile_id},
        )

    def switch_to_legacy(self, *, actor: str, reason: str) -> KnowledgeServingConfig:
        current = self.repository.load()
        target = replace(
            current,
            official_engine="legacy",
            official_generation_id=None,
        )
        return self.repository.save(
            target,
            operation="rollback",
            actor=actor,
            previous=current,
            details={"reason": reason},
        )

    def configure_shadow(
        self,
        engine: EngineName,
        generation_id: str | None,
        *,
        actor: str,
    ) -> KnowledgeServingConfig:
        if engine == "lightrag":
            if not generation_id:
                raise ValueError("LightRAG Shadow 必须显式绑定 Generation。")
            self._require_servable_generation(generation_id)
        elif generation_id is not None:
            raise ValueError("Legacy/none Shadow 不接受 Generation。")
        current = self.repository.load()
        target = replace(
            current,
            shadow_engine=engine,
            shadow_generation_id=generation_id,
        )
        return self.repository.save(
            target,
            operation="shadow_configured",
            actor=actor,
            previous=current,
        )

