"""统一 Legacy Hybrid RAG 知识服务与索引编排入口。"""

from app.knowledge_engine.index_service import KnowledgeIndexService
from app.knowledge_engine.unified_service import (
    UnifiedKnowledgeService,
    build_configured_unified_service,
    build_legacy_unified_service,
    knowledge_runtime_status,
)

__all__ = [
    "KnowledgeIndexService",
    "UnifiedKnowledgeService",
    "build_configured_unified_service",
    "build_legacy_unified_service",
    "knowledge_runtime_status",
]
