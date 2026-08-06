"""Agent 可见的统一知识服务边界。"""

from app.knowledge_engine.unified_service import UnifiedKnowledgeService, build_legacy_unified_service

__all__ = [
    "UnifiedKnowledgeService",
    "build_legacy_unified_service",
]
