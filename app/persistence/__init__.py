"""Structured persistence boundary for the research knowledge layer.

MySQL is the production fact store.  JSON/JSONL remains a rebuildable
compatibility projection and is used whenever the database is disabled or
temporarily unavailable.
"""

from app.persistence.mysql import (
    MySQLConfig,
    MySQLKnowledgeRepository,
    build_knowledge_repository,
)
from app.persistence.sync import sync_knowledge_to_mysql

__all__ = [
    "MySQLConfig",
    "MySQLKnowledgeRepository",
    "build_knowledge_repository",
    "sync_knowledge_to_mysql",
]
