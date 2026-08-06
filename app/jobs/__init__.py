"""持久化后台任务 Repository 与 Worker。"""

from app.jobs.repository import JobRecord, PersistentJobRepository

__all__ = ["JobRecord", "PersistentJobRepository"]
