"""Research Workspace 与版本化 Scope 隔离。"""

from app.workspaces.repository import WorkspaceRepository
from app.workspaces.service import DEFAULT_WORKSPACE_ID, WorkspaceService

__all__ = ["DEFAULT_WORKSPACE_ID", "WorkspaceRepository", "WorkspaceService"]
