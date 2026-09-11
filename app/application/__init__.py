"""稳定的应用层 Facade。"""

from app.application.legacy import LegacySessionAdapter
from app.application.research_facade import ResearchApplicationFacade

__all__ = ["LegacySessionAdapter", "ResearchApplicationFacade"]
