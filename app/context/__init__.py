"""将检索结果组装为具有严格来源边界的 LLM 证据包。"""

from app.context.assembly import assemble_context_bundle
from app.context.models import ContextBundle, EvidenceItem

__all__ = ["ContextBundle", "EvidenceItem", "assemble_context_bundle"]
