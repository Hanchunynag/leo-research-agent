"""生成业务层依赖的最小回答模型契约。"""

from __future__ import annotations

from typing import Protocol

from app.context.models import ContextBundle
from app.generation.models import AnswerDraft


class AnswerProvider(Protocol):
    """根据一个经过边界校验的证据包生成结构化回答草稿。"""

    def generate(self, query: str, context: ContextBundle) -> AnswerDraft:
        """返回 claim 级引用，不直接渲染最终回答。"""

        ...
