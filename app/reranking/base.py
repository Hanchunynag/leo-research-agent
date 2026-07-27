"""检索业务层依赖的最小 Reranker 契约。"""

from __future__ import annotations

from typing import Protocol


class RerankerProvider(Protocol):
    """为一个查询与一组候选文档生成可排序相关性分数。"""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 顺序严格一致的相关性分数。"""

        ...
