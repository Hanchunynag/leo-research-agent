"""Dense Retrieval 业务层依赖的最小 Embedding 契约。"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """屏蔽本地模型和远程 Embedding API 的调用差异。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量，输出顺序必须与输入一致。"""

        ...

    def embed_query(self, query: str) -> list[float]:
        """生成单个查询向量。"""

        ...
