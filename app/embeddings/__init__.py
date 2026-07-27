"""与具体模型和 API 解耦的向量表示接口。"""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider

__all__ = ["BGEM3Config", "BGEM3EmbeddingProvider", "EmbeddingProvider"]
