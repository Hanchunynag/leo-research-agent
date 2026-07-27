"""与具体 Cross-Encoder 解耦的精排接口与实现。"""

from app.reranking.base import RerankerProvider
from app.reranking.bge import BGERerankerConfig, BGERerankerProvider

__all__ = ["BGERerankerConfig", "BGERerankerProvider", "RerankerProvider"]
