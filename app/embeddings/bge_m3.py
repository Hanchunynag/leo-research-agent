"""BGE-M3 的单向量 Dense EmbeddingProvider。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BGE_M3_MODEL = "BAAI/bge-m3"


@dataclass(frozen=True)
class BGEM3Config:
    model_name: str = DEFAULT_BGE_M3_MODEL
    revision: str | None = None
    device: str | None = None
    cache_folder: Path | None = None
    batch_size: int = 8
    normalize_embeddings: bool = True
    local_files_only: bool = False
    show_progress_bar: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name 不能为空。")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于 0。")


class BGEM3EmbeddingProvider:
    """仅使用 BGE-M3 dense 向量，不启用 sparse 或 multi-vector 输出。"""

    provider_id = "sentence-transformers:bge-m3:dense"

    def __init__(
        self,
        config: BGEM3Config | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or BGEM3Config()
        self._model = model

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def revision(self) -> str | None:
        return self.config.revision

    @property
    def normalized(self) -> bool:
        return self.config.normalize_embeddings

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {
                "device": self.config.device,
                "local_files_only": self.config.local_files_only,
            }
            if self.config.cache_folder is not None:
                kwargs["cache_folder"] = str(self.config.cache_folder)
            if self.config.revision:
                kwargs["revision"] = self.config.revision
            self._model = SentenceTransformer(self.config.model_name, **kwargs)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("待编码文本必须是非空字符串。")
        vectors = self._load_model().encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.config.show_progress_bar,
        )
        raw_vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
            raise RuntimeError("BGE-M3 返回的向量数量与输入不一致。")
        normalized: list[list[float]] = []
        dimension: int | None = None
        for vector in raw_vectors:
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("BGE-M3 返回了无效向量。")
            converted = [float(value) for value in vector]
            if dimension is None:
                dimension = len(converted)
            elif len(converted) != dimension:
                raise RuntimeError("BGE-M3 返回的向量维度不一致。")
            normalized.append(converted)
        return normalized

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空。")
        return self._encode([query.strip()])[0]
