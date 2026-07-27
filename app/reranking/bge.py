"""BGE Reranker v2 M3 Cross-Encoder Provider。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class BGERerankerConfig:
    model_name: str = DEFAULT_BGE_RERANKER_MODEL
    revision: str | None = None
    device: str | None = None
    cache_folder: Path | None = None
    batch_size: int = 4
    max_length: int = 1024
    local_files_only: bool = False
    show_progress_bar: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name 不能为空。")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于 0。")
        if self.max_length < 32:
            raise ValueError("max_length 不能小于 32。")


class BGERerankerProvider:
    """只输出 Cross-Encoder 原始相关性 logits。"""

    provider_id = "sentence-transformers:bge-reranker-v2-m3"
    score_transform = "identity_logits"

    def __init__(
        self,
        config: BGERerankerConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or BGERerankerConfig()
        self._model = model

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def revision(self) -> str | None:
        return self.config.revision

    @property
    def max_length(self) -> int:
        return self.config.max_length

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            self._model = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
                cache_folder=(
                    str(self.config.cache_folder)
                    if self.config.cache_folder is not None
                    else None
                ),
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                max_length=self.config.max_length,
                activation_fn=Identity(),
            )
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空。")
        if not documents:
            return []
        if any(not isinstance(value, str) or not value.strip() for value in documents):
            raise ValueError("候选文档必须是非空字符串。")
        pairs = [(cleaned_query, document) for document in documents]
        values = self._load_model().predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            convert_to_numpy=True,
        )
        raw_scores = values.tolist() if hasattr(values, "tolist") else values
        if not isinstance(raw_scores, list) or len(raw_scores) != len(documents):
            raise RuntimeError("BGE Reranker 返回的分数数量与候选文档不一致。")
        scores: list[float] = []
        for value in raw_scores:
            if isinstance(value, list):
                if len(value) != 1:
                    raise RuntimeError("BGE Reranker 返回了非标量分数。")
                value = value[0]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeError("BGE Reranker 返回了无效分数。")
            scores.append(float(value))
        return scores
