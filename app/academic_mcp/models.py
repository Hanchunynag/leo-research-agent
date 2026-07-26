"""Academic Discovery MCP 的内部数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PaperRecord:
    """一个外部学术数据源返回的论文记录。"""

    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    cited_by_count: int | None = None
    open_access: bool | None = None
    landing_page_url: str | None = None
    external_ids: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    match_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullTextLocation:
    """由可信学术数据源发现的一个全文位置。"""

    url: str
    source: str
    landing_page_url: str | None = None
    version: str | None = None
    license: str | None = None
    host_type: str | None = None
    download_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderFailure:
    """单个外部 provider 的非阻塞失败。"""

    provider: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryResult:
    """聚合搜索或解析的结果。"""

    papers: list[PaperRecord]
    failures: list[ProviderFailure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_count": len(self.papers),
            "results": [paper.to_dict() for paper in self.papers],
            "provider_failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class FullTextResult:
    """全文位置聚合结果。"""

    locations: list[FullTextLocation]
    failures: list[ProviderFailure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_count": len(self.locations),
            "locations": [location.to_dict() for location in self.locations],
            "provider_failures": [failure.to_dict() for failure in self.failures],
        }
