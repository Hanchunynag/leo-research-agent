"""外部学术搜索、解析、全文发现与下载的应用服务。"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from app.academic_mcp.downloader import (
    DEFAULT_MAX_PDF_BYTES,
    DownloadRegistry,
    OpenPDFDownloader,
)
from app.academic_mcp.models import (
    DiscoveryResult,
    FullTextLocation,
    FullTextResult,
    PaperRecord,
    ProviderFailure,
)
from app.academic_mcp.providers import (
    AcademicProvider,
    ArxivProvider,
    CrossrefProvider,
    OpenAlexProvider,
    normalize_doi,
    unpaywall_locations,
)


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def title_similarity(left: str, right: str) -> float:
    return round(
        SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio(),
        4,
    )


def paper_key(record: PaperRecord) -> str:
    if record.doi:
        return f"doi:{normalize_doi(record.doi)}"
    if record.arxiv_id:
        return f"arxiv:{record.arxiv_id.casefold()}"
    return f"title:{normalize_title(record.title)}"


def prefer(left: Any, right: Any) -> Any:
    if left not in (None, "", [], {}):
        return left
    return right


def merge_papers(left: PaperRecord, right: PaperRecord) -> PaperRecord:
    authors = left.authors if len(left.authors) >= len(right.authors) else right.authors
    abstract = left.abstract
    if right.abstract and (abstract is None or len(right.abstract) > len(abstract)):
        abstract = right.abstract
    cited_values = [
        value
        for value in (left.cited_by_count, right.cited_by_count)
        if value is not None
    ]
    return PaperRecord(
        title=left.title if len(left.title) >= len(right.title) else right.title,
        authors=authors,
        abstract=abstract,
        publication_year=prefer(left.publication_year, right.publication_year),
        doi=prefer(left.doi, right.doi),
        arxiv_id=prefer(left.arxiv_id, right.arxiv_id),
        venue=prefer(left.venue, right.venue),
        cited_by_count=max(cited_values) if cited_values else None,
        open_access=(
            True
            if left.open_access is True or right.open_access is True
            else prefer(left.open_access, right.open_access)
        ),
        landing_page_url=prefer(left.landing_page_url, right.landing_page_url),
        external_ids={**right.external_ids, **left.external_ids},
        sources=sorted(set(left.sources + right.sources)),
        match_score=prefer(left.match_score, right.match_score),
    )


def deduplicate_papers(records: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    order: list[str] = []
    for record in records:
        key = paper_key(record)
        if key not in merged:
            merged[key] = record
            order.append(key)
        else:
            merged[key] = merge_papers(merged[key], record)
    return [merged[key] for key in order]


def provider_failure(provider: str, error: BaseException) -> ProviderFailure:
    return ProviderFailure(
        provider=provider,
        error_type=type(error).__name__,
        message=str(error),
    )


class AcademicDiscoveryService:
    """外部连接层；不读取或修改 canonical、Chunk 和索引。"""

    def __init__(
        self,
        providers: list[AcademicProvider],
        client: httpx.AsyncClient,
        openalex: OpenAlexProvider | None,
        contact_email: str | None,
        inbox: Path,
        max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
    ) -> None:
        self.providers = providers
        self.client = client
        self.openalex = openalex
        self.contact_email = contact_email
        self.registry = DownloadRegistry()
        self.downloader = OpenPDFDownloader(
            client=client,
            registry=self.registry,
            inbox=inbox,
            max_pdf_bytes=max_pdf_bytes,
        )

    @classmethod
    def create(
        cls,
        project_root: Path,
        contact_email: str | None = None,
        max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
    ) -> AcademicDiscoveryService:
        user_agent = "leo-research-agent/0.2 academic-discovery-mcp"
        if contact_email:
            user_agent += f" (mailto:{contact_email})"
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(25.0, connect=10.0),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json, application/atom+xml, application/pdf",
            },
            follow_redirects=False,
        )
        openalex = OpenAlexProvider(client, contact_email)
        providers: list[AcademicProvider] = [
            CrossrefProvider(client, contact_email),
            openalex,
            ArxivProvider(client),
        ]
        return cls(
            providers=providers,
            client=client,
            openalex=openalex,
            contact_email=contact_email,
            inbox=project_root / "data" / "inbox",
            max_pdf_bytes=max_pdf_bytes,
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def require_query(query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query 不能为空。")
        return cleaned

    @staticmethod
    def require_limit(limit: int, maximum: int = 100) -> int:
        if isinstance(limit, bool) or limit < 1 or limit > maximum:
            raise ValueError(f"limit 必须在 1 到 {maximum} 之间。")
        return limit

    async def search(self, query: str, limit: int = 20) -> DiscoveryResult:
        return await self.search_filtered(query=query, limit=limit)

    async def search_filtered(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access_only: bool = False,
    ) -> DiscoveryResult:
        cleaned_query = self.require_query(query)
        validated_limit = self.require_limit(limit)
        for year, field in ((year_from, "year_from"), (year_to, "year_to")):
            if year is not None and (
                isinstance(year, bool) or year < 1000 or year > 3000
            ):
                raise ValueError(f"{field} 必须是 1000 到 3000 之间的年份。")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("year_from 不能大于 year_to。")

        provider_limit = min(max(validated_limit * 3, 20), 100)
        outcomes = await asyncio.gather(
            *(
                provider.search(cleaned_query, provider_limit)
                for provider in self.providers
            ),
            return_exceptions=True,
        )
        provider_records: list[list[PaperRecord]] = []
        failures: list[ProviderFailure] = []
        for provider, outcome in zip(self.providers, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(provider_failure(provider.name, outcome))
            else:
                provider_records.append(outcome)

        # 轮询混合 provider 顺序，避免第一个数据源独占结果上限。
        records: list[PaperRecord] = []
        for index in range(max((len(items) for items in provider_records), default=0)):
            for items in provider_records:
                if index < len(items):
                    records.append(items[index])
        papers = [
            paper
            for paper in deduplicate_papers(records)
            if (
                (
                    year_from is None
                    or paper.publication_year is None
                    or paper.publication_year >= year_from
                )
                and (
                    year_to is None
                    or paper.publication_year is None
                    or paper.publication_year <= year_to
                )
                and (not open_access_only or paper.open_access is True)
            )
        ][:validated_limit]
        return DiscoveryResult(papers=papers, failures=failures)

    async def resolve(self, title: str, limit: int = 5) -> DiscoveryResult:
        cleaned_title = self.require_query(title)
        validated_limit = self.require_limit(limit, maximum=20)
        search_result = await self.search(
            cleaned_title,
            limit=max(validated_limit * 3, 10),
        )
        scored = [
            replace(
                paper,
                match_score=title_similarity(cleaned_title, paper.title),
            )
            for paper in search_result.papers
        ]
        scored.sort(
            key=lambda paper: (
                paper.match_score or 0.0,
                len(paper.sources),
                paper.cited_by_count or 0,
            ),
            reverse=True,
        )
        return DiscoveryResult(
            papers=scored[:validated_limit],
            failures=search_result.failures,
        )

    async def find_fulltext(
        self,
        doi: str | None = None,
        openalex_id: str | None = None,
        arxiv_id: str | None = None,
    ) -> FullTextResult:
        normalized_doi = normalize_doi(doi)
        cleaned_openalex = openalex_id.strip() if openalex_id else None
        cleaned_arxiv = arxiv_id.strip() if arxiv_id else None
        if not any((normalized_doi, cleaned_openalex, cleaned_arxiv)):
            raise ValueError("doi、openalex_id、arxiv_id 至少提供一个。")

        locations: list[FullTextLocation] = []
        failures: list[ProviderFailure] = []
        if cleaned_arxiv:
            versionless = re.sub(r"v\d+$", "", cleaned_arxiv)
            locations.append(
                FullTextLocation(
                    url=f"https://arxiv.org/pdf/{versionless}.pdf",
                    landing_page_url=f"https://arxiv.org/abs/{versionless}",
                    source="arxiv",
                    version="submittedVersion",
                    host_type="repository",
                )
            )

        if self.openalex is not None and (normalized_doi or cleaned_openalex):
            try:
                locations.extend(
                    await self.openalex.fulltext_locations(
                        doi=normalized_doi,
                        openalex_id=cleaned_openalex,
                    )
                )
            except Exception as error:
                failures.append(provider_failure("openalex", error))

        if normalized_doi and self.contact_email:
            try:
                locations.extend(
                    await unpaywall_locations(
                        self.client,
                        normalized_doi,
                        self.contact_email,
                    )
                )
            except Exception as error:
                failures.append(provider_failure("unpaywall", error))
        elif normalized_doi and not self.contact_email:
            failures.append(
                ProviderFailure(
                    provider="unpaywall",
                    error_type="ConfigurationError",
                    message=("未配置 LEO_ACADEMIC_CONTACT_EMAIL，已跳过 Unpaywall。"),
                )
            )

        unique: dict[str, FullTextLocation] = {}
        for location in locations:
            if location.url.startswith("https://"):
                unique.setdefault(location.url, location)
        registered = [self.registry.register(location) for location in unique.values()]
        return FullTextResult(locations=registered, failures=failures)

    async def download_open_pdf(
        self,
        download_token: str,
        filename: str | None = None,
    ) -> dict[str, str | int | bool]:
        return (
            await self.downloader.download(
                token=download_token,
                filename=filename,
            )
        ).to_dict()
