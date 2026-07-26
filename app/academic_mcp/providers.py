"""Crossref、OpenAlex 与 arXiv 的只读学术搜索适配器。"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any, Protocol

import httpx

from app.academic_mcp.models import FullTextLocation, PaperRecord


DEFAULT_TIMEOUT_SECONDS = 25.0
ARXIV_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_EXTENSION_NAMESPACE = "http://arxiv.org/schemas/atom"


class AcademicProvider(Protocol):
    """学术搜索 provider 需要实现的最小接口。"""

    name: str

    async def search(self, query: str, limit: int) -> list[PaperRecord]: ...


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    without_markup = re.sub(r"<[^>]+>", " ", html.unescape(value))
    cleaned = re.sub(r"\s+", " ", without_markup).strip()
    return cleaned or None


def normalize_doi(value: Any) -> str | None:
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return lowered.strip() or None


def first_string(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            cleaned = clean_text(item)
            if cleaned:
                return cleaned
        return None
    return clean_text(value)


def crossref_year(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued", "created"):
        date_value = item.get(field)
        if not isinstance(date_value, dict):
            continue
        date_parts = date_value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None


def parse_crossref_items(payload: dict[str, Any], provider: str) -> list[PaperRecord]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    items = message.get("items")
    if not isinstance(items, list):
        return []

    records: list[PaperRecord] = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        title = first_string(raw_item.get("title"))
        if title is None:
            continue
        authors: list[str] = []
        raw_authors = raw_item.get("author")
        if isinstance(raw_authors, list):
            for raw_author in raw_authors:
                if not isinstance(raw_author, dict):
                    continue
                name = " ".join(
                    part
                    for part in (
                        clean_text(raw_author.get("given")),
                        clean_text(raw_author.get("family")),
                    )
                    if part
                )
                if name:
                    authors.append(name)
        doi = normalize_doi(raw_item.get("DOI"))
        external_ids = {"doi": doi} if doi else {}
        records.append(
            PaperRecord(
                title=title,
                authors=authors,
                abstract=clean_text(raw_item.get("abstract")),
                publication_year=crossref_year(raw_item),
                doi=doi,
                venue=first_string(raw_item.get("container-title")),
                landing_page_url=clean_text(raw_item.get("URL")),
                external_ids=external_ids,
                sources=[provider],
            )
        )
    return records


def reconstruct_openalex_abstract(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, word))
    if not positioned:
        return None
    return " ".join(word for _, word in sorted(positioned))


def openalex_location(
    raw: Any,
    source_name: str,
    assume_open_access: bool = False,
) -> FullTextLocation | None:
    if not isinstance(raw, dict):
        return None
    if not assume_open_access and raw.get("is_oa") is not True:
        return None
    pdf_url = clean_text(raw.get("pdf_url"))
    if pdf_url is None:
        return None
    source = raw.get("source")
    host_type = None
    if isinstance(source, dict):
        host_type = clean_text(source.get("type"))
    return FullTextLocation(
        url=pdf_url,
        source=source_name,
        landing_page_url=clean_text(raw.get("landing_page_url")),
        version=clean_text(raw.get("version")),
        license=clean_text(raw.get("license")),
        host_type=host_type,
    )


def parse_openalex_work(raw_work: dict[str, Any], provider: str) -> PaperRecord | None:
    title = clean_text(raw_work.get("title") or raw_work.get("display_name"))
    if title is None:
        return None
    authors: list[str] = []
    authorships = raw_work.get("authorships")
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if isinstance(author, dict):
                name = clean_text(author.get("display_name"))
                if name:
                    authors.append(name)
    ids = raw_work.get("ids")
    external_ids: dict[str, str] = {}
    if isinstance(ids, dict):
        for key, value in ids.items():
            cleaned = clean_text(value)
            if isinstance(key, str) and cleaned:
                external_ids[key] = cleaned
    openalex_id = clean_text(raw_work.get("id"))
    if openalex_id:
        external_ids["openalex"] = openalex_id.rsplit("/", 1)[-1]
    doi = normalize_doi(raw_work.get("doi"))
    if doi:
        external_ids["doi"] = doi
    primary_location = raw_work.get("primary_location")
    venue = None
    landing_page_url = None
    if isinstance(primary_location, dict):
        landing_page_url = clean_text(primary_location.get("landing_page_url"))
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue = clean_text(source.get("display_name"))
    open_access = raw_work.get("open_access")
    is_oa = None
    if isinstance(open_access, dict) and isinstance(open_access.get("is_oa"), bool):
        is_oa = open_access["is_oa"]
    year = raw_work.get("publication_year")
    cited_by_count = raw_work.get("cited_by_count")
    return PaperRecord(
        title=title,
        authors=authors,
        abstract=reconstruct_openalex_abstract(raw_work.get("abstract_inverted_index")),
        publication_year=year if isinstance(year, int) else None,
        doi=doi,
        venue=venue,
        cited_by_count=cited_by_count if isinstance(cited_by_count, int) else None,
        open_access=is_oa,
        landing_page_url=landing_page_url,
        external_ids=external_ids,
        sources=[provider],
    )


def parse_arxiv_feed(xml_text: str, provider: str) -> list[PaperRecord]:
    root = ET.fromstring(xml_text)
    namespace = {"atom": ARXIV_NAMESPACE, "arxiv": ARXIV_EXTENSION_NAMESPACE}
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", namespace):
        title = clean_text(entry.findtext("atom:title", namespaces=namespace))
        if title is None:
            continue
        entry_id = clean_text(entry.findtext("atom:id", namespaces=namespace))
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None
        authors = [
            name
            for author in entry.findall("atom:author", namespace)
            if (name := clean_text(author.findtext("atom:name", namespaces=namespace)))
        ]
        published = clean_text(entry.findtext("atom:published", namespaces=namespace))
        year = int(published[:4]) if published and published[:4].isdigit() else None
        doi = normalize_doi(entry.findtext("arxiv:doi", namespaces=namespace))
        external_ids: dict[str, str] = {}
        if arxiv_id:
            external_ids["arxiv"] = arxiv_id
        if doi:
            external_ids["doi"] = doi
        records.append(
            PaperRecord(
                title=title,
                authors=authors,
                abstract=clean_text(
                    entry.findtext("atom:summary", namespaces=namespace)
                ),
                publication_year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                venue=clean_text(
                    entry.findtext("arxiv:journal_ref", namespaces=namespace)
                ),
                open_access=True,
                landing_page_url=entry_id,
                external_ids=external_ids,
                sources=[provider],
            )
        )
    return records


class CrossrefProvider:
    name = "crossref"

    def __init__(self, client: httpx.AsyncClient, contact_email: str | None) -> None:
        self.client = client
        self.contact_email = contact_email

    async def search(self, query: str, limit: int) -> list[PaperRecord]:
        params: dict[str, str | int] = {
            "query.bibliographic": query,
            "rows": limit,
            "select": (
                "DOI,title,author,abstract,published-print,published-online,"
                "issued,created,container-title,URL"
            ),
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        response = await self.client.get(
            "https://api.crossref.org/works",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        return parse_crossref_items(payload, self.name)


class OpenAlexProvider:
    name = "openalex"

    def __init__(self, client: httpx.AsyncClient, contact_email: str | None) -> None:
        self.client = client
        self.contact_email = contact_email

    def common_params(self) -> dict[str, str]:
        return {"mailto": self.contact_email} if self.contact_email else {}

    async def search(self, query: str, limit: int) -> list[PaperRecord]:
        params: dict[str, str | int] = {
            "search": query,
            "per-page": limit,
            **self.common_params(),
        }
        response = await self.client.get(
            "https://api.openalex.org/works",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            return []
        records: list[PaperRecord] = []
        for raw_work in raw_results:
            if isinstance(raw_work, dict):
                record = parse_openalex_work(raw_work, self.name)
                if record:
                    records.append(record)
        return records

    async def find_work(
        self,
        doi: str | None,
        openalex_id: str | None,
    ) -> dict[str, Any] | None:
        if openalex_id:
            identifier = openalex_id.rsplit("/", 1)[-1]
            response = await self.client.get(
                f"https://api.openalex.org/works/{identifier}",
                params=self.common_params(),
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        if doi:
            params: dict[str, str | int] = {
                "filter": f"doi:{normalize_doi(doi)}",
                "per-page": 1,
                **self.common_params(),
            }
            response = await self.client.get(
                "https://api.openalex.org/works",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(results, list) and results and isinstance(results[0], dict):
                return results[0]
        return None

    async def fulltext_locations(
        self,
        doi: str | None,
        openalex_id: str | None,
    ) -> list[FullTextLocation]:
        work = await self.find_work(doi=doi, openalex_id=openalex_id)
        if work is None:
            return []
        locations: list[FullTextLocation] = []
        best_oa = openalex_location(
            work.get("best_oa_location"),
            self.name,
            assume_open_access=True,
        )
        if best_oa:
            locations.append(best_oa)
        primary = openalex_location(work.get("primary_location"), self.name)
        if primary:
            locations.append(primary)
        raw_locations = work.get("locations")
        if isinstance(raw_locations, list):
            for raw_location in raw_locations:
                location = openalex_location(raw_location, self.name)
                if location:
                    locations.append(location)
        return locations


class ArxivProvider:
    name = "arxiv"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(self, query: str, limit: int) -> list[PaperRecord]:
        response = await self.client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            },
        )
        response.raise_for_status()
        return parse_arxiv_feed(response.text, self.name)


async def unpaywall_locations(
    client: httpx.AsyncClient,
    doi: str,
    contact_email: str,
) -> list[FullTextLocation]:
    response = await client.get(
        f"https://api.unpaywall.org/v2/{normalize_doi(doi)}",
        params={"email": contact_email},
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    candidates: list[Any] = [payload.get("best_oa_location")]
    raw_locations = payload.get("oa_locations")
    if isinstance(raw_locations, list):
        candidates.extend(raw_locations)
    locations: list[FullTextLocation] = []
    for raw_location in candidates:
        if not isinstance(raw_location, dict):
            continue
        pdf_url = clean_text(raw_location.get("url_for_pdf"))
        if not pdf_url:
            continue
        locations.append(
            FullTextLocation(
                url=pdf_url,
                source="unpaywall",
                landing_page_url=clean_text(raw_location.get("url_for_landing_page")),
                version=clean_text(raw_location.get("version")),
                license=clean_text(raw_location.get("license")),
                host_type=clean_text(raw_location.get("host_type")),
            )
        )
    return locations
