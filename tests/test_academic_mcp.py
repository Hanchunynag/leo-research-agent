from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.academic_mcp.client import AcademicMCPClient
from app.academic_mcp.models import PaperRecord
from app.academic_mcp.providers import (
    parse_arxiv_feed,
    parse_crossref_items,
    parse_openalex_work,
)
from app.academic_mcp.server import create_mcp
from app.academic_mcp.service import AcademicDiscoveryService


def test_provider_payloads_are_normalized() -> None:
    crossref = parse_crossref_items(
        {
            "message": {
                "items": [
                    {
                        "title": ["LEO Ephemeris Study"],
                        "author": [{"given": "Ada", "family": "Lovelace"}],
                        "abstract": "<jats:p>Useful abstract.</jats:p>",
                        "DOI": "10.1000/LEO",
                        "published-online": {"date-parts": [[2025, 1, 2]]},
                        "container-title": ["Navigation Journal"],
                        "URL": "https://doi.org/10.1000/leo",
                    }
                ]
            }
        },
        "crossref",
    )[0]
    assert crossref.authors == ["Ada Lovelace"]
    assert crossref.abstract == "Useful abstract."
    assert crossref.doi == "10.1000/leo"
    assert crossref.publication_year == 2025

    openalex = parse_openalex_work(
        {
            "id": "https://openalex.org/W123",
            "title": "LEO Ephemeris Study",
            "publication_year": 2025,
            "doi": "https://doi.org/10.1000/leo",
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "abstract_inverted_index": {
                "Study": [1],
                "A": [0],
            },
            "open_access": {"is_oa": True},
            "cited_by_count": 12,
            "primary_location": {
                "landing_page_url": "https://example.org/work",
                "source": {"display_name": "Navigation Journal"},
            },
        },
        "openalex",
    )
    assert openalex is not None
    assert openalex.abstract == "A Study"
    assert openalex.external_ids["openalex"] == "W123"
    assert openalex.open_access is True

    arxiv = parse_arxiv_feed(
        """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>https://arxiv.org/abs/2601.12345v2</id>
            <title>LEO Ephemeris Study</title>
            <summary>Open preprint.</summary>
            <published>2026-01-20T00:00:00Z</published>
            <author><name>Ada Lovelace</name></author>
            <arxiv:doi>10.1000/leo</arxiv:doi>
          </entry>
        </feed>""",
        "arxiv",
    )[0]
    assert arxiv.arxiv_id == "2601.12345v2"
    assert arxiv.publication_year == 2026
    assert arxiv.open_access is True


class FakeProvider:
    def __init__(
        self,
        name: str,
        records: list[PaperRecord] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.records = records or []
        self.error = error

    async def search(self, query: str, limit: int) -> list[PaperRecord]:
        assert query
        assert limit > 0
        if self.error:
            raise self.error
        return self.records


def test_search_merges_sources_and_keeps_provider_failure(tmp_path: Path) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    )
    service = AcademicDiscoveryService(
        providers=[
            FakeProvider(
                "crossref",
                [
                    PaperRecord(
                        title="LEO Ephemeris Study",
                        authors=["Ada Lovelace"],
                        doi="10.1000/leo",
                        sources=["crossref"],
                    )
                ],
            ),
            FakeProvider(
                "openalex",
                [
                    PaperRecord(
                        title="LEO Ephemeris Study",
                        abstract="A longer abstract from OpenAlex.",
                        doi="https://doi.org/10.1000/LEO",
                        cited_by_count=12,
                        sources=["openalex"],
                    )
                ],
            ),
            FakeProvider("arxiv", error=RuntimeError("rate limited")),
        ],
        client=client,
        openalex=None,
        contact_email=None,
        inbox=tmp_path / "inbox",
    )

    async def run() -> None:
        result = await service.search_filtered(
            "LEO ephemeris",
            limit=10,
            year_from=2020,
        )
        assert len(result.papers) == 1
        assert result.papers[0].sources == ["crossref", "openalex"]
        assert result.papers[0].abstract == "A longer abstract from OpenAlex."
        assert result.failures[0].provider == "arxiv"

        resolved = await service.resolve("LEO Ephemeris Study", limit=1)
        assert resolved.papers[0].match_score == 1.0
        await service.close()

    asyncio.run(run())


def test_find_then_download_open_pdf_to_inbox(tmp_path: Path) -> None:
    pdf_bytes = b"%PDF-1.7\nfixture"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://arxiv.org/pdf/2601.12345.pdf"
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
                "content-disposition": 'attachment; filename="paper.pdf"',
            },
            content=pdf_bytes,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = AcademicDiscoveryService(
        providers=[],
        client=client,
        openalex=None,
        contact_email=None,
        inbox=tmp_path / "data" / "inbox",
    )

    async def allow_test_url(_: str) -> None:
        return None

    service.downloader.url_validator = allow_test_url

    async def run() -> None:
        found = await service.find_fulltext(arxiv_id="2601.12345v2")
        assert len(found.locations) == 1
        token = found.locations[0].download_token
        assert token

        downloaded = await service.download_open_pdf(token, "论文 最终版.pdf")
        path = Path(str(downloaded["path"]))
        assert path.name == "论文_最终版.pdf"
        assert path.read_bytes() == pdf_bytes
        assert downloaded["reused"] is False

        reused = await service.download_open_pdf(token, "论文 最终版.pdf")
        assert reused["reused"] is True
        await service.close()

    asyncio.run(run())


def test_mcp_exposes_only_external_discovery_tools(tmp_path: Path) -> None:
    server = create_mcp(project_root=tmp_path)
    assert set(server._tool_manager._tools) == {
        "search_papers",
        "resolve_paper",
        "find_fulltext",
        "download_open_pdf",
    }


def test_client_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    executable = tmp_path / ".venv" / "bin" / "python"
    client = AcademicMCPClient(
        project_root=tmp_path,
        python_executable=executable,
    )

    assert client.python_executable == executable.absolute()
