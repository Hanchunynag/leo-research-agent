"""Academic Discovery MCP 的 stdio/HTTP 服务入口。"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.server import FastMCP as FastMCPServer

from app.academic_mcp.downloader import DEFAULT_MAX_PDF_BYTES
from app.academic_mcp.service import AcademicDiscoveryService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Transport = Literal["stdio", "sse", "streamable-http"]


@dataclass(frozen=True)
class AcademicMCPLifespan:
    service: AcademicDiscoveryService


def configured_max_pdf_bytes() -> int:
    raw_value = os.environ.get("LEO_ACADEMIC_MAX_PDF_BYTES")
    if raw_value is None:
        return DEFAULT_MAX_PDF_BYTES
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("LEO_ACADEMIC_MAX_PDF_BYTES 必须是正整数。") from error
    if value < 1:
        raise ValueError("LEO_ACADEMIC_MAX_PDF_BYTES 必须是正整数。")
    return value


def service_from_context(context: Context[Any, Any, Any]) -> AcademicDiscoveryService:
    lifespan = cast(
        AcademicMCPLifespan,
        context.request_context.lifespan_context,
    )
    return lifespan.service


def create_mcp(
    project_root: Path = PROJECT_ROOT,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP[AcademicMCPLifespan]:
    root = project_root.expanduser().resolve()

    @asynccontextmanager
    async def lifespan(
        _: FastMCPServer[AcademicMCPLifespan],
    ) -> AsyncIterator[AcademicMCPLifespan]:
        service = AcademicDiscoveryService.create(
            project_root=root,
            contact_email=os.environ.get("LEO_ACADEMIC_CONTACT_EMAIL"),
            max_pdf_bytes=configured_max_pdf_bytes(),
        )
        try:
            yield AcademicMCPLifespan(service=service)
        finally:
            await service.close()

    server = FastMCP(
        name="leo-academic-discovery",
        instructions=(
            "只连接外部学术数据源：搜索论文、按标题解析元数据、查找开放全文，"
            "以及把用户选定的开放 PDF 下载到 data/inbox。"
            "本服务不解析本地 PDF，也不修改 canonical、Chunk 或索引。"
        ),
        host=host,
        port=port,
        lifespan=lifespan,
    )

    @server.tool(
        name="search_papers",
        description=(
            "同时搜索 Crossref、OpenAlex 和 arXiv，聚合标题、作者、摘要、"
            "年份、DOI、开放获取状态等外部元数据。"
        ),
    )
    async def search_papers(
        query: str,
        limit: int = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access_only: bool = False,
        context: Context[Any, Any, Any] | None = None,
    ) -> dict[str, Any]:
        if context is None:
            raise RuntimeError("MCP context 不可用。")
        result = await service_from_context(context).search_filtered(
            query=query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
        return result.to_dict()

    @server.tool(
        name="resolve_paper",
        description=(
            "按本地提取的论文标题搜索候选记录，并按标题相似度、多来源和引用量排序；"
            "结果仅供本地 Agent 核验，不会修改本地论文库。"
        ),
    )
    async def resolve_paper(
        title: str,
        limit: int = 5,
        context: Context[Any, Any, Any] | None = None,
    ) -> dict[str, Any]:
        if context is None:
            raise RuntimeError("MCP context 不可用。")
        result = await service_from_context(context).resolve(title=title, limit=limit)
        return result.to_dict()

    @server.tool(
        name="find_fulltext",
        description=(
            "根据 DOI、OpenAlex ID 或 arXiv ID 查找合法开放全文位置。"
            "返回的临时 download_token 可交给 download_open_pdf。"
        ),
    )
    async def find_fulltext(
        doi: str | None = None,
        openalex_id: str | None = None,
        arxiv_id: str | None = None,
        context: Context[Any, Any, Any] | None = None,
    ) -> dict[str, Any]:
        if context is None:
            raise RuntimeError("MCP context 不可用。")
        result = await service_from_context(context).find_fulltext(
            doi=doi,
            openalex_id=openalex_id,
            arxiv_id=arxiv_id,
        )
        return result.to_dict()

    @server.tool(
        name="download_open_pdf",
        description=(
            "使用 find_fulltext 返回的 download_token 下载已发现的开放 PDF；"
            "仅允许公开 HTTPS、校验 PDF 文件头和体积，并写入 data/inbox。"
        ),
    )
    async def download_open_pdf(
        download_token: str,
        filename: str | None = None,
        context: Context[Any, Any, Any] | None = None,
    ) -> dict[str, str | int | bool]:
        if context is None:
            raise RuntimeError("MCP context 不可用。")
        return await service_from_context(context).download_open_pdf(
            download_token=download_token,
            filename=filename,
        )

    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动仅负责外部学术发现与开放全文获取的 MCP。",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    server = create_mcp(host=args.host, port=args.port)
    server.run(transport=cast(Transport, args.transport))


if __name__ == "__main__":
    main()
