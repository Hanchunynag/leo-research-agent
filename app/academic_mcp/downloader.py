"""受控的开放全文下载器。"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import secrets
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app.academic_mcp.models import FullTextLocation


MAX_REDIRECTS = 5
DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024
URLValidator = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class DownloadResult:
    path: str
    sha256: str
    byte_count: int
    source_url: str
    reused: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "source_url": self.source_url,
            "reused": self.reused,
        }


class DownloadRegistry:
    """只允许下载由全文发现工具登记过的 URL。"""

    def __init__(self) -> None:
        self._locations: dict[str, FullTextLocation] = {}

    def register(self, location: FullTextLocation) -> FullTextLocation:
        token = secrets.token_urlsafe(24)
        registered = FullTextLocation(
            url=location.url,
            source=location.source,
            landing_page_url=location.landing_page_url,
            version=location.version,
            license=location.license,
            host_type=location.host_type,
            download_token=token,
        )
        self._locations[token] = registered
        return registered

    def resolve(self, token: str) -> FullTextLocation:
        try:
            return self._locations[token]
        except KeyError as error:
            raise ValueError(
                "下载 token 无效或已过期；请先调用 find_fulltext。"
            ) from error


def sanitize_filename(value: str | None) -> str:
    name = Path(value or "paper.pdf").name
    stem = Path(name).stem.strip()
    safe_stem = re.sub(r"[^\w\-().]+", "_", stem, flags=re.UNICODE)
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
    if not safe_stem:
        safe_stem = "paper"
    return f"{safe_stem[:150]}.pdf"


def _public_addresses(
    hostname: str, port: int
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(info[4][0])
        addresses.append(address)
    return addresses


async def validate_public_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("只允许从公开 HTTPS 地址下载 PDF。")
    if parsed.username or parsed.password:
        raise ValueError("下载 URL 不得包含用户名或密码。")
    port = parsed.port or 443
    addresses = await asyncio.to_thread(_public_addresses, parsed.hostname, port)
    if not addresses:
        raise ValueError("下载地址无法解析。")
    for address in addresses:
        if not address.is_global:
            raise ValueError("拒绝访问内网、回环或保留地址。")


def filename_from_headers(response: httpx.Response) -> str | None:
    content_disposition = response.headers.get("content-disposition", "")
    match = re.search(
        r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', content_disposition
    )
    if not match:
        return None
    return match.group(1).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OpenPDFDownloader:
    """下载已登记的公开 PDF，并把写入范围限制在 inbox。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        registry: DownloadRegistry,
        inbox: Path,
        max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
        url_validator: URLValidator = validate_public_https_url,
    ) -> None:
        self.client = client
        self.registry = registry
        self.inbox = inbox.expanduser().resolve()
        self.max_pdf_bytes = max_pdf_bytes
        self.url_validator = url_validator

    async def _open_response(self, url: str) -> httpx.Response:
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            await self.url_validator(current_url)
            request = self.client.build_request("GET", current_url)
            response = await self.client.send(request, stream=True)
            if response.is_redirect:
                redirect = response.headers.get("location")
                await response.aclose()
                if not redirect:
                    raise ValueError("全文下载返回了没有 Location 的重定向。")
                current_url = urljoin(current_url, redirect)
                continue
            response.raise_for_status()
            return response
        raise ValueError(f"全文下载重定向超过 {MAX_REDIRECTS} 次。")

    async def download(
        self,
        token: str,
        filename: str | None = None,
    ) -> DownloadResult:
        location = self.registry.resolve(token)
        response = await self._open_response(location.url)
        declared_size = response.headers.get("content-length")
        if declared_size and declared_size.isdigit():
            if int(declared_size) > self.max_pdf_bytes:
                await response.aclose()
                raise ValueError("PDF 超过允许的最大下载体积。")

        selected_name = sanitize_filename(
            filename or filename_from_headers(response) or urlparse(location.url).path
        )
        self.inbox.mkdir(parents=True, exist_ok=True)
        temporary = self.inbox / f".{secrets.token_hex(12)}.part"
        digest = hashlib.sha256()
        byte_count = 0
        first_bytes = b""
        try:
            with temporary.open("wb") as stream:
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    if byte_count > self.max_pdf_bytes:
                        raise ValueError("PDF 超过允许的最大下载体积。")
                    if len(first_bytes) < 5:
                        first_bytes += chunk[: 5 - len(first_bytes)]
                    digest.update(chunk)
                    stream.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            await response.aclose()

        if first_bytes != b"%PDF-":
            temporary.unlink(missing_ok=True)
            raise ValueError("远端内容不是有效 PDF。")

        sha256 = digest.hexdigest()
        destination = self.inbox / selected_name
        reused = False
        if destination.exists():
            if file_sha256(destination) == sha256:
                temporary.unlink(missing_ok=True)
                reused = True
            else:
                destination = destination.with_name(
                    f"{destination.stem}-{sha256[:12]}.pdf"
                )
        if not reused:
            temporary.replace(destination)

        return DownloadResult(
            path=destination.as_posix(),
            sha256=sha256,
            byte_count=byte_count,
            source_url=location.url,
            reused=reused,
        )
