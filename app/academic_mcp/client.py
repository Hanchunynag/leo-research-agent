"""本地 Agent 调用 Academic Discovery MCP 的 stdio 客户端。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class AcademicMCPClientError(RuntimeError):
    """MCP 协议或工具调用没有返回可用的结构化结果。"""


class AcademicMCPClient:
    """通过 MCP 协议访问外部学术连接器，不直接导入 provider。"""

    def __init__(
        self,
        project_root: Path,
        python_executable: Path | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.python_executable = (
            python_executable.expanduser().absolute()
            if python_executable
            else Path(sys.executable).absolute()
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        parameters = StdioServerParameters(
            command=str(self.python_executable),
            args=[
                str(self.project_root / "main.py"),
                "academic-mcp",
            ],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        if result.isError:
            messages = [
                content.text for content in result.content if hasattr(content, "text")
            ]
            raise AcademicMCPClientError(
                "\n".join(messages) or f"MCP 工具 {name} 调用失败。"
            )
        payload = result.structuredContent
        if not isinstance(payload, dict):
            raise AcademicMCPClientError(f"MCP 工具 {name} 没有返回 JSON 对象。")
        return payload

    async def resolve_paper(
        self,
        title: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "resolve_paper",
            {"title": title, "limit": limit},
        )
