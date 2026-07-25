"""统一的论文解析流程。

对外只暴露 :func:`parse_paper`。主项目使用 ``.venv`` 运行本模块，
MinerU 则始终通过 ``.venv-mineru/bin/mineru`` 子进程执行。两个 Python
环境不会互相导入依赖。

流程：

1. 校验、哈希和存储原始 PDF；
2. 执行轻量 PDF 预检查；
3. 调用或复用 MinerU 输出；
4. 将 MinerU content_list_v2 转为统一 blocks；
5. 输出单一 ``paper.json``。

MinerU 的原始输出仍保存在 ``data/parsed``，作为可重复处理的底层资产；
下游只需要读取 ``data/canonical/<paper_id>/paper.json``。
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.ingestion.ingest import IngestResult, ingest_paper
from app.normalization.mineru_adapter import (
    build_canonical_document,
    write_json_atomic,
)
from app.parsing.precheck import PDFPrecheckResult, precheck_pdf


MinerUMethod = Literal["auto", "txt", "ocr"]
MinerUBackend = Literal[
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
    "vlm-http-client",
    "hybrid-http-client",
]


@dataclass(frozen=True)
class PaperParseConfig:
    """统一解析流程配置。"""

    project_root: Path
    mineru_executable: Path | None = None
    method: MinerUMethod = "auto"
    backend: MinerUBackend = "pipeline"
    language: str | None = None
    formula_enabled: bool = True
    table_enabled: bool = True
    force_mineru: bool = False

    def normalized(self) -> "PaperParseConfig":
        return PaperParseConfig(
            project_root=self.project_root.expanduser().resolve(),
            mineru_executable=(
                self.mineru_executable.expanduser().resolve()
                if self.mineru_executable
                else None
            ),
            method=self.method,
            backend=self.backend,
            language=self.language,
            formula_enabled=self.formula_enabled,
            table_enabled=self.table_enabled,
            force_mineru=self.force_mineru,
        )


@dataclass(frozen=True)
class MinerUArtifacts:
    """一次 MinerU 解析中供统一流程消费的核心文件。"""

    content_list_v2: Path
    middle_json: Path
    output_directory: Path


@dataclass(frozen=True)
class PaperParseResult:
    """统一论文解析结果。"""

    paper_id: str
    sha256: str
    raw_pdf: Path
    paper_json: Path
    mineru_output_directory: Path
    mineru_reused: bool
    page_count: int
    block_count: int
    formula_count: int
    table_count: int
    figure_count: int
    precheck: dict[str, Any]


class MinerUExecutionError(RuntimeError):
    """MinerU 子进程执行失败。"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bool_cli(value: bool) -> str:
    """转换为 Click 能稳定识别的布尔参数。"""

    return "true" if value else "false"


def project_relative(path: Path, project_root: Path) -> str:
    """尽量保存项目相对路径，避免 canonical 数据绑定到某台机器。"""

    resolved = path.expanduser().resolve()

    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_mineru_executable(
    project_root: Path,
    configured_path: Path | None = None,
) -> Path:
    """只解析 MinerU 专用环境中的可执行文件。

    不回退到 ``PATH``，从而避免意外使用主项目虚拟环境中的同名命令。
    可通过配置参数或 ``LEO_MINERU_EXECUTABLE`` 显式覆盖。
    """

    candidates: list[Path] = []

    if configured_path is not None:
        candidates.append(configured_path)

    environment_path = os.environ.get("LEO_MINERU_EXECUTABLE")

    if environment_path:
        candidates.append(Path(environment_path))

    candidates.extend(
        [
            project_root / ".venv-mineru" / "bin" / "mineru",
            project_root / ".venv-mineru" / "Scripts" / "mineru.exe",
        ]
    )

    for candidate in candidates:
        path = candidate.expanduser().resolve()

        if path.is_file() and os.access(path, os.X_OK):
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "未找到 MinerU 专用环境的可执行文件。\n"
        "请创建 .venv-mineru，或设置 LEO_MINERU_EXECUTABLE。\n"
        f"已检查：\n{searched}"
    )


def build_mineru_command(
    executable: Path,
    pdf_path: Path,
    output_directory: Path,
    config: PaperParseConfig,
) -> list[str]:
    """构造显式、可记录的 MinerU CLI 命令。"""

    command = [
        str(executable),
        "--path",
        str(pdf_path),
        "--output",
        str(output_directory),
        "--method",
        config.method,
        "--backend",
        config.backend,
        "--formula",
        bool_cli(config.formula_enabled),
        "--table",
        bool_cli(config.table_enabled),
    ]

    if config.language:
        command.extend(["--lang", config.language])

    return command


def find_mineru_artifacts(mineru_root: Path) -> MinerUArtifacts | None:
    """查找同一输出目录内最新且完整的 MinerU 核心产物。"""

    candidates = sorted(
        mineru_root.rglob("*_content_list_v2.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if mineru_root.exists() else []

    for content_list in candidates:
        middle_files = sorted(content_list.parent.glob("*_middle.json"))

        if len(middle_files) == 1:
            return MinerUArtifacts(
                content_list_v2=content_list.resolve(),
                middle_json=middle_files[0].resolve(),
                output_directory=content_list.parent.resolve(),
            )

    return None


def run_mineru(
    pdf_path: Path,
    mineru_root: Path,
    config: PaperParseConfig,
) -> tuple[MinerUArtifacts, list[str]]:
    """使用 MinerU 专用 venv 执行解析。"""

    executable = resolve_mineru_executable(
        project_root=config.project_root,
        configured_path=config.mineru_executable,
    )

    mineru_root.mkdir(parents=True, exist_ok=True)

    command = build_mineru_command(
        executable=executable,
        pdf_path=pdf_path,
        output_directory=mineru_root,
        config=config,
    )

    completed = subprocess.run(
        command,
        cwd=config.project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    log_directory = mineru_root / "_pipeline"
    log_directory.mkdir(parents=True, exist_ok=True)
    (log_directory / "mineru.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (log_directory / "mineru.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        error_tail = completed.stderr[-4000:] or completed.stdout[-4000:]
        raise MinerUExecutionError(
            f"MinerU 执行失败，退出码 {completed.returncode}。\n"
            f"命令：{' '.join(command)}\n"
            f"末尾输出：\n{error_tail}"
        )

    artifacts = find_mineru_artifacts(mineru_root)

    if artifacts is None:
        raise MinerUExecutionError(
            "MinerU 返回成功，但没有找到 content_list_v2.json 和 middle.json。"
        )

    return artifacts, command


def collect_assets(
    blocks: list[dict[str, Any]],
    block_type: str,
    id_prefix: str,
) -> list[dict[str, Any]]:
    """为常用资产建立轻量视图，实际内容仍以 blocks 为准。"""

    assets: list[dict[str, Any]] = []

    for block in blocks:
        if block.get("type") != block_type:
            continue

        assets.append(
            {
                "asset_id": f"{id_prefix}_{len(assets) + 1:03d}",
                "block_id": block.get("block_id"),
                "paper_id": block.get("paper_id"),
                "page_number": block.get("page_number"),
                "bbox": block.get("bbox"),
                "text": block.get("text"),
                "latex": block.get("latex") or block.get("latex_raw"),
                "caption": block.get("caption"),
                "table_html": (
                    block.get("table_html")
                    or block.get("table_html_raw")
                ),
                "image_path": block.get("image_path"),
            }
        )

    return assets


def normalize_document_paths(
    document: dict[str, Any],
    project_root: Path,
) -> None:
    """将统一文档中的本地资源路径转换为项目相对路径。"""

    source_files = document.get("source_files")

    if isinstance(source_files, dict):
        for key, value in source_files.items():
            if isinstance(value, str):
                source_files[key] = project_relative(Path(value), project_root)

    blocks = document.get("blocks")

    if not isinstance(blocks, list):
        return

    for block in blocks:
        if not isinstance(block, dict):
            continue

        image_path = block.get("image_path")

        if isinstance(image_path, str) and image_path:
            block["image_path"] = project_relative(
                Path(image_path),
                project_root,
            )


def parse_paper(
    input_path: Path,
    config: PaperParseConfig,
) -> PaperParseResult:
    """通过一个入口完成论文入库、MinerU 解析和统一输出。"""

    config = config.normalized()
    project_root = config.project_root

    raw_root = project_root / "data" / "raw"
    parsed_root = project_root / "data" / "parsed"
    canonical_root = project_root / "data" / "canonical"

    ingest_result: IngestResult = ingest_paper(
        input_path=input_path,
        raw_dir=raw_root,
    )
    precheck_result: PDFPrecheckResult = precheck_pdf(
        ingest_result.source_path
    )

    mineru_root = (
        parsed_root
        / ingest_result.paper_id
        / "mineru"
    )
    paper_directory = canonical_root / ingest_result.paper_id
    paper_json = paper_directory / "paper.json"

    previous_pipeline: dict[str, Any] = {}
    if paper_json.exists():
        try:
            previous_document = json.loads(
                paper_json.read_text(encoding="utf-8")
            )
            pipeline_value = previous_document.get("pipeline")
            if isinstance(pipeline_value, dict):
                previous_pipeline = pipeline_value
        except (json.JSONDecodeError, OSError):
            previous_pipeline = {}

    artifacts = (
        None
        if config.force_mineru
        else find_mineru_artifacts(mineru_root)
    )
    mineru_reused = artifacts is not None
    command: list[str] | None = None

    if artifacts is None:
        artifacts, command = run_mineru(
            pdf_path=ingest_result.source_path,
            mineru_root=mineru_root,
            config=config,
        )

    document, adapter_report = build_canonical_document(
        paper_id=ingest_result.paper_id,
        content_list_v2_path=artifacts.content_list_v2,
        middle_path=artifacts.middle_json,
    )

    normalize_document_paths(document, project_root)

    blocks = document.get("blocks")
    if not isinstance(blocks, list):
        blocks = []

    formulas = collect_assets(
        blocks=blocks,
        block_type="equation",
        id_prefix=f"{ingest_result.paper_id}_formula",
    )
    tables = collect_assets(
        blocks=blocks,
        block_type="table",
        id_prefix=f"{ingest_result.paper_id}_table",
    )
    figures = collect_assets(
        blocks=blocks,
        block_type="figure",
        id_prefix=f"{ingest_result.paper_id}_figure",
    )

    document["source"] = {
        "sha256": ingest_result.sha256,
        "original_filename": ingest_result.original_filename,
        "stored_filename": ingest_result.stored_filename,
        "raw_pdf": project_relative(ingest_result.source_path, project_root),
    }
    precheck_payload = asdict(precheck_result)
    precheck_payload["source_path"] = project_relative(
        ingest_result.source_path,
        project_root,
    )
    document["precheck"] = precheck_payload

    requested_options = {
        "method": config.method,
        "backend": config.backend,
        "language": config.language,
        "formula_enabled": config.formula_enabled,
        "table_enabled": config.table_enabled,
    }
    previous_applied = previous_pipeline.get("applied_mineru_options")
    applied_options = (
        requested_options
        if command is not None
        else previous_applied
        if isinstance(previous_applied, dict)
        else None
    )

    document["parser"]["formula_enabled"] = (
        applied_options.get("formula_enabled")
        if applied_options
        else None
    )
    document["parser"]["table_enabled"] = (
        applied_options.get("table_enabled")
        if applied_options
        else None
    )
    document["pipeline"] = {
        "created_at": utc_now_iso(),
        "mineru_reused": mineru_reused,
        "mineru_executable": (
            project_relative(Path(command[0]), project_root)
            if command
            else previous_pipeline.get("mineru_executable")
        ),
        "mineru_command": command or previous_pipeline.get("mineru_command"),
        "requested_mineru_options": requested_options,
        "applied_mineru_options": applied_options,
        "mineru_output_directory": project_relative(
            artifacts.output_directory,
            project_root,
        ),
        "adapter_report": adapter_report,
    }
    document["formulas"] = formulas
    document["tables"] = tables
    document["figures"] = figures

    write_json_atomic(paper_json, document)

    return PaperParseResult(
        paper_id=ingest_result.paper_id,
        sha256=ingest_result.sha256,
        raw_pdf=ingest_result.source_path,
        paper_json=paper_json,
        mineru_output_directory=artifacts.output_directory,
        mineru_reused=mineru_reused,
        page_count=int(document.get("page_count", 0)),
        block_count=len(blocks),
        formula_count=len(formulas),
        table_count=len(tables),
        figure_count=len(figures),
        precheck=asdict(precheck_result),
    )
