"""统一论文解析流程的 Gradio 界面。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from html import escape
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterator

import gradio as gr

from app.ingestion.batch import (
    BatchParseReport,
    BatchProgressStage,
    batch_parse_files,
)
from app.knowledge.catalog import rebuild_catalog
from app.parsing.pipeline import (
    MinerUBackend,
    MinerUMethod,
    PaperParseConfig,
    PaperParseResult,
    PaperParseStage,
    parse_paper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"


METHOD_CHOICES = [
    ("自动识别（推荐）", "auto"),
    ("文本解析", "txt"),
    ("OCR（扫描件）", "ocr"),
]
BACKEND_CHOICES = [
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
    "vlm-http-client",
    "hybrid-http-client",
]
PAPER_STAGE_LABELS: dict[PaperParseStage, tuple[str, float]] = {
    "ingesting": ("正在入库", 0.05),
    "prechecking": ("正在预检查", 0.10),
    "waiting_for_mineru": ("正在检查 MinerU 产物", 0.15),
    "running_mineru": ("MinerU 正在解析", 0.20),
    "normalizing": ("正在标准化解析结果", 0.85),
    "writing": ("正在写入 paper.json", 0.95),
}


def render_parse_progress(
    *,
    label: str,
    detail: str,
    progress: float,
    color: str = "#f97316",
) -> str:
    """渲染页面内唯一的批处理进度区域。"""

    percentage = max(0.0, min(100.0, progress * 100))
    safe_label = escape(label)
    safe_detail = escape(detail)
    safe_color = escape(color)
    return f"""
    <div style="
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 16px;
        background: var(--block-background-fill);
    ">
      <div style="
          display: flex;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 10px;
      ">
        <strong>{safe_label}</strong>
        <span style="font-variant-numeric: tabular-nums;">
          {percentage:.1f}%
        </span>
      </div>
      <div style="
          height: 10px;
          overflow: hidden;
          border-radius: 999px;
          background: var(--border-color-primary);
          margin-bottom: 10px;
      ">
        <div
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow="{percentage:.1f}"
          style="
              width: {percentage:.1f}%;
              height: 100%;
              border-radius: inherit;
              background: {safe_color};
              transition: width 180ms ease;
          "
        ></div>
      </div>
      <div style="
          color: var(--body-text-color-subdued);
          overflow-wrap: anywhere;
      ">{safe_detail}</div>
    </div>
    """


def make_json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-compatible values.

    Gradio's JSON component accepts dictionaries and lists, but objects such
    as pathlib.Path and datetime need to be converted first.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    return value


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an ingestion result into a serializable dictionary.

    This function supports:
    - dataclasses
    - Pydantic models
    - dictionaries
    - ordinary Python objects
    """

    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)

    elif hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")

    elif isinstance(result, dict):
        payload = result

    elif hasattr(result, "__dict__"):
        payload = vars(result)

    else:
        raise TypeError(f"Unsupported ingestion result type: {type(result).__name__}")

    safe_payload = make_json_safe(payload)

    if not isinstance(safe_payload, dict):
        raise TypeError("The ingestion result must be converted to a dictionary.")

    return safe_payload


def list_local_papers() -> list[list[str]]:
    """Return the normalized PDFs currently stored in data/raw."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []

    for paper_directory in sorted(RAW_DIR.iterdir()):
        if not paper_directory.is_dir():
            continue

        pdf_files = sorted(paper_directory.glob("*.pdf"))

        if not pdf_files:
            continue

        pdf_path = pdf_files[0]
        stat = pdf_path.stat()
        parsed_directory = DATA_ROOT / "parsed" / paper_directory.name / "mineru"
        paper_json = DATA_ROOT / "canonical" / paper_directory.name / "paper.json"

        rows.append(
            [
                paper_directory.name,
                pdf_path.name,
                f"{stat.st_size / 1024 / 1024:.2f} MB",
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "是" if parsed_directory.exists() else "否",
                "是" if paper_json.exists() else "否",
            ]
        )

    return rows


def parse_from_ui(
    uploaded_pdf: str | None,
    method: MinerUMethod,
    backend: MinerUBackend,
    formula_enabled: bool,
    table_enabled: bool,
    force_mineru: bool,
) -> Iterator[tuple[Any, ...]]:
    """通过唯一 pipeline 解析上传的 PDF。"""

    if not uploaded_pdf:
        yield (
            render_parse_progress(
                label="等待上传",
                detail="请选择一篇 PDF 论文。",
                progress=0,
            ),
            "### 入库失败\n请先选择一个 PDF 文件。",
            {},
            None,
            None,
            {},
            list_local_papers(),
        )
        return

    pdf_path = Path(uploaded_pdf)
    progress_updates: Queue[PaperParseStage | None] = Queue()
    results: list[PaperParseResult] = []
    worker_errors: list[Exception] = []

    def run_single() -> None:
        try:
            results.append(
                parse_paper(
                    input_path=pdf_path,
                    config=PaperParseConfig(
                        project_root=PROJECT_ROOT,
                        method=method,
                        backend=backend,
                        formula_enabled=formula_enabled,
                        table_enabled=table_enabled,
                        force_mineru=force_mineru,
                    ),
                    progress_callback=progress_updates.put,
                )
            )
            rebuild_catalog(PROJECT_ROOT)
        except Exception as error:
            worker_errors.append(error)
        finally:
            progress_updates.put(None)

    yield (
        render_parse_progress(
            label="准备解析",
            detail=pdf_path.name,
            progress=0,
        ),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    worker = Thread(
        target=run_single,
        name="leo-single-parser",
        daemon=True,
    )
    worker.start()

    while True:
        stage = progress_updates.get()

        if stage is None:
            break

        label, progress = PAPER_STAGE_LABELS[stage]
        yield (
            render_parse_progress(
                label=label,
                detail=pdf_path.name,
                progress=progress,
            ),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    worker.join()

    if worker_errors:
        error = worker_errors[0]
        status = (
            "### 入库失败\n"
            f"- **错误类型：** `{type(error).__name__}`\n"
            f"- **错误信息：** {error}"
        )
        yield (
            render_parse_progress(
                label="解析失败",
                detail=f"{type(error).__name__}: {error}",
                progress=0,
                color="#dc2626",
            ),
            status,
            {},
            None,
            None,
            {},
            list_local_papers(),
        )
        return

    result = results[0]
    payload = result_to_dict(result)
    stored_pdf = str(result.raw_pdf)
    paper_json = str(result.paper_json)
    precheck_payload = result.precheck
    paper_id = str(payload.get("paper_id", "未知"))
    status = (
        "### 解析完成\n"
        f"- **Paper ID：** `{paper_id}`\n"
        f"- **页数：** {result.page_count}\n"
        f"- **结构块：** {result.block_count}\n"
        f"- **公式：** {result.formula_count}\n"
        f"- **表格：** {result.table_count}\n"
        f"- **复用 MinerU 结果：** {'是' if result.mineru_reused else '否'}"
    )

    yield (
        render_parse_progress(
            label="单篇解析完成",
            detail=f"{pdf_path.name} · Paper ID: {paper_id}",
            progress=1,
            color="#16a34a",
        ),
        status,
        payload,
        stored_pdf,
        paper_json,
        precheck_payload,
        list_local_papers(),
    )


def parse_batch_from_ui(
    uploaded_pdfs: list[str] | str | None,
    method: MinerUMethod,
    backend: MinerUBackend,
    formula_enabled: bool,
    table_enabled: bool,
    force_mineru: bool,
) -> Iterator[tuple[Any, ...]]:
    """批量解析多个上传文件，并返回统计与失败明细。"""

    if not uploaded_pdfs:
        yield (
            render_parse_progress(
                label="等待批量上传",
                detail="请选择至少一篇 PDF。",
                progress=0,
            ),
            "### 批量入库失败\n请至少选择一个 PDF 文件。",
            [],
            [],
            {},
            list_local_papers(),
        )
        return

    uploaded_values = (
        [uploaded_pdfs] if isinstance(uploaded_pdfs, str) else uploaded_pdfs
    )
    pdf_paths = [Path(value) for value in uploaded_values]
    total = len(pdf_paths)
    progress_updates: Queue[tuple[int, int, Path, BatchProgressStage] | None] = Queue()
    reports: list[BatchParseReport] = []
    worker_errors: list[Exception] = []

    def update_progress(
        completed: int,
        item_total: int,
        current_pdf: Path,
        stage: BatchProgressStage,
    ) -> None:
        progress_updates.put((completed, item_total, current_pdf, stage))

    def run_batch() -> None:
        try:
            reports.append(
                batch_parse_files(
                    input_paths=pdf_paths,
                    config=PaperParseConfig(
                        project_root=PROJECT_ROOT,
                        method=method,
                        backend=backend,
                        formula_enabled=formula_enabled,
                        table_enabled=table_enabled,
                        force_mineru=force_mineru,
                    ),
                    input_source="gradio-multiple-upload",
                    progress_callback=update_progress,
                )
            )
        except Exception as error:
            worker_errors.append(error)
        finally:
            progress_updates.put(None)

    yield (
        render_parse_progress(
            label=f"准备解析 {total} 篇 PDF",
            detail="正在创建批处理任务……",
            progress=0,
        ),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    worker = Thread(
        target=run_batch,
        name="leo-batch-parser",
        daemon=True,
    )
    worker.start()

    stage_labels: dict[BatchProgressStage, tuple[str, float]] = {
        "ingesting": ("正在入库", 0.05),
        "prechecking": ("正在预检查", 0.10),
        "waiting_for_mineru": ("正在等待 MinerU 解析锁", 0.15),
        "running_mineru": ("MinerU 正在解析", 0.20),
        "normalizing": ("正在标准化解析结果", 0.85),
        "writing": ("正在写入 paper.json", 0.95),
        "completed": ("处理完成", 1.0),
    }

    while True:
        update = progress_updates.get()

        if update is None:
            break

        completed, item_total, current_pdf, stage = update
        stage_label, item_fraction = stage_labels[stage]
        overall_progress = (
            completed / item_total
            if stage == "completed"
            else (completed + item_fraction) / item_total
        )
        item_number = completed if stage == "completed" else completed + 1
        yield (
            render_parse_progress(
                label=f"{stage_label} · 第 {item_number}/{item_total} 篇",
                detail=current_pdf.name,
                progress=overall_progress,
            ),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    worker.join()

    if worker_errors:
        error = worker_errors[0]
        yield (
            render_parse_progress(
                label="批处理失败",
                detail=f"{type(error).__name__}: {error}",
                progress=0,
                color="#dc2626",
            ),
            "### 批量入库失败\n"
            f"- **错误类型：** `{type(error).__name__}`\n"
            f"- **错误信息：** {error}",
            [],
            [
                [
                    "批处理任务",
                    type(error).__name__,
                    str(error),
                ]
            ],
            {},
            list_local_papers(),
        )
        return

    report = reports[0]
    status_labels = {
        "success": "新解析成功",
        "reused": "复用已有结果",
        "failed": "失败",
    }
    result_rows = [
        [
            Path(item.input_path).name,
            status_labels[item.status],
            item.paper_id or "",
            item.error_message or "",
        ]
        for item in report.items
    ]
    failure_rows = [
        [
            Path(item.input_path).name,
            item.error_type or "",
            item.error_message or "",
        ]
        for item in report.items
        if item.status == "failed"
    ]
    heading = (
        "### 批量入库完成"
        if report.failed_count == 0 and not report.catalog_issues
        else "### 批量入库完成，但存在失败项"
    )
    summary = (
        f"{heading}\n"
        f"- **总数：** {report.total_count}\n"
        f"- **新解析成功：** {report.success_count}\n"
        f"- **复用已有结果：** {report.reused_count}\n"
        f"- **失败：** {report.failed_count}\n"
        f"- **目录记录：** {report.catalog_record_count}\n"
        f"- **目录问题：** {len(report.catalog_issues)}\n"
        f"- **详细报告：** `{report.report_path}`"
    )

    yield (
        render_parse_progress(
            label="批量解析完成",
            detail=(
                f"成功或复用 {report.success_count + report.reused_count} 篇，"
                f"失败 {report.failed_count} 篇。"
            ),
            progress=1,
            color=("#16a34a" if report.failed_count == 0 else "#d97706"),
        ),
        summary,
        result_rows,
        failure_rows,
        make_json_safe(report.to_dict()),
        list_local_papers(),
    )


def build_demo() -> gr.Blocks:
    """Build and return the Gradio application."""

    with gr.Blocks(title="LEO Research Agent") as demo:
        gr.Markdown(
            """
            # LEO Research Agent

            面向低轨机会信号定位研究的本地论文知识库。

            一次完成：**入库、预检查、MinerU 解析和统一 paper.json 输出**。
            """
        )

        with gr.Tabs():
            with gr.Tab("单篇入库"):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_pdf = gr.File(
                            label="选择论文 PDF",
                            file_types=[".pdf"],
                            file_count="single",
                            type="filepath",
                        )
                        with gr.Accordion("解析设置", open=False):
                            single_method = gr.Dropdown(
                                choices=METHOD_CHOICES,
                                value="auto",
                                label="解析方式",
                            )
                            single_backend = gr.Dropdown(
                                choices=BACKEND_CHOICES,
                                value="pipeline",
                                label="MinerU backend",
                            )
                            with gr.Row():
                                single_formula = gr.Checkbox(
                                    value=True,
                                    label="解析公式",
                                )
                                single_table = gr.Checkbox(
                                    value=True,
                                    label="解析表格",
                                )
                                single_force = gr.Checkbox(
                                    value=False,
                                    label="强制重新运行 MinerU",
                                )

                        ingest_button = gr.Button(
                            "开始单篇解析",
                            variant="primary",
                        )

                    with gr.Column(scale=1):
                        single_progress_output = gr.HTML(
                            value=render_parse_progress(
                                label="等待上传",
                                detail="请选择一篇 PDF，解析进度会显示在这里。",
                                progress=0,
                            ),
                            label="单篇解析进度",
                        )
                        status_output = gr.Markdown(
                            value="### 等待上传\n请选择一篇 PDF 论文。"
                        )

                        with gr.Row():
                            stored_pdf_output = gr.File(
                                label="本地原始 PDF",
                                interactive=False,
                            )

                            paper_json_output = gr.File(
                                label="统一结果 paper.json",
                                interactive=False,
                            )

                with gr.Row():
                    with gr.Accordion("论文入库元数据", open=False):
                        metadata_output = gr.JSON(
                            label="论文入库元数据",
                            value={},
                        )
                    with gr.Accordion("PDF 预检查结果", open=False):
                        precheck_output = gr.JSON(
                            label="PDF 预检查结果",
                            value={},
                        )

            with gr.Tab("批量入库"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_uploaded_pdfs = gr.File(
                            label="选择多个论文 PDF",
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                        )
                        batch_method = gr.Dropdown(
                            choices=METHOD_CHOICES,
                            value="auto",
                            label="解析方式",
                        )
                        batch_backend = gr.Dropdown(
                            choices=BACKEND_CHOICES,
                            value="pipeline",
                            label="MinerU backend",
                        )
                        with gr.Row():
                            batch_formula = gr.Checkbox(
                                value=True,
                                label="解析公式",
                            )
                            batch_table = gr.Checkbox(
                                value=True,
                                label="解析表格",
                            )
                            batch_force = gr.Checkbox(
                                value=False,
                                label="强制重新运行 MinerU",
                            )
                        batch_button = gr.Button(
                            "开始批量解析",
                            variant="primary",
                        )

                    with gr.Column(scale=1):
                        batch_progress_output = gr.HTML(
                            value=render_parse_progress(
                                label="等待批量上传",
                                detail="请选择多篇 PDF，解析进度会显示在这里。",
                                progress=0,
                            ),
                            label="批处理进度",
                        )
                        batch_status_output = gr.Markdown(
                            value=(
                                "### 等待批量上传\n"
                                "请选择多篇 PDF，解析时会显示总体进度。"
                            )
                        )
                        batch_report_output = gr.JSON(
                            label="批处理详细报告",
                            value={},
                        )

                with gr.Row():
                    batch_results_output = gr.Dataframe(
                        headers=[
                            "PDF文件",
                            "状态",
                            "Paper ID",
                            "错误信息",
                        ],
                        datatype=["str", "str", "str", "str"],
                        value=[],
                        label="逐篇处理结果",
                        interactive=False,
                        wrap=True,
                    )
                    batch_failures_output = gr.Dataframe(
                        headers=[
                            "PDF文件",
                            "错误类型",
                            "错误信息",
                        ],
                        datatype=["str", "str", "str"],
                        value=[],
                        label="失败列表",
                        interactive=False,
                        wrap=True,
                    )

            with gr.Tab("本地论文库"):
                refresh_button = gr.Button("刷新论文列表")

                library_table = gr.Dataframe(
                    headers=[
                        "Paper ID",
                        "PDF文件",
                        "文件大小",
                        "更新时间",
                        "MinerU产物",
                        "paper.json",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[],
                    interactive=False,
                )

        # Gradio 在运行时动态注册事件方法；Linux 类型声明未暴露 click。
        ingest_button.click(  # type: ignore[attr-defined]
            fn=parse_from_ui,
            inputs=[
                uploaded_pdf,
                single_method,
                single_backend,
                single_formula,
                single_table,
                single_force,
            ],
            outputs=[
                single_progress_output,
                status_output,
                metadata_output,
                stored_pdf_output,
                paper_json_output,
                precheck_output,
                library_table,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="paper-parsing",
        )

        refresh_button.click(  # type: ignore[attr-defined]
            fn=list_local_papers,
            inputs=None,
            outputs=library_table,
        )

        batch_button.click(  # type: ignore[attr-defined]
            fn=parse_batch_from_ui,
            inputs=[
                batch_uploaded_pdfs,
                batch_method,
                batch_backend,
                batch_formula,
                batch_table,
                batch_force,
            ],
            outputs=[
                batch_progress_output,
                batch_status_output,
                batch_results_output,
                batch_failures_output,
                batch_report_output,
                library_table,
            ],
            # Gradio 的全局进度会覆盖每个输出组件；页面改用上方唯一的
            # batch_progress_output，因此彻底隐藏框架自带的进度遮罩。
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="paper-parsing",
        )

        demo.load(
            fn=list_local_papers,
            inputs=None,
            outputs=library_table,
        )

    return demo


def main() -> None:
    """Launch the local Gradio application."""

    demo = build_demo()

    demo.queue()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
