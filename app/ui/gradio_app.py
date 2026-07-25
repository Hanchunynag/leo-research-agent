"""统一论文解析流程的 Gradio 界面。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import gradio as gr
from app.parsing.pipeline import PaperParseConfig, parse_paper


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"


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
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

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
        raise TypeError(
            f"Unsupported ingestion result type: {type(result).__name__}"
        )

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
        parsed_directory = (
            DATA_ROOT
            / "parsed"
            / paper_directory.name
            / "mineru"
        )
        paper_json = (
            DATA_ROOT
            / "canonical"
            / paper_directory.name
            / "paper.json"
        )

        rows.append(
            [
                paper_directory.name,
                pdf_path.name,
                f"{stat.st_size / 1024 / 1024:.2f} MB",
                datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "是" if parsed_directory.exists() else "否",
                "是" if paper_json.exists() else "否",
            ]
        )

    return rows


def parse_from_ui(
    uploaded_pdf: str | None,
) -> tuple[
    str,
    dict[str, Any],
    str | None,
    str | None,
    dict[str, Any],
    list[list[str]],
]:
    """通过唯一 pipeline 解析上传的 PDF。"""

    if not uploaded_pdf:
        return (
            "### 入库失败\n请先选择一个 PDF 文件。",
            {},
            None,
            None,
            {},
            list_local_papers(),
        )

    try:
        result = parse_paper(
            input_path=Path(uploaded_pdf),
            config=PaperParseConfig(
                project_root=PROJECT_ROOT,
            ),
        )

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

        return (
            status,
            payload,
            stored_pdf,
            paper_json,
            precheck_payload,
            list_local_papers(),
        )

    except Exception as error:
        status = (
            "### 入库失败\n"
            f"- **错误类型：** `{type(error).__name__}`\n"
            f"- **错误信息：** {error}"
        )

        return (
            status,
            {},
            None,
            None,
            {},
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
            with gr.Tab("论文入库"):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_pdf = gr.File(
                            label="选择论文 PDF",
                            file_types=[".pdf"],
                            file_count="single",
                            type="filepath",
                        )

                        ingest_button = gr.Button(
                            "开始解析",
                            variant="primary",
                        )

                    with gr.Column(scale=1):
                        status_output = gr.Markdown(
                            value="### 等待上传\n请选择一篇 PDF 论文。"
                        )

                        stored_pdf_output = gr.File(
                            label="本地原始 PDF",
                            interactive=False,
                        )

                        paper_json_output = gr.File(
                            label="统一解析结果 paper.json",
                            interactive=False,
                        )

                metadata_output = gr.JSON(
                    label="论文入库元数据",
                    value={},
                )
                precheck_output = gr.JSON(
                    label="PDF预检查结果",
                    value={},
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

        ingest_button.click(
            fn=parse_from_ui,
            inputs=[uploaded_pdf],
            outputs=[
                status_output,
                metadata_output,
                stored_pdf_output,
                paper_json_output,
                precheck_output,
                library_table,
            ],
        )

        refresh_button.click(
            fn=list_local_papers,
            inputs=None,
            outputs=library_table,
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
