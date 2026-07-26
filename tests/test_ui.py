from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.ingestion.batch import BatchItemResult, BatchParseReport
from app.parsing.pipeline import PaperParseResult
from app.ui import gradio_app


def test_single_ui_streams_stages_and_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "single.pdf"
    paper_json = tmp_path / "paper.json"
    captured_config: list[Any] = []

    def fake_parse_paper(
        input_path: Path,
        config: Any,
        progress_callback: Any,
    ) -> PaperParseResult:
        assert input_path == pdf
        captured_config.append(config)
        for stage in gradio_app.PAPER_STAGE_LABELS:
            progress_callback(stage)
        return PaperParseResult(
            paper_id="P_SINGLE",
            sha256="abc",
            raw_pdf=pdf,
            paper_json=paper_json,
            mineru_output_directory=tmp_path / "mineru",
            mineru_reused=False,
            page_count=12,
            block_count=34,
            formula_count=5,
            table_count=2,
            figure_count=3,
            precheck={"pdf_type": "native_text"},
        )

    monkeypatch.setattr(gradio_app, "parse_paper", fake_parse_paper)
    monkeypatch.setattr(gradio_app, "rebuild_catalog", lambda _: None)
    monkeypatch.setattr(gradio_app, "list_local_papers", lambda: [])

    updates = list(
        gradio_app.parse_from_ui(
            uploaded_pdf=str(pdf),
            method="txt",
            backend="pipeline",
            formula_enabled=False,
            table_enabled=True,
            force_mineru=True,
        )
    )
    final = updates[-1]

    assert len(updates) == 8
    assert "MinerU 正在解析" in updates[4][0]
    assert 'aria-valuenow="20.0"' in updates[4][0]
    assert "单篇解析完成" in final[0]
    assert "P_SINGLE" in final[1]
    assert final[2]["paper_id"] == "P_SINGLE"
    assert final[3] == str(pdf)
    assert final[4] == str(paper_json)
    assert final[5] == {"pdf_type": "native_text"}
    assert captured_config[0].method == "txt"
    assert captured_config[0].formula_enabled is False
    assert captured_config[0].force_mineru is True


def test_batch_ui_requires_at_least_one_pdf() -> None:
    result = list(
        gradio_app.parse_batch_from_ui(
            uploaded_pdfs=None,
            method="auto",
            backend="pipeline",
            formula_enabled=True,
            table_enabled=True,
            force_mineru=False,
        )
    )
    final = result[-1]

    assert "等待批量上传" in final[0]
    assert "至少选择一个 PDF" in final[1]
    assert final[2] == []
    assert final[3] == []
    assert final[4] == {}


def test_batch_ui_displays_progress_counts_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_paths: list[Path] = []

    def fake_batch_parse_files(
        input_paths: list[Path],
        **kwargs: Any,
    ) -> BatchParseReport:
        captured_paths.extend(input_paths)
        callback = kwargs["progress_callback"]
        for index, path in enumerate(input_paths, start=1):
            callback(index - 1, len(input_paths), path, "running_mineru")
            callback(index, len(input_paths), path, "completed")

        return BatchParseReport(
            run_id="test-run",
            input_source="gradio-multiple-upload",
            recursive=False,
            started_at="2026-07-25T00:00:00+00:00",
            finished_at="2026-07-25T00:01:00+00:00",
            total_count=3,
            success_count=1,
            reused_count=1,
            failed_count=1,
            catalog_record_count=2,
            catalog_issues=[],
            report_path=tmp_path / "last_batch_report.json",
            items=[
                BatchItemResult(
                    input_path=str(tmp_path / "new.pdf"),
                    status="success",
                    paper_id="P_111111111111",
                    paper_json=str(tmp_path / "new.json"),
                    error_type=None,
                    error_message=None,
                ),
                BatchItemResult(
                    input_path=str(tmp_path / "reused.pdf"),
                    status="reused",
                    paper_id="P_222222222222",
                    paper_json=str(tmp_path / "reused.json"),
                    error_type=None,
                    error_message=None,
                ),
                BatchItemResult(
                    input_path=str(tmp_path / "broken.pdf"),
                    status="failed",
                    paper_id=None,
                    paper_json=None,
                    error_type="ValueError",
                    error_message="文件损坏",
                ),
            ],
        )

    monkeypatch.setattr(
        gradio_app,
        "batch_parse_files",
        fake_batch_parse_files,
    )
    monkeypatch.setattr(
        gradio_app,
        "list_local_papers",
        lambda: [],
    )

    updates = list(
        gradio_app.parse_batch_from_ui(
            uploaded_pdfs=[
                str(tmp_path / "new.pdf"),
                str(tmp_path / "reused.pdf"),
                str(tmp_path / "broken.pdf"),
            ],
            method="auto",
            backend="pipeline",
            formula_enabled=True,
            table_enabled=True,
            force_mineru=False,
        )
    )
    result = updates[-1]

    _, summary, rows, failures, report, _ = result
    assert "总数：** 3" in summary
    assert "新解析成功：** 1" in summary
    assert "复用已有结果：** 1" in summary
    assert "失败：** 1" in summary
    assert [row[1] for row in rows] == [
        "新解析成功",
        "复用已有结果",
        "失败",
    ]
    assert failures == [["broken.pdf", "ValueError", "文件损坏"]]
    assert report["failed_count"] == 1
    assert len(captured_paths) == 3
    assert len(updates) == 8
    assert "MinerU 正在解析" in updates[1][0]
    assert 'aria-valuenow="6.7"' in updates[1][0]
    assert 'aria-valuenow="33.3"' in updates[2][0]
    assert "批量解析完成" in result[0]


def test_parse_events_share_one_embedded_progress_region() -> None:
    demo = gradio_app.build_demo()
    parse_functions = {
        function.name: function
        for function in demo.fns.values()
        if function.name in {"parse_from_ui", "parse_batch_from_ui"}
    }
    parse_dependencies = {
        dependency["api_name"]: dependency
        for dependency in demo.config["dependencies"]
        if dependency["api_name"] in parse_functions
    }

    assert set(parse_dependencies) == set(parse_functions)

    expected_output_counts = {
        "parse_from_ui": 7,
        "parse_batch_from_ui": 6,
    }
    for api_name, dependency in parse_dependencies.items():
        assert dependency["show_progress"] == "hidden"
        assert dependency["trigger_mode"] == "once"
        assert len(dependency["outputs"]) == expected_output_counts[api_name]

    for function in parse_functions.values():
        assert function.concurrency_limit == 1
        assert function.concurrency_id == "paper-parsing"
