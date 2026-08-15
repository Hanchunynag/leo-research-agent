"""Run PaddleOCR table recognition in its dedicated virtual environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover one existing MinerU table image with PaddleOCR."
    )
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cache-home", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # PaddleX creates model-download locks under this directory. Keep the
    # cache inside the project so the isolated worker does not depend on the
    # parent process' home-directory permissions.
    args.cache_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(args.cache_home.resolve()))

    from paddleocr import TableRecognitionPipelineV2, __version__

    pipeline = TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
    )
    results = pipeline.predict(
        str(args.input_image),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
        use_wired_table_cells_trans_to_html=True,
        use_wireless_table_cells_trans_to_html=True,
        use_table_orientation_classify=True,
        use_ocr_results_with_table_cells=True,
    )

    if not results:
        raise RuntimeError("PaddleOCR 未返回表格识别结果。")

    result_json = results[0].json
    if isinstance(result_json, dict) and isinstance(result_json.get("res"), dict):
        result_json = result_json["res"]

    table_results = result_json.get("table_res_list", [])
    if not isinstance(table_results, list) or not table_results:
        raise RuntimeError("PaddleOCR 未返回 table_res_list。")

    table_result = table_results[0]
    if not isinstance(table_result, dict):
        raise RuntimeError("PaddleOCR table_res_list 格式异常。")

    payload = {
        "done": True,
        "paddleocr_version": __version__,
        "input_image": str(args.input_image.resolve()),
        "pred_html": table_result.get("pred_html"),
        "cell_box_list": _jsonable(table_result.get("cell_box_list", [])),
        "table_ocr_pred": _jsonable(table_result.get("table_ocr_pred", [])),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
