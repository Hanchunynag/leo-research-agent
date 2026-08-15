"""Run PaddleOCR-VL formula recognition in the isolated PaddleOCR venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_formula(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    if result.startswith("$$") and result.endswith("$$"):
        result = result[2:-2].strip()
    elif result.startswith("$") and result.endswith("$"):
        result = result[1:-1].strip()
    return result or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--cache-home", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise RuntimeError("公式恢复 manifest 必须是 JSON 数组。")

    # Do not override PADDLE_PDX_CACHE_HOME here. The validated PaddleOCR-VL
    # model is installed in the user's standard ~/.paddlex cache. The parent
    # process still passes --cache-home for audit compatibility, but changing
    # the model cache root would hide the existing local model.
    args.cache_home.mkdir(parents=True, exist_ok=True)

    from paddleocr import PaddleOCRVL, __version__

    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_for_image_block=False,
        device=args.device,
    )
    for item in manifest:
        if not isinstance(item, dict):
            continue
        input_image = Path(str(item.get("input_image", ""))).expanduser()
        output_json = Path(str(item.get("output_json", ""))).expanduser()
        result: dict[str, Any]
        try:
            results = pipeline.predict(
                str(input_image),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_ocr_for_image_block=False,
                prompt_label="formula",
                format_block_content=True,
                max_new_tokens=1024,
            )
            if not results:
                raise RuntimeError("PaddleOCR-VL 未返回公式识别结果。")
            payload = results[0].json
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload["res"]
            parsing = payload.get("parsing_res_list", [])
            latex = None
            if isinstance(parsing, list):
                for parsing_item in parsing:
                    if (
                        isinstance(parsing_item, dict)
                        and parsing_item.get("block_label") == "formula"
                    ):
                        latex = _normalize_formula(
                            parsing_item.get("block_content")
                        )
                        if latex:
                            break
            if not latex:
                raise RuntimeError("PaddleOCR-VL 未返回可用公式 LaTeX。")
            result = {
                "done": True,
                "paddleocr_version": __version__,
                "input_image": str(input_image.resolve()),
                "latex": latex,
                "prompt_label": "formula",
            }
        except Exception as error:
            result = {
                "done": False,
                "input_image": str(input_image),
                "error": str(error),
            }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
