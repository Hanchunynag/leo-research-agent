"""Run lightweight PaddleOCR formula recognition in the isolated venv."""

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

    # Keep model downloads inside the mounted project data directory. This is
    # also the Docker runtime path, so PaddleOCR-VL never depends on a host
    # user's ~/.paddlex cache or another local Python environment.
    args.cache_home.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(args.cache_home.resolve())

    from paddleocr_compat import enable_headless_opencv_compat

    enable_headless_opencv_compat()

    from paddleocr import FormulaRecognitionPipeline, __version__

    # FormulaRecognitionPipeline uses the dedicated LaTeX formula model and
    # does not load the 0.9B PaddleOCR-VL language model. This keeps formula
    # fallback usable on ordinary CPU Docker deployments.
    pipeline = FormulaRecognitionPipeline(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
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
            )
            if not results:
                raise RuntimeError("PaddleOCR 公式管线未返回识别结果。")
            payload = results[0].json
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload["res"]
            parsing = payload.get("formula_res_list", [])
            latex = None
            if isinstance(parsing, list):
                for parsing_item in parsing:
                    if isinstance(parsing_item, dict):
                        latex = _normalize_formula(
                            parsing_item.get("rec_formula")
                        )
                        if latex:
                            break
            if not latex:
                raise RuntimeError("PaddleOCR 公式管线未返回可用 LaTeX。")
            result = {
                "done": True,
                "paddleocr_version": __version__,
                "input_image": str(input_image.resolve()),
                "latex": latex,
                "engine": "FormulaRecognitionPipeline",
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
