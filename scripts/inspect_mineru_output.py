from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint
from typing import Any


PAPER_ID = "P_2e250a42c5f9"

PARSED_ROOT = (
    Path("data")
    / "parsed"
    / PAPER_ID
    / "mineru"
)


def find_one_file(pattern: str) -> Path:
    """在 MinerU 输出目录中查找唯一匹配文件。"""

    matches = list(PARSED_ROOT.rglob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"没有找到文件：{pattern}\n"
            f"搜索目录：{PARSED_ROOT.resolve()}"
        )

    if len(matches) > 1:
        print(f"警告：{pattern} 找到多个文件，将使用第一个：")

        for path in matches:
            print(f"  - {path}")

    return matches[0]


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def shorten(value: Any, max_length: int = 300) -> Any:
    """缩短过长文本，避免终端输出太多内容。"""

    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "...<省略>"

    if isinstance(value, list):
        return [
            shorten(item, max_length=max_length)
            for item in value[:5]
        ]

    if isinstance(value, dict):
        return {
            key: shorten(item, max_length=max_length)
            for key, item in value.items()
        }

    return value


def inspect_content_list(path: Path) -> None:
    """检查 content_list_v2.json 的整体结构和前几个块。"""

    data = load_json(path)

    print("\n" + "=" * 80)
    print("CONTENT_LIST_V2")
    print("=" * 80)

    print("文件：", path)
    print("根对象类型：", type(data).__name__)

    if isinstance(data, dict):
        print("根字段：")
        pprint(list(data.keys()))

        for key, value in data.items():
            print(
                f"字段 {key!r} 的类型："
                f"{type(value).__name__}"
            )

        candidate_lists = [
            (key, value)
            for key, value in data.items()
            if isinstance(value, list)
        ]

        for key, value in candidate_lists:
            print(f"\n列表字段：{key}")
            print("元素数量：", len(value))

            for index, item in enumerate(value[:5]):
                print(f"\n--- {key}[{index}] ---")
                pprint(shorten(item))

    elif isinstance(data, list):
        print("元素数量：", len(data))

        for index, item in enumerate(data[:8]):
            print(f"\n--- block[{index}] ---")

            if isinstance(item, dict):
                print("字段：", list(item.keys()))

            pprint(shorten(item))

    else:
        print("无法识别的数据结构：")
        pprint(data)


def inspect_middle(path: Path) -> None:
    """检查 middle.json 的页级和块级结构。"""

    data = load_json(path)

    print("\n" + "=" * 80)
    print("MIDDLE_JSON")
    print("=" * 80)

    print("文件：", path)
    print("根对象类型：", type(data).__name__)

    if not isinstance(data, dict):
        pprint(shorten(data))
        return

    print("根字段：")
    pprint(list(data.keys()))

    for key, value in data.items():
        print(
            f"字段 {key!r} 的类型："
            f"{type(value).__name__}"
        )

    pdf_info = data.get("pdf_info")

    if isinstance(pdf_info, list):
        print("\npdf_info 页数：", len(pdf_info))

        for page_index, page_info in enumerate(pdf_info[:2]):
            print(f"\n--- 第 {page_index + 1} 页 ---")

            if isinstance(page_info, dict):
                print("页面字段：")
                pprint(list(page_info.keys()))

                para_blocks = page_info.get("para_blocks")

                if isinstance(para_blocks, list):
                    print(
                        "para_blocks 数量：",
                        len(para_blocks),
                    )

                    for block_index, block in enumerate(
                        para_blocks[:5]
                    ):
                        print(
                            f"\n第 {page_index + 1} 页"
                            f" block[{block_index}]"
                        )

                        if isinstance(block, dict):
                            print(
                                "block 字段：",
                                list(block.keys()),
                            )

                        pprint(shorten(block))

    else:
        print("\n没有找到 list 类型的 pdf_info。")


def main() -> None:
    if not PARSED_ROOT.exists():
        raise FileNotFoundError(
            f"MinerU输出目录不存在：{PARSED_ROOT.resolve()}"
        )

    content_list_v2_path = find_one_file(
        "*_content_list_v2.json"
    )

    middle_path = find_one_file(
        "*_middle.json"
    )

    inspect_content_list(content_list_v2_path)
    inspect_middle(middle_path)


if __name__ == "__main__":
    main()