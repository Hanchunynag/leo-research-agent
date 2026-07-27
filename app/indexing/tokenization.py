"""Chunk 计数与 BM25 共用的轻量、确定性分词。"""

from __future__ import annotations

import re
import unicodedata


WORD_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]",
    flags=re.IGNORECASE,
)


def normalize_search_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def tokenize(value: str) -> list[str]:
    """英文按词、中文同时保留单字和相邻双字。"""

    raw = WORD_PATTERN.findall(normalize_search_text(value))
    tokens: list[str] = []
    chinese_run: list[str] = []

    def flush_chinese() -> None:
        if not chinese_run:
            return
        tokens.extend(chinese_run)
        tokens.extend(
            chinese_run[index] + chinese_run[index + 1]
            for index in range(len(chinese_run) - 1)
        )
        chinese_run.clear()

    for token in raw:
        if len(token) == 1 and "\u3400" <= token <= "\u9fff":
            chinese_run.append(token)
        else:
            flush_chinese()
            tokens.append(token)
    flush_chinese()
    return tokens


def token_count(value: str) -> int:
    return len(tokenize(value))
