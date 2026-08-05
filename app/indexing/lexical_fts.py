"""Incremental SQLite FTS5 index with epoch-visible version rows."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.index_registry.diff import versioned_chunk
from app.indexing.tokenization import tokenize


@dataclass(frozen=True)
class LexicalSyncReport:
    upserted_count: int
    invalidated_count: int
    unchanged_count: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lexical_index_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / "data/index/lexical.sqlite3"


def _parent_context(chunk: dict[str, Any]) -> str:
    values = chunk.get("parent_contexts")
    if not isinstance(values, list):
        return ""
    parts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            path = value.get("section_path")
            if isinstance(path, list):
                parts.append(" > ".join(map(str, path)))
            parts.append(str(value.get("content") or ""))
    return "\n".join(parts)


def _overlap_context(chunk: dict[str, Any]) -> str:
    value = chunk.get("overlap_context")
    return str(value.get("content") or "") if isinstance(value, dict) else ""


def _fts_text(value: str) -> str:
    """Materialize the same English/Chinese token stream used for query preprocessing."""
    return " ".join(tokenize(value))


def connect_lexical(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunk_meta (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_key TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            work_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            content_zone TEXT,
            chunk_json TEXT NOT NULL,
            valid_from_epoch INTEGER NOT NULL,
            valid_to_epoch INTEGER,
            UNIQUE(chunk_key, valid_from_epoch)
        );
        CREATE INDEX IF NOT EXISTS lexical_visibility
        ON chunk_meta(valid_from_epoch, valid_to_epoch, work_id, document_id, content_zone);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            title, section_path, parent_context, overlap_context, content,
            content='', tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    return connection


def upsert_chunk(connection: sqlite3.Connection, epoch: int, chunk: dict[str, Any]) -> bool:
    import json

    value = versioned_chunk(chunk) if "chunk_key" not in chunk else dict(chunk)
    existing = connection.execute(
        "SELECT rowid FROM chunk_meta WHERE chunk_key=? AND valid_from_epoch=?",
        (value["chunk_key"], epoch),
    ).fetchone()
    if existing:
        return False
    section = value.get("section_path")
    section_text = " > ".join(map(str, section)) if isinstance(section, list) else ""
    cursor = connection.execute(
        """INSERT INTO chunk_meta(chunk_key,chunk_id,work_id,document_id,paper_id,
        content_zone,chunk_json,valid_from_epoch) VALUES(?,?,?,?,?,?,?,?)""",
        (value["chunk_key"], value["chunk_id"], value["work_id"], value["document_id"],
         value["paper_id"], value.get("content_zone"),
         json.dumps(value, ensure_ascii=False, sort_keys=True), epoch),
    )
    connection.execute(
        "INSERT INTO chunk_fts(rowid,title,section_path,parent_context,overlap_context,content) "
        "VALUES(?,?,?,?,?,?)",
        (cursor.lastrowid, _fts_text(str(value.get("title") or "")),
         _fts_text(section_text), _fts_text(_parent_context(value)),
         _fts_text(_overlap_context(value)), _fts_text(str(value.get("content") or ""))),
    )
    return True


def invalidate_chunk(connection: sqlite3.Connection, epoch: int, chunk_key: str) -> int:
    cursor = connection.execute(
        """UPDATE chunk_meta SET valid_to_epoch=? WHERE chunk_key=?
        AND valid_to_epoch IS NULL AND valid_from_epoch < ?""", (epoch, chunk_key, epoch)
    )
    return cursor.rowcount


def sync_lexical(
    project_root: Path, epoch: int, diffs: list[Any], *, path: Path | None = None
) -> LexicalSyncReport:
    started = perf_counter()
    upserted = invalidated = unchanged = 0
    connection = connect_lexical(path or lexical_index_path(project_root))
    try:
        connection.execute("BEGIN IMMEDIATE")
        for item in diffs:
            if item.kind == "unchanged":
                unchanged += 1
                continue
            invalidated += invalidate_chunk(connection, epoch, item.chunk_key)
            if item.current is not None:
                upserted += int(upsert_chunk(connection, epoch, item.current))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return LexicalSyncReport(upserted, invalidated, unchanged,
                             round((perf_counter() - started) * 1000, 3))
