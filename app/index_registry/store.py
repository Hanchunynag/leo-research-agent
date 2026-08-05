"""Transactional registry store and active-epoch visibility queries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.index_registry.models import IndexOperation
from app.index_registry.schema import initialize_schema


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexRegistryStore:
    def __init__(self, project_root: Path, database_path: Path | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.path = (database_path or root / "data/index/index_registry.sqlite3").expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            initialize_schema(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_epoch(self, **metadata: Any) -> int:
        required = {
            "embedding_model": "BAAI/bge-m3", "embedding_revision": None,
            "dense_text_policy_version": "1.0", "chunk_policy_version": "2.1",
            "ontology_version": "1.0", "extractor_model": None,
            "extractor_prompt_version": "1.0", "community_prompt_version": "1.0",
        }
        required.update(metadata)
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO index_epochs(status,started_at,embedding_model,
                embedding_revision,dense_text_policy_version,chunk_policy_version,
                ontology_version,extractor_model,extractor_prompt_version,
                community_prompt_version) VALUES('pending',?,?,?,?,?,?,?,?,?)""",
                (utc_now(), required["embedding_model"], required["embedding_revision"],
                 required["dense_text_policy_version"], required["chunk_policy_version"],
                 required["ontology_version"], required["extractor_model"],
                 required["extractor_prompt_version"], required["community_prompt_version"]),
            )
            return int(cursor.lastrowid)

    def active_epoch(self) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT epoch FROM index_epochs WHERE status='active'"
            ).fetchone()
        return int(row[0]) if row else None

    def epoch_record(self, epoch: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM index_epochs WHERE epoch=?", (epoch,)).fetchone()
        if row is None:
            raise KeyError(epoch)
        return dict(row)

    def activate_epoch(self, epoch: int) -> None:
        with self.transaction() as connection:
            pending = connection.execute(
                "SELECT status FROM index_epochs WHERE epoch=?", (epoch,)
            ).fetchone()
            if pending is None or pending[0] != "pending":
                raise ValueError(f"epoch {epoch} is not pending")
            incomplete = connection.execute(
                "SELECT COUNT(*) FROM index_operations WHERE epoch=? AND status!='completed'",
                (epoch,),
            ).fetchone()[0]
            if incomplete:
                raise RuntimeError(f"epoch {epoch} has {incomplete} incomplete operations")
            connection.execute(
                "UPDATE index_epochs SET status='superseded' WHERE status='active'"
            )
            connection.execute(
                "UPDATE index_epochs SET status='active',completed_at=? WHERE epoch=?",
                (utc_now(), epoch),
            )

    def fail_epoch(self, epoch: int, error_type: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE index_epochs SET status='failed',completed_at=?,error_type=? "
                "WHERE epoch=? AND status='pending'", (utc_now(), error_type, epoch)
            )

    def visible_chunks(self, epoch: int | None = None) -> dict[str, dict[str, Any]]:
        visible_epoch = epoch if epoch is not None else self.active_epoch()
        if visible_epoch is None:
            return {}
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM chunk_versions WHERE valid_from_epoch <= ?
                AND (valid_to_epoch IS NULL OR ? < valid_to_epoch)""",
                (visible_epoch, visible_epoch),
            ).fetchall()
        return {str(row["chunk_key"]): {**dict(row), **json.loads(row["chunk_json"])} for row in rows}

    def put_chunk_version(self, epoch: int, chunk: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO chunk_versions(chunk_key,chunk_id,document_id,
                work_id,paper_id,content_hash,dense_text_hash,graph_text_hash,chunk_json,
                valid_from_epoch,valid_to_epoch,active) VALUES(?,?,?,?,?,?,?,?,?,?,NULL,1)""",
                (chunk["chunk_key"], chunk["chunk_id"], chunk["document_id"],
                 chunk["work_id"], chunk["paper_id"], chunk["content_hash"],
                 chunk["dense_text_hash"], chunk["graph_text_hash"],
                 json.dumps(chunk, ensure_ascii=False, sort_keys=True), epoch),
            )

    def put_embedding_entry(self, *, epoch: int, chunk_key: str, point_id: str,
                            collection_name: str, model_name: str,
                            model_revision: str | None, dense_text_hash: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO embedding_entries(chunk_key,point_id,
                collection_name,model_name,model_revision,dense_text_hash,valid_from_epoch)
                VALUES(?,?,?,?,?,?,?)""",
                (chunk_key, point_id, collection_name, model_name, model_revision,
                 dense_text_hash, epoch),
            )

    def invalidate_embedding(self, *, epoch: int, chunk_key: str) -> int:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE embedding_entries SET valid_to_epoch=? WHERE chunk_key=?
                AND valid_to_epoch IS NULL AND valid_from_epoch < ?""",
                (epoch, chunk_key, epoch),
            ).rowcount

    def invalidate_chunk(self, epoch: int, chunk_key: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE chunk_versions SET valid_to_epoch=?,active=0
                WHERE chunk_key=? AND valid_to_epoch IS NULL AND valid_from_epoch < ?""",
                (epoch, chunk_key, epoch),
            )

    def enqueue(self, operation: IndexOperation) -> None:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO index_operations(operation_id,epoch,store,
                operation_type,object_id,target_hash,payload_json,status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,'pending',?,?)""",
                (operation.operation_id, operation.epoch, operation.store,
                 operation.operation_type, operation.object_id, operation.target_hash,
                 json.dumps(operation.payload, ensure_ascii=False, sort_keys=True), now, now),
            )

    def list_operations(self, epoch: int, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM index_operations WHERE epoch=?"
        params: list[Any] = [epoch]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at,operation_id"
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def mark_operation(self, operation_id: str, status: str, error_type: str | None = None) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE index_operations SET status=?, attempts=attempts+1,
                last_error_type=?,updated_at=? WHERE operation_id=?""",
                (status, error_type, utc_now(), operation_id),
            )

    def operation_completed(self, operation_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM index_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        return bool(row and row[0] == "completed")

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            epochs = [dict(row) for row in connection.execute(
                "SELECT * FROM index_epochs ORDER BY epoch DESC"
            ).fetchall()]
            counts = {row[0]: row[1] for row in connection.execute(
                "SELECT status,COUNT(*) FROM index_operations GROUP BY status"
            ).fetchall()}
        return {"path": str(self.path), "active_epoch": self.active_epoch(),
                "epochs": epochs, "operation_counts": counts}

    def get_graph_extraction(self, cache_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT extraction_json FROM graph_extractions WHERE cache_key=? AND status='completed'",
                (cache_key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_community_report(self, fingerprint: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT report_json FROM community_versions
                WHERE fingerprint=? AND report_json IS NOT NULL
                ORDER BY valid_from_epoch DESC LIMIT 1""", (fingerprint,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_community_version(self, *, epoch: int, community_id: str,
                              fingerprint: str, report: dict[str, Any]) -> None:
        report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
        report_hash = __import__("hashlib").sha256(report_json.encode()).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO community_versions(community_id,fingerprint,
                report_hash,report_json,valid_from_epoch) VALUES(?,?,?,?,?)""",
                (community_id, fingerprint, report_hash, report_json, epoch),
            )

    def put_graph_extraction(
        self, *, cache_key: str, chunk_key: str, graph_text_hash: str,
        extractor_model: str, extractor_prompt_version: str, ontology_version: str,
        extraction: dict[str, Any], status: str = "completed",
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO graph_extractions(cache_key,chunk_key,graph_text_hash,
                extractor_model,extractor_prompt_version,ontology_version,extraction_json,
                status,created_at) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET extraction_json=excluded.extraction_json,
                status=excluded.status""",
                (cache_key, chunk_key, graph_text_hash, extractor_model,
                 extractor_prompt_version, ontology_version,
                 json.dumps(extraction, ensure_ascii=False, sort_keys=True), status, utc_now()),
            )

    def retry_failed(self, epoch: int | None = None) -> list[int]:
        with self.transaction() as connection:
            if epoch is None:
                rows = connection.execute(
                    "SELECT epoch FROM index_epochs WHERE status='failed' ORDER BY epoch"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT epoch FROM index_epochs WHERE epoch=? AND status='failed'", (epoch,)
                ).fetchall()
            epochs = [int(row[0]) for row in rows]
            for value in epochs:
                connection.execute(
                    "UPDATE index_epochs SET status='pending',completed_at=NULL,error_type=NULL WHERE epoch=?",
                    (value,),
                )
                connection.execute(
                    "UPDATE index_operations SET status='pending',last_error_type=NULL WHERE epoch=? AND status='failed'",
                    (value,),
                )
        return epochs

    def cleanup_epochs(self, *, keep_failed: int = 2) -> dict[str, int]:
        with self.transaction() as connection:
            failed = [int(row[0]) for row in connection.execute(
                "SELECT epoch FROM index_epochs WHERE status='failed' ORDER BY epoch DESC"
            ).fetchall()]
            remove = failed[max(0, keep_failed):]
            operations = chunks = epochs = 0
            for epoch in remove:
                operations += connection.execute(
                    "DELETE FROM index_operations WHERE epoch=?", (epoch,)
                ).rowcount
                chunks += connection.execute(
                    "DELETE FROM chunk_versions WHERE valid_from_epoch=?", (epoch,)
                ).rowcount
                connection.execute("DELETE FROM embedding_entries WHERE valid_from_epoch=?", (epoch,))
                connection.execute("DELETE FROM community_versions WHERE valid_from_epoch=?", (epoch,))
                connection.execute("DELETE FROM documents WHERE valid_from_epoch=?", (epoch,))
                epochs += connection.execute("DELETE FROM index_epochs WHERE epoch=?", (epoch,)).rowcount
        return {"epochs_removed": epochs, "operations_removed": operations,
                "chunk_versions_removed": chunks}
