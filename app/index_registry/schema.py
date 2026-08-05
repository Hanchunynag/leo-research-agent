"""SQLite schema for index state; deliberately separate from agent sessions."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS registry_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_epochs (
    epoch INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL CHECK(status IN ('pending','active','failed','superseded')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT,
    dense_text_policy_version TEXT NOT NULL,
    chunk_policy_version TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    extractor_model TEXT,
    extractor_prompt_version TEXT NOT NULL,
    community_prompt_version TEXT NOT NULL,
    error_type TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_epoch
ON index_epochs(status) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT NOT NULL,
    work_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    valid_from_epoch INTEGER NOT NULL REFERENCES index_epochs(epoch),
    valid_to_epoch INTEGER REFERENCES index_epochs(epoch),
    active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(document_id, valid_from_epoch)
);

CREATE TABLE IF NOT EXISTS chunk_versions (
    chunk_key TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    work_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dense_text_hash TEXT NOT NULL,
    graph_text_hash TEXT NOT NULL,
    chunk_json TEXT NOT NULL,
    valid_from_epoch INTEGER NOT NULL REFERENCES index_epochs(epoch),
    valid_to_epoch INTEGER REFERENCES index_epochs(epoch),
    active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(chunk_key, valid_from_epoch)
);

CREATE INDEX IF NOT EXISTS chunk_versions_visibility
ON chunk_versions(chunk_key, valid_from_epoch, valid_to_epoch);
CREATE INDEX IF NOT EXISTS chunk_versions_document
ON chunk_versions(document_id, valid_from_epoch, valid_to_epoch);

CREATE TABLE IF NOT EXISTS embedding_entries (
    chunk_key TEXT NOT NULL,
    point_id TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT,
    dense_text_hash TEXT NOT NULL,
    valid_from_epoch INTEGER NOT NULL REFERENCES index_epochs(epoch),
    valid_to_epoch INTEGER REFERENCES index_epochs(epoch),
    PRIMARY KEY(point_id, collection_name)
);

CREATE TABLE IF NOT EXISTS graph_extractions (
    cache_key TEXT PRIMARY KEY,
    chunk_key TEXT NOT NULL,
    graph_text_hash TEXT NOT NULL,
    extractor_model TEXT NOT NULL,
    extractor_prompt_version TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS graph_extractions_lookup
ON graph_extractions(chunk_key, graph_text_hash, extractor_model,
                     extractor_prompt_version, ontology_version);

CREATE TABLE IF NOT EXISTS community_versions (
    community_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    report_hash TEXT,
    report_json TEXT,
    valid_from_epoch INTEGER NOT NULL REFERENCES index_epochs(epoch),
    valid_to_epoch INTEGER REFERENCES index_epochs(epoch),
    PRIMARY KEY(community_id, valid_from_epoch)
);

CREATE TABLE IF NOT EXISTS index_operations (
    operation_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL REFERENCES index_epochs(epoch),
    store TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS index_operations_epoch_status
ON index_operations(epoch, status, store);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_normalized TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_epoch INTEGER NOT NULL,
    PRIMARY KEY(alias_normalized, entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS entity_resolution_events (
    event_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    local_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    score REAL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_merge_events (
    merge_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    reversible INTEGER NOT NULL DEFAULT 1,
    rolled_back_at TEXT,
    created_at TEXT NOT NULL
);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO registry_meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
