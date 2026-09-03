"""MySQL/SQLite structured repository with fail-safe JSON compatibility."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.engine import Connection, Engine

from app.persistence.schema import (
    citation_records,
    chunks,
    index_epochs,
    metadata,
    paper_authors,
    paper_sections,
    paper_versions,
    papers,
    query_history,
    research_jobs,
    schema_migrations,
    temporary_evidence,
)


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_file(root: Path) -> dict[str, str]:
    path = root / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


class DatabaseUnavailable(RuntimeError):
    """Raised when the configured structured database cannot be reached."""


class MySQLConfig:
    def __init__(
        self,
        *,
        url: str | None = None,
        enabled: bool = False,
        fallback_to_json: bool = True,
        echo: bool = False,
    ) -> None:
        self.url = url
        self.enabled = enabled
        self.fallback_to_json = fallback_to_json
        self.echo = echo

    @classmethod
    def from_environment(cls, project_root: Path) -> "MySQLConfig":
        file_values = _env_file(project_root.expanduser().resolve())
        def raw(name: str) -> str | None:
            return os.getenv(name, file_values.get(name))
        url = raw("LEO_MYSQL_URL") or raw("DATABASE_URL")
        return cls(
            url=url,
            enabled=_bool(raw("LEO_MYSQL_ENABLED"), bool(url)),
            fallback_to_json=_bool(raw("LEO_MYSQL_FALLBACK_TO_JSON"), True),
            echo=_bool(raw("LEO_MYSQL_ECHO"), False),
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MySQLKnowledgeRepository:
    """Repository used for facts and audit records; never stores vectors."""

    def __init__(self, project_root: Path, config: MySQLConfig | None = None) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.config = config or MySQLConfig.from_environment(self.project_root)
        self.engine: Engine | None = None
        if self.config.enabled:
            if not self.config.url:
                raise ValueError("LEO_MYSQL_ENABLED=true 但未配置 LEO_MYSQL_URL。")
            try:
                self.engine = create_engine(
                    self.config.url,
                    echo=self.config.echo,
                    future=True,
                    pool_pre_ping=True,
                )
            except Exception as error:
                raise DatabaseUnavailable(str(error)) from error

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    def initialize(self) -> None:
        if self.engine is None:
            return
        try:
            metadata.create_all(self.engine)
            with self.engine.begin() as connection:
                if connection.execute(select(schema_migrations.c.version).where(schema_migrations.c.version == "1.0")).first() is None:
                    connection.execute(schema_migrations.insert().values(version="1.0", applied_time=_now()))
        except Exception as error:
            raise DatabaseUnavailable(str(error)) from error

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        if self.engine is None:
            raise DatabaseUnavailable("结构化数据库未启用。")
        try:
            with self.engine.begin() as connection:
                yield connection
        except Exception as error:
            raise DatabaseUnavailable(str(error)) from error

    @staticmethod
    def _upsert(connection: Connection, table: Any, key: Mapping[str, Any], values: Mapping[str, Any]) -> None:
        where = [table.c[name] == value for name, value in key.items()]
        if connection.execute(select(table).where(*where)).first() is None:
            connection.execute(table.insert().values(**values))
        else:
            connection.execute(update(table).where(*where).values(**dict(values)))

    def upsert_paper(self, value: Mapping[str, Any]) -> None:
        paper_id = str(value.get("paper_id") or "").strip()
        title = str(value.get("title") or "").strip()
        if not paper_id or not title:
            raise ValueError("paper_id/title 不能为空。")
        now = _now()
        row = {
            "paper_id": paper_id,
            "document_id": str(value.get("document_id") or "") or None,
            "work_id": str(value.get("work_id") or "") or None,
            "title": title,
            "authors_json": _json(list(value.get("authors") or [])),
            "year": int(value["year"]) if isinstance(value.get("year"), int) else None,
            "abstract": str(value.get("abstract") or ""),
            "keywords_json": _json(list(value.get("keywords") or [])),
            "doi": str(value.get("doi") or "") or None,
            "citation_count": int(value["citation_count"]) if isinstance(value.get("citation_count"), int) else None,
            "pdf_path": str(value.get("pdf_path") or "") or None,
            "source_sha256": str(value.get("source_sha256") or "") or None,
            "status": str(value.get("status") or "indexed"),
            "metadata_json": _json(dict(value.get("metadata") or {})),
            "index_epoch": str(value.get("index_epoch") or "") or None,
            "created_time": now,
            "updated_time": now,
        }
        with self._connection() as connection:
            existing = connection.execute(select(papers.c.created_time).where(papers.c.paper_id == paper_id)).first()
            if existing is not None:
                row["created_time"] = existing[0]
            self._upsert(connection, papers, {"paper_id": paper_id}, row)
            connection.execute(delete(paper_authors).where(paper_authors.c.paper_id == paper_id))
            authors = [str(item).strip() for item in value.get("authors") or [] if str(item).strip()]
            if authors:
                connection.execute(paper_authors.insert(), [{"paper_id": paper_id, "author_order": i, "author_name": name} for i, name in enumerate(authors)])

    def upsert_papers(self, values: Sequence[Mapping[str, Any]]) -> int:
        for value in values:
            self.upsert_paper(value)
        return len(values)

    def upsert_version(self, value: Mapping[str, Any]) -> None:
        document_id = str(value.get("document_id") or "").strip()
        paper_id = str(value.get("paper_id") or "").strip()
        if not document_id or not paper_id:
            return
        now = _now()
        row = {
            "document_id": document_id,
            "paper_id": paper_id,
            "source_sha256": str(value.get("source_sha256") or "") or None,
            "canonical_path": str(value.get("canonical_path") or "") or None,
            "pdf_path": str(value.get("pdf_path") or "") or None,
            "parse_status": str(value.get("parse_status") or "parsed"),
            "metadata_json": _json(dict(value.get("metadata") or {})),
            "created_time": now,
            "updated_time": now,
        }
        with self._connection() as connection:
            existing = connection.execute(select(paper_versions.c.created_time).where(paper_versions.c.document_id == document_id)).first()
            if existing is not None:
                row["created_time"] = existing[0]
            self._upsert(connection, paper_versions, {"document_id": document_id}, row)

    def upsert_sections(self, values: Sequence[Mapping[str, Any]]) -> int:
        if not values:
            return 0
        now = _now()
        with self._connection() as connection:
            for value in values:
                section_id = str(value.get("section_id") or "").strip()
                if not section_id:
                    continue
                row = {
                    "section_id": section_id,
                    "paper_id": str(value.get("paper_id") or ""),
                    "document_id": str(value.get("document_id") or ""),
                    "parent_section_id": str(value.get("parent_section_id") or "") or None,
                    "title": str(value.get("title") or ""),
                    "level": int(value.get("level") or 0),
                    "section_path_json": _json(value.get("section_path") or []),
                    "content_zone": str(value.get("content_zone") or "") or None,
                    "page_start": int(value["page_start"]) if isinstance(value.get("page_start"), int) else None,
                    "page_end": int(value["page_end"]) if isinstance(value.get("page_end"), int) else None,
                    "metadata_json": _json(dict(value.get("metadata") or {})),
                    "created_time": now,
                    "updated_time": now,
                }
                self._upsert(connection, paper_sections, {"section_id": section_id}, row)
        return len(values)

    def upsert_chunks(self, values: Sequence[Mapping[str, Any]]) -> int:
        if not values:
            return 0
        now = _now()
        with self._connection() as connection:
            for value in values:
                chunk_id = str(value.get("chunk_id") or "").strip()
                content = str(value.get("content") or "").strip()
                if not chunk_id or not content:
                    continue
                row = {
                    "chunk_id": chunk_id,
                    "paper_id": str(value.get("paper_id") or ""),
                    "document_id": str(value.get("document_id") or "") or None,
                    "section_id": str(value.get("section_id") or "") or None,
                    "section": str(value.get("section") or value.get("section_title") or "") or None,
                    "section_path_json": _json(value.get("section_path") or []),
                    "page_start": int(value["page_start"]) if isinstance(value.get("page_start"), int) else None,
                    "page_end": int(value["page_end"]) if isinstance(value.get("page_end"), int) else None,
                    "content": content,
                    "token_count": int(value.get("token_count") or 0),
                    "parent_context_json": _json(value.get("parent_contexts") or []),
                    "overlap_context_json": _json(value.get("overlap_context") or {}),
                    "metadata_json": _json(dict(value.get("metadata") or {})),
                    "content_hash": str(value.get("content_hash") or "") or None,
                    "status": str(value.get("status") or "indexed"),
                    "created_time": now,
                    "updated_time": now,
                }
                self._upsert(connection, chunks, {"chunk_id": chunk_id}, row)
        return len(values)

    def mark_missing_local_records(
        self,
        *,
        paper_ids: Sequence[str],
        chunk_ids: Sequence[str],
    ) -> None:
        """Tombstone local facts removed from the Canonical projection."""
        current_papers = [str(value) for value in paper_ids if str(value)]
        current_chunks = [str(value) for value in chunk_ids if str(value)]
        with self._connection() as connection:
            paper_statement = update(papers).values(status="deleted", updated_time=_now())
            paper_statement = (
                paper_statement.where(papers.c.paper_id.not_in(current_papers))
                if current_papers
                else paper_statement
            )
            connection.execute(paper_statement)
            chunk_statement = update(chunks).values(status="deleted", updated_time=_now())
            chunk_statement = (
                chunk_statement.where(chunks.c.chunk_id.not_in(current_chunks))
                if current_chunks
                else chunk_statement
            )
            connection.execute(chunk_statement)

    @staticmethod
    def _paper(row: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(row)
        value.pop("created_time", None)
        value.pop("updated_time", None)
        value["authors"] = _decode(value.pop("authors_json", None), [])
        value["keywords"] = _decode(value.pop("keywords_json", None), [])
        value["metadata"] = _decode(value.pop("metadata_json", None), {})
        return value

    @staticmethod
    def _chunk(row: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(row)
        value.pop("id", None)
        value.pop("created_time", None)
        value.pop("updated_time", None)
        value["section_path"] = _decode(value.pop("section_path_json", None), [])
        value["parent_contexts"] = _decode(value.pop("parent_context_json", None), [])
        value["overlap_context"] = _decode(value.pop("overlap_context_json", None), None)
        value["metadata"] = _decode(value.pop("metadata_json", None), {})
        return value

    def list_papers(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(select(papers).where(papers.c.status.not_in(("deleted", "superseded"))).order_by(papers.c.paper_id)).mappings().all()
        return [self._paper(cast(Mapping[str, Any], row)) for row in rows]

    def list_chunks(self, paper_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        statement = select(chunks).where(chunks.c.status != "deleted")
        ids = [str(item) for item in (paper_ids or []) if str(item)]
        if ids:
            statement = statement.where(chunks.c.paper_id.in_(ids))
        statement = statement.order_by(chunks.c.chunk_id)
        with self._connection() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._chunk(cast(Mapping[str, Any], row)) for row in rows]

    def record_query(self, value: Mapping[str, Any]) -> str:
        query_id = str(value.get("id") or value.get("run_id") or "").strip()
        if not query_id:
            raise ValueError("query_history id/run_id 不能为空。")
        now = _now()
        row = {
            "id": query_id,
            "session_id": str(value.get("session_id") or "") or None,
            "workspace_id": str(value.get("workspace_id") or "") or None,
            "scope_version": int(value["scope_version"]) if isinstance(value.get("scope_version"), int) else None,
            "user_query": str(value.get("user_query") or value.get("query") or ""),
            "planner_result_json": _json(value.get("planner_result") or {}),
            "retrieved_papers_json": _json(value.get("retrieved_papers") or []),
            "retrieved_chunks_json": _json(value.get("retrieved_chunks") or []),
            "selected_evidence_json": _json(value.get("selected_evidence") or []),
            "coverage_report_json": _json(value.get("coverage_report") or value.get("coverage") or {}),
            "final_answer": str(value.get("final_answer") or "") or None,
            "status": str(value.get("status") or "completed"),
            "created_time": now,
            "updated_time": now,
        }
        with self._connection() as connection:
            self._upsert(connection, query_history, {"id": query_id}, row)
        return query_id

    def get_query(self, query_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                select(query_history).where(query_history.c.id == query_id)
            ).mappings().first()
        if row is None:
            return None
        value = dict(row)
        for field_name, default in (
            ("planner_result_json", {}),
            ("retrieved_papers_json", []),
            ("retrieved_chunks_json", []),
            ("selected_evidence_json", []),
            ("coverage_report_json", {}),
        ):
            value[field_name.removesuffix("_json")] = _decode(value.pop(field_name), default)
        value.pop("created_time", None)
        value.pop("updated_time", None)
        return value

    def list_queries(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or limit < 1 or limit > 1000:
            raise ValueError("query_history limit 必须在 1 到 1000 之间。")
        with self._connection() as connection:
            rows = connection.execute(
                select(query_history)
                .order_by(query_history.c.created_time.desc())
                .limit(limit)
            ).mappings().all()
        return [value for row in rows if (value := self.get_query(str(row["id"]))) is not None]

    def record_citations(self, query_id: str | None, citations: Sequence[Mapping[str, Any]]) -> int:
        if not citations:
            return 0
        with self._connection() as connection:
            for value in citations:
                connection.execute(citation_records.insert().values(
                    query_history_id=query_id,
                    claim_id=str(value.get("claim_id") or "") or None,
                    claim=str(value.get("claim") or value.get("text") or ""),
                    chunk_id=str(value.get("chunk_id") or "") or None,
                    paper_id=str(value.get("paper_id") or "") or None,
                    support_score=float(value["support_score"]) if isinstance(value.get("support_score"), (int, float)) else None,
                    verified=bool(value.get("verified")),
                    validation_json=_json(dict(value.get("validation") or {})),
                    created_time=_now(),
                ))
        return len(citations)

    def record_temporary_evidence(self, values: Sequence[Mapping[str, Any]], *, query_id: str | None = None) -> int:
        if not values:
            return 0
        now = _now()
        with self._connection() as connection:
            for value in values:
                evidence_id = str(value.get("evidence_id") or "").strip()
                if not evidence_id:
                    continue
                row = {
                    "evidence_id": evidence_id,
                    "query_history_id": query_id,
                    "session_id": str((value.get("provenance") or {}).get("session_id") or "") or None,
                    "external_paper_key": str(value.get("external_paper_key") or value.get("paper_id") or evidence_id),
                    "title": str(value.get("title") or ""),
                    "abstract": str(value.get("content") or value.get("abstract") or ""),
                    "year": int(value["year"]) if isinstance(value.get("year"), int) else None,
                    "source_url": str((value.get("provenance") or {}).get("source_url") or "") or None,
                    "abstract_only": bool(value.get("abstract_only", True)),
                    "metadata_json": _json(dict(value.get("provenance") or {})),
                    "status": "provisional",
                    "fulltext_job_id": str(value.get("fulltext_job_id") or "") or None,
                    "created_time": now,
                    "updated_time": now,
                }
                self._upsert(connection, temporary_evidence, {"evidence_id": evidence_id}, row)
        return len(values)

    def record_job(self, value: Mapping[str, Any]) -> str:
        job_id = str(value.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("research_jobs job_id 不能为空。")
        now = _now()
        row = {
            "job_id": job_id,
            "job_type": str(value.get("job_type") or "fulltext_ingestion"),
            "paper_id": str(value.get("paper_id") or "") or None,
            "document_id": str(value.get("document_id") or "") or None,
            "status": str(value.get("status") or "queued"),
            "payload_json": _json(dict(value.get("payload") or {})),
            "result_reference": str(value.get("result_reference") or "") or None,
            "error_summary": str(value.get("error_summary") or "") or None,
            "created_time": now,
            "updated_time": now,
        }
        with self._connection() as connection:
            self._upsert(connection, research_jobs, {"job_id": job_id}, row)
        return job_id

    def record_index_epoch(self, value: Mapping[str, Any]) -> str:
        epoch_id = str(value.get("epoch_id") or "").strip()
        if not epoch_id:
            raise ValueError("index_epochs epoch_id 不能为空。")
        row = {
            "epoch_id": epoch_id,
            "index_kind": str(value.get("index_kind") or "knowledge"),
            "source_fingerprint": str(value.get("source_fingerprint") or ""),
            "embedding_model": str(value.get("embedding_model") or "") or None,
            "embedding_revision": str(value.get("embedding_revision") or "") or None,
            "tokenizer_version": str(value.get("tokenizer_version") or "") or None,
            "chunker_version": str(value.get("chunker_version") or "") or None,
            "status": str(value.get("status") or "active"),
            "created_time": _now(),
            "activated_time": _now() if value.get("status", "active") == "active" else None,
        }
        with self._connection() as connection:
            self._upsert(connection, index_epochs, {"epoch_id": epoch_id}, row)
        return epoch_id

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()


def build_knowledge_repository(project_root: Path, *, initialize: bool = True) -> MySQLKnowledgeRepository | None:
    """Return a configured repository, or None when MySQL is intentionally off."""
    repository = MySQLKnowledgeRepository(project_root)
    if not repository.enabled:
        return None
    if initialize:
        repository.initialize()
    return repository
