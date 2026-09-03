"""Portable SQLAlchemy schema for the structured research knowledge layer."""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)


metadata = MetaData()

schema_migrations = Table(
    "schema_migrations", metadata,
    Column("version", String(32), primary_key=True),
    Column("applied_time", DateTime, nullable=False),
)

papers = Table(
    "papers", metadata,
    Column("paper_id", String(191), primary_key=True),
    Column("document_id", String(191), unique=True, nullable=True),
    Column("work_id", String(191), nullable=True),
    Column("title", Text, nullable=False),
    Column("authors_json", Text, nullable=False),
    Column("year", Integer, nullable=True),
    Column("abstract", Text, nullable=False),
    Column("keywords_json", Text, nullable=False),
    Column("doi", String(512), nullable=True),
    Column("citation_count", Integer, nullable=True),
    Column("pdf_path", Text, nullable=True),
    Column("source_sha256", String(128), nullable=True),
    Column("status", String(32), nullable=False, default="indexed"),
    Column("metadata_json", Text, nullable=False),
    Column("index_epoch", String(191), nullable=True),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_papers_year", papers.c.year)
Index("ix_papers_work_id", papers.c.work_id)
Index("ix_papers_status", papers.c.status)

paper_authors = Table(
    "paper_authors", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("paper_id", String(191), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False),
    Column("author_order", Integer, nullable=False),
    Column("author_name", String(512), nullable=False),
)
Index("uq_paper_author_order", paper_authors.c.paper_id, paper_authors.c.author_order, unique=True)

paper_versions = Table(
    "paper_versions", metadata,
    Column("document_id", String(191), primary_key=True),
    Column("paper_id", String(191), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False),
    Column("source_sha256", String(128), nullable=True),
    Column("canonical_path", Text, nullable=True),
    Column("pdf_path", Text, nullable=True),
    Column("parse_status", String(32), nullable=False, default="parsed"),
    Column("metadata_json", Text, nullable=False),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_paper_versions_paper_id", paper_versions.c.paper_id)

paper_sections = Table(
    "paper_sections", metadata,
    Column("section_id", String(191), primary_key=True),
    Column("paper_id", String(191), nullable=False),
    Column("document_id", String(191), nullable=False),
    Column("parent_section_id", String(191), nullable=True),
    Column("title", Text, nullable=False),
    Column("level", Integer, nullable=False),
    Column("section_path_json", Text, nullable=False),
    Column("content_zone", String(64), nullable=True),
    Column("page_start", Integer, nullable=True),
    Column("page_end", Integer, nullable=True),
    Column("metadata_json", Text, nullable=False),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_paper_sections_document", paper_sections.c.document_id)
Index("ix_paper_sections_paper", paper_sections.c.paper_id)

chunks = Table(
    "chunks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chunk_id", String(191), unique=True, nullable=False),
    Column("paper_id", String(191), nullable=False),
    Column("document_id", String(191), nullable=True),
    Column("section_id", String(191), nullable=True),
    Column("section", Text, nullable=True),
    Column("section_path_json", Text, nullable=False),
    Column("page_start", Integer, nullable=True),
    Column("page_end", Integer, nullable=True),
    Column("content", Text, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("parent_context_json", Text, nullable=False),
    Column("overlap_context_json", Text, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("content_hash", String(128), nullable=True),
    Column("status", String(32), nullable=False, default="indexed"),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_chunks_paper_id", chunks.c.paper_id)
Index("ix_chunks_document_id", chunks.c.document_id)
Index("ix_chunks_section_id", chunks.c.section_id)

query_history = Table(
    "query_history", metadata,
    Column("id", String(191), primary_key=True),
    Column("session_id", String(191), nullable=True),
    Column("workspace_id", String(191), nullable=True),
    Column("scope_version", Integer, nullable=True),
    Column("user_query", Text, nullable=False),
    Column("planner_result_json", Text, nullable=False),
    Column("retrieved_papers_json", Text, nullable=False),
    Column("retrieved_chunks_json", Text, nullable=False),
    Column("selected_evidence_json", Text, nullable=False),
    Column("coverage_report_json", Text, nullable=False),
    Column("final_answer", Text, nullable=True),
    Column("status", String(32), nullable=False, default="started"),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_query_history_created", query_history.c.created_time)
Index("ix_query_history_session", query_history.c.session_id)

citation_records = Table(
    "citation_records", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_history_id", String(191), nullable=True),
    Column("claim_id", String(191), nullable=True),
    Column("claim", Text, nullable=False),
    Column("chunk_id", String(191), nullable=True),
    Column("paper_id", String(191), nullable=True),
    Column("support_score", Float, nullable=True),
    Column("verified", Boolean, nullable=False),
    Column("validation_json", Text, nullable=False),
    Column("created_time", DateTime, nullable=False),
)
Index("ix_citation_records_query", citation_records.c.query_history_id)
Index("ix_citation_records_chunk", citation_records.c.chunk_id)

temporary_evidence = Table(
    "temporary_evidence", metadata,
    Column("evidence_id", String(191), primary_key=True),
    Column("query_history_id", String(191), nullable=True),
    Column("session_id", String(191), nullable=True),
    Column("external_paper_key", String(512), nullable=False),
    Column("title", Text, nullable=False),
    Column("abstract", Text, nullable=False),
    Column("year", Integer, nullable=True),
    Column("source_url", Text, nullable=True),
    Column("abstract_only", Boolean, nullable=False, default=True),
    Column("metadata_json", Text, nullable=False),
    Column("status", String(32), nullable=False, default="provisional"),
    Column("fulltext_job_id", String(191), nullable=True),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_temporary_evidence_query", temporary_evidence.c.query_history_id)
Index("ix_temporary_evidence_job", temporary_evidence.c.fulltext_job_id)

research_jobs = Table(
    "research_jobs", metadata,
    Column("job_id", String(191), primary_key=True),
    Column("job_type", String(64), nullable=False),
    Column("paper_id", String(191), nullable=True),
    Column("document_id", String(191), nullable=True),
    Column("status", String(32), nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("result_reference", Text, nullable=True),
    Column("error_summary", Text, nullable=True),
    Column("created_time", DateTime, nullable=False),
    Column("updated_time", DateTime, nullable=False),
)
Index("ix_research_jobs_status", research_jobs.c.status)

index_epochs = Table(
    "index_epochs", metadata,
    Column("epoch_id", String(191), primary_key=True),
    Column("index_kind", String(64), nullable=False),
    Column("source_fingerprint", String(128), nullable=False),
    Column("embedding_model", String(512), nullable=True),
    Column("embedding_revision", String(512), nullable=True),
    Column("tokenizer_version", String(128), nullable=True),
    Column("chunker_version", String(128), nullable=True),
    Column("status", String(32), nullable=False),
    Column("created_time", DateTime, nullable=False),
    Column("activated_time", DateTime, nullable=True),
)
Index("ix_index_epochs_kind_status", index_epochs.c.index_kind, index_epochs.c.status)


TABLES = {
    table.name: table
    for table in metadata.sorted_tables
}
