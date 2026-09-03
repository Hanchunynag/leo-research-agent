"""Temporary abstract evidence for bounded external research fallback.

Abstract evidence is intentionally not a Chunk and is never written to the
formal local Chunk index.  It can support a provisional answer only within the
current run, while a separate ingestion job completes the full-text path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from app.storage import write_jsonl_atomic


def provisional_evidence_path(project_root: Path, session_id: str) -> Path:
    safe = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return (
        project_root.expanduser().resolve()
        / "data"
        / "candidate_knowledge"
        / "provisional_evidence"
        / f"{safe}.jsonl"
    )


def abstract_evidence_from_paper(
    paper: Mapping[str, Any],
    *,
    session_id: str,
    ordinal: int,
) -> dict[str, Any] | None:
    abstract = str(paper.get("abstract") or "").strip()
    title = str(paper.get("title") or "").strip()
    if not abstract or not title:
        return None
    identity = str(
        paper.get("doi")
        or paper.get("arxiv_id")
        or paper.get("paper_id")
        or title
    ).strip()
    digest = hashlib.sha256(identity.casefold().encode("utf-8")).hexdigest()[:20]
    return {
        "evidence_id": f"AE_{digest}",
        "source_id": f"A{ordinal}",
        "evidence_state": "selected",
        "evidence_grade": "abstract_based",
        "directness": "indirect",
        "directness_grade": 1,
        "source_type": "abstract_based",
        "abstract_based": True,
        "paper_id": paper.get("paper_id") or identity,
        "external_paper_key": identity,
        "title": title,
        "authors": list(paper.get("authors") or []),
        "year": paper.get("publication_year", paper.get("year")),
        "section_path": ["Abstract"],
        "page_start": None,
        "page_end": None,
        "content": abstract,
        "citation": f"{title} ({paper.get('publication_year', paper.get('year')) or 'year unavailable'}), Abstract, page unavailable",
        "provenance": {
            "session_id": session_id,
            "source_url": paper.get("landing_page_url") or paper.get("source_url"),
            "external_ids": paper.get("external_ids") or {},
        },
    }


def register_abstract_evidence(
    project_root: Path,
    session_id: str,
    papers: Iterable[Mapping[str, Any]],
    *,
    enqueue_fulltext: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    jobs: list[Any] = []
    seen: set[str] = set()
    for ordinal, paper in enumerate(papers, 1):
        value = abstract_evidence_from_paper(paper, session_id=session_id, ordinal=ordinal)
        if value is None or value["evidence_id"] in seen:
            continue
        seen.add(value["evidence_id"])
        evidence.append(value)
        if enqueue_fulltext is not None:
            job = enqueue_fulltext(dict(paper))
            jobs.append(job)
            if isinstance(job, Mapping) and job.get("job_id"):
                value["fulltext_job_id"] = str(job["job_id"])
    path = provisional_evidence_path(project_root, session_id)
    write_jsonl_atomic(path, evidence)
    try:
        from app.persistence import build_knowledge_repository

        repository = build_knowledge_repository(project_root)
        if repository is not None:
            repository.record_temporary_evidence(evidence, query_id=session_id)
            for job in jobs:
                if isinstance(job, Mapping) and job.get("job_id"):
                    repository.record_job(job)
            repository.close()
    except Exception:
        from app.persistence.mysql import MySQLConfig

        if not MySQLConfig.from_environment(project_root).fallback_to_json:
            raise
        # Abstract evidence remains available through the existing JSONL
        # projection when the optional database is offline.
        pass
    return {
        "evidence": evidence,
        "evidence_count": len(evidence),
        "fulltext_jobs": jobs,
        "path": str(path),
    }
