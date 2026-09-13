"""Project-level Manuscript Facts 与 Contribution Registry。"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.scholar.models import (
    Contribution,
    DraftPatch,
    ManuscriptFact,
    ManuscriptState,
    ReviewIssue,
    ReviewReport,
)


class ScholarProjectStore:
    """事实与创新点的唯一项目级持久化入口。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.directory = self.project_root / ".scholar"
        self.database_path = self.directory / "project.db"
        self._project_id: str | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS manuscript_facts (
                    fact_id TEXT PRIMARY KEY,
                    fact_key TEXT NOT NULL UNIQUE,
                    value_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confirmed INTEGER NOT NULL,
                    source_run_id TEXT
                );
                CREATE TABLE IF NOT EXISTS contributions (
                    contribution_id TEXT PRIMARY KEY,
                    statement TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confirmed_by_user INTEGER NOT NULL,
                    source_run_id TEXT
                );
                CREATE TABLE IF NOT EXISTS project_info (
                    project_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS patches (
                    patch_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    patch_json TEXT NOT NULL,
                    review_report_json TEXT,
                    source_session_id TEXT,
                    source_run_id TEXT,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS patch_audits (
                    audit_id TEXT PRIMARY KEY,
                    patch_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    decision_at TEXT NOT NULL,
                    expected_base_hash TEXT,
                    previous_hash TEXT,
                    new_hash TEXT,
                    build_status TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS manuscript_state (
                    project_id TEXT PRIMARY KEY,
                    project_hash TEXT NOT NULL,
                    root_tex TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stale_sections_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS manuscript_sections (
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stale INTEGER NOT NULL,
                    PRIMARY KEY(project_id, name)
                );
                CREATE TABLE IF NOT EXISTS build_results (
                    build_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    patch_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    diagnostics_json TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS citation_registry (
                    project_id TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    bibkey TEXT,
                    evidence_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL DEFAULT '',
                    source_type TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, identity_key),
                    UNIQUE(project_id, binding_id)
                );
                CREATE TABLE IF NOT EXISTS external_evidence_projection (
                    project_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    canonical_id TEXT,
                    source_locator TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    publication_date TEXT,
                    retrieved_at TEXT,
                    content_hash TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, evidence_id)
                );
                """
            )
            row = connection.execute(
                "SELECT project_id FROM project_info WHERE root_path=?",
                (str(self.project_root),),
            ).fetchone()
            if row is None:
                self._project_id = f"PROJECT_{secrets.token_hex(8)}"
                connection.execute(
                    "INSERT INTO project_info(project_id, root_path) VALUES (?, ?)",
                    (self._project_id, str(self.project_root)),
                )
            else:
                self._project_id = str(row["project_id"])

    @property
    def project_id(self) -> str:
        assert self._project_id is not None
        return self._project_id

    def put_fact(self, fact: ManuscriptFact) -> None:
        if fact.source != "user_confirmed" or not fact.confirmed:
            raise ValueError("V1 只允许写入用户确认的 Manuscript Fact。")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO manuscript_facts
                    (fact_id, fact_key, value_json, source, confirmed, source_run_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    fact_key=excluded.fact_key,
                    value_json=excluded.value_json,
                    source=excluded.source,
                    confirmed=excluded.confirmed,
                    source_run_id=excluded.source_run_id
                """,
                (
                    fact.fact_id,
                    fact.key,
                    json.dumps(fact.value, ensure_ascii=False, sort_keys=True),
                    fact.source,
                    int(fact.confirmed),
                    fact.source_run_id,
                ),
            )

    def list_facts(self) -> list[ManuscriptFact]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM manuscript_facts ORDER BY fact_key"
            ).fetchall()
        return [
            ManuscriptFact(
                fact_id=str(row["fact_id"]),
                key=str(row["fact_key"]),
                value=json.loads(row["value_json"]),
                source=str(row["source"]),
                confirmed=bool(row["confirmed"]),
                source_run_id=row["source_run_id"],
            )
            for row in rows
        ]

    def put_contribution(self, contribution: Contribution) -> None:
        if contribution.status == "confirmed" and not contribution.confirmed_by_user:
            raise ValueError("LLM 或普通 Agent 不能直接确认 Contribution。")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO contributions
                    (contribution_id, statement, status, confirmed_by_user, source_run_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(contribution_id) DO UPDATE SET
                    statement=excluded.statement,
                    status=excluded.status,
                    confirmed_by_user=excluded.confirmed_by_user,
                    source_run_id=excluded.source_run_id
                """,
                (
                    contribution.contribution_id,
                    contribution.statement,
                    contribution.status,
                    int(contribution.confirmed_by_user),
                    contribution.source_run_id,
                ),
            )

    def confirm_contribution(self, contribution_id: str) -> Contribution:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE contributions
                SET status='confirmed', confirmed_by_user=1
                WHERE contribution_id=?
                """,
                (contribution_id,),
            ).rowcount
        if updated != 1:
            raise KeyError(f"Contribution 不存在：{contribution_id}")
        return self.get_contribution(contribution_id)

    def get_contribution(self, contribution_id: str) -> Contribution:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM contributions WHERE contribution_id=?",
                (contribution_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Contribution 不存在：{contribution_id}")
        return Contribution(
            contribution_id=str(row["contribution_id"]),
            statement=str(row["statement"]),
            status=str(row["status"]),
            confirmed_by_user=bool(row["confirmed_by_user"]),
            source_run_id=row["source_run_id"],
        )

    def list_contributions(self) -> list[Contribution]:
        """只读列出 Project Registry；Writing Vertical 只能消费 confirmed 项。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM contributions ORDER BY contribution_id"
            ).fetchall()
        return [
            Contribution(
                contribution_id=str(row["contribution_id"]),
                statement=str(row["statement"]),
                status=str(row["status"]),
                confirmed_by_user=bool(row["confirmed_by_user"]),
                source_run_id=row["source_run_id"],
            )
            for row in rows
        ]

    def save_citation_binding(self, binding: object) -> None:
        """Persist only the Citation Registry projection, never .bib content."""

        from app.scholar.citation.models import CitationBinding

        if not isinstance(binding, CitationBinding):
            raise TypeError("citation binding 类型无效。")
        if binding.project_id != self.project_id:
            raise ValueError("PROJECT_CONFLICT: CitationBinding 不属于当前 Project。")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO citation_registry(
                    project_id, identity_key, binding_id, identity_json, bibkey,
                    evidence_ids_json, status, metadata_hash, source_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, identity_key) DO UPDATE SET
                    binding_id=excluded.binding_id,
                    identity_json=excluded.identity_json,
                    bibkey=excluded.bibkey,
                    evidence_ids_json=excluded.evidence_ids_json,
                    status=excluded.status,
                    metadata_hash=excluded.metadata_hash,
                    source_type=excluded.source_type,
                    updated_at=excluded.updated_at
                """,
                (
                    self.project_id,
                    binding.identity_key,
                    binding.binding_id,
                    json.dumps(binding.citation_identity.to_dict(), ensure_ascii=False, sort_keys=True),
                    binding.bibkey,
                    json.dumps(list(binding.evidence_ids), ensure_ascii=False),
                    binding.status,
                    binding.metadata_hash,
                    binding.source_type,
                    now,
                ),
            )

    def list_citation_bindings(self) -> list[object]:
        from datetime import date
        from app.scholar.citation.models import CitationBinding, CitationIdentity

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM citation_registry WHERE project_id=? ORDER BY identity_key",
                (self.project_id,),
            ).fetchall()
        result: list[object] = []
        for row in rows:
            value = json.loads(row["identity_json"])
            publication = value.get("publication_date")
            identity = CitationIdentity(
                canonical_id=value.get("canonical_id"),
                doi=value.get("doi"),
                arxiv_id=value.get("arxiv_id"),
                title=str(value.get("title") or ""),
                authors=tuple(str(item) for item in value.get("authors", [])),
                publication_date=date.fromisoformat(publication) if publication else None,
                venue=value.get("venue"),
                identity_method=str(value.get("identity_method") or ""),
                provenance=value.get("provenance", {}),
            )
            result.append(CitationBinding(
                binding_id=str(row["binding_id"]),
                project_id=str(row["project_id"]),
                citation_identity=identity,
                bibkey=row["bibkey"],
                evidence_ids=tuple(str(item) for item in json.loads(row["evidence_ids_json"])),
                status=str(row["status"]),
                metadata_hash=str(row["metadata_hash"]),
                source_type=row["source_type"],
            ))
        return result

    def get_citation_binding(self, binding_id: str) -> object:
        values = [value for value in self.list_citation_bindings() if value.binding_id == binding_id]  # type: ignore[attr-defined]
        if not values:
            raise KeyError(f"CitationBinding 不存在：{binding_id}")
        return values[0]

    def save_external_evidence_projection(
        self,
        *,
        project_id: str,
        evidence_id: str,
        identity_key: str,
        canonical_id: str | None,
        source_locator: str,
        provider: str,
        publication_date: str | None,
        retrieved_at: str | None,
        content_hash: str,
        validation_status: str,
        metadata: dict[str, object],
    ) -> None:
        if project_id != self.project_id:
            raise ValueError("PROJECT_CONFLICT: External Evidence 不属于当前 Project。")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO external_evidence_projection(
                    project_id, evidence_id, identity_key, canonical_id, source_locator,
                    provider, publication_date, retrieved_at, content_hash,
                    validation_status, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, evidence_id) DO UPDATE SET
                    identity_key=excluded.identity_key,
                    canonical_id=excluded.canonical_id,
                    source_locator=excluded.source_locator,
                    provider=excluded.provider,
                    publication_date=excluded.publication_date,
                    retrieved_at=excluded.retrieved_at,
                    content_hash=excluded.content_hash,
                    validation_status=excluded.validation_status,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    project_id,
                    evidence_id,
                    identity_key,
                    canonical_id,
                    source_locator,
                    provider,
                    publication_date,
                    retrieved_at,
                    content_hash,
                    validation_status,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
                    _utc_now(),
                ),
            )

    def list_external_evidence_projections(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM external_evidence_projection WHERE project_id=? ORDER BY evidence_id",
                (self.project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # Patch / manuscript / Build artifacts are Project-owned.  These methods
    # intentionally live beside Facts and Contributions so there is one local
    # SQLite owner instead of a second artifact database.
    def save_patch(
        self,
        patch: DraftPatch,
        *,
        review_report: ReviewReport | None = None,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        status: str = "AWAITING_APPROVAL",
    ) -> object:
        if patch.project_id != self.project_id:
            raise ValueError("PROJECT_CONFLICT: Patch 不属于当前 Scholar Project。")
        now = _utc_now()
        patch_json = json.dumps(_patch_to_dict(patch), ensure_ascii=False, sort_keys=True)
        review_json = (
            json.dumps(_review_to_dict(review_report), ensure_ascii=False, sort_keys=True)
            if review_report is not None
            else None
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM patches WHERE patch_id=?", (patch.patch_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO patches(
                        patch_id, project_id, status, patch_json, review_report_json,
                        source_session_id, source_run_id, result_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        patch.patch_id,
                        self.project_id,
                        status,
                        patch_json,
                        review_json,
                        source_session_id,
                        source_run_id,
                        now,
                        now,
                    ),
                )
            elif existing["patch_json"] != patch_json or existing["project_id"] != self.project_id:
                raise ValueError("PATCH_IMMUTABLE: 已存在同 ID 但内容不同的 Patch。")
            row = connection.execute(
                "SELECT * FROM patches WHERE patch_id=?", (patch.patch_id,)
            ).fetchone()
        assert row is not None
        return _stored_patch_from_row(row)

    def get_patch(self, patch_id: str) -> object:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM patches WHERE patch_id=?", (patch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Patch 不存在：{patch_id}")
        return _stored_patch_from_row(row)

    def list_patches(self) -> list[object]:
        """Read-only Project patch projection for Console/approval views."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM patches WHERE project_id=? ORDER BY created_at, patch_id",
                (self.project_id,),
            ).fetchall()
        return [_stored_patch_from_row(row) for row in rows]

    def transition_patch(
        self,
        patch_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        result: dict[str, object] | None = None,
    ) -> object:
        now = _utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM patches WHERE patch_id=?", (patch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Patch 不存在：{patch_id}")
            if expected_status is not None and row["status"] != expected_status:
                raise ValueError(
                    f"PATCH_STATE_CONFLICT: 期望 {expected_status}，实际为 {row['status']}。"
                )
            connection.execute(
                """
                UPDATE patches SET status=?, result_json=?, updated_at=?
                WHERE patch_id=?
                """,
                (
                    status,
                    json.dumps(result or json.loads(row["result_json"]), ensure_ascii=False, sort_keys=True),
                    now,
                    patch_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM patches WHERE patch_id=?", (patch_id,)
            ).fetchone()
        assert updated is not None
        return _stored_patch_from_row(updated)

    def record_patch_audit(
        self,
        patch_id: str,
        *,
        decision: str,
        actor: str,
        decision_at: str,
        expected_base_hash: str | None = None,
        previous_hash: str | None = None,
        new_hash: str | None = None,
        build_status: str | None = None,
        details: dict[str, object] | None = None,
    ) -> str:
        audit_id = f"AUDIT_{secrets.token_hex(8)}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO patch_audits(
                    audit_id, patch_id, decision, actor, decision_at,
                    expected_base_hash, previous_hash, new_hash, build_status, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    patch_id,
                    decision,
                    actor,
                    decision_at,
                    expected_base_hash,
                    previous_hash,
                    new_hash,
                    build_status,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
        return audit_id

    def list_patch_audits(self, patch_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM patch_audits WHERE patch_id=? ORDER BY decision_at, audit_id",
                (patch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_manuscript_state(self, state: ManuscriptState) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO manuscript_state(project_id, project_hash, root_tex, version, stale_sections_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    project_hash=excluded.project_hash,
                    root_tex=excluded.root_tex,
                    version=excluded.version,
                    stale_sections_json=excluded.stale_sections_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.project_id,
                    state.project_hash,
                    state.root_tex,
                    state.version,
                    json.dumps(state.stale_sections, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM manuscript_sections WHERE project_id=?", (self.project_id,)
            )
            connection.executemany(
                """
                INSERT INTO manuscript_sections(project_id, name, relative_path, content_hash, version, stale)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self.project_id,
                        section.name,
                        section.relative_path,
                        section.content_hash,
                        section.version,
                        int(section.stale),
                    )
                    for section in state.sections.values()
                ],
            )

    def get_manuscript_state_projection(self) -> dict[str, object] | None:
        with self._connect() as connection:
            state = connection.execute(
                "SELECT * FROM manuscript_state WHERE project_id=?", (self.project_id,)
            ).fetchone()
            sections = connection.execute(
                "SELECT * FROM manuscript_sections WHERE project_id=? ORDER BY name",
                (self.project_id,),
            ).fetchall()
        if state is None:
            return None
        return {
            "project_hash": state["project_hash"],
            "root_tex": state["root_tex"],
            "version": state["version"],
            "stale_sections": json.loads(state["stale_sections_json"]),
            "sections": [dict(row) for row in sections],
        }

    def load_manuscript_state(self) -> ManuscriptState | None:
        """Rehydrate only the deterministic state projection, never file content."""

        projection = self.get_manuscript_state_projection()
        if projection is None:
            return None
        from app.scholar.models import ManuscriptSection

        sections = {
            str(value["name"]): ManuscriptSection(
                name=str(value["name"]),
                relative_path=str(value["relative_path"]),
                content_hash=str(value["content_hash"]),
                version=int(value["version"]),
                stale=bool(value["stale"]),
            )
            for value in projection["sections"]  # type: ignore[index]
        }
        return ManuscriptState(
            project_root=str(self.project_root),
            root_tex=str(projection["root_tex"]),
            project_hash=str(projection["project_hash"]),
            version=int(projection["version"]),
            sections=sections,
            stale_sections=tuple(str(value) for value in projection["stale_sections"]),  # type: ignore[index]
        )

    def save_build_result(self, result: object) -> None:
        diagnostics = [
            {
                "file": value.file,
                "line": value.line,
                "column": value.column,
                "severity": value.severity,
                "message": value.message,
                "source": value.source,
            }
            for value in result.diagnostics  # type: ignore[attr-defined]
        ]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO build_results(
                    build_id, project_id, patch_id, status, started_at,
                    completed_at, diagnostics_json, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(build_id) DO UPDATE SET
                    status=excluded.status,
                    completed_at=excluded.completed_at,
                    diagnostics_json=excluded.diagnostics_json,
                    message=excluded.message
                """,
                (
                    result.build_id,  # type: ignore[attr-defined]
                    self.project_id,
                    result.patch_id,  # type: ignore[attr-defined]
                    result.status,  # type: ignore[attr-defined]
                    result.started_at,  # type: ignore[attr-defined]
                    result.completed_at,  # type: ignore[attr-defined]
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                    result.message,  # type: ignore[attr-defined]
                ),
            )

    def get_build_result(self, build_id: str) -> object:
        from app.scholar.approval.models import BuildResult, LatexDiagnostic

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM build_results WHERE build_id=?", (build_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Build 不存在：{build_id}")
        diagnostics = tuple(
            LatexDiagnostic(**value)
            for value in json.loads(row["diagnostics_json"])
        )
        return BuildResult(
            build_id=str(row["build_id"]),
            project_id=str(row["project_id"]),
            patch_id=row["patch_id"],
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            completed_at=row["completed_at"],
            diagnostics=diagnostics,
            message=str(row["message"]),
        )

    def latest_build_result(self, project_id: str | None = None) -> object | None:
        project_id = project_id or self.project_id
        with self._connect() as connection:
            row = connection.execute(
                "SELECT build_id FROM build_results WHERE project_id=? ORDER BY started_at DESC, build_id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        return self.get_build_result(str(row["build_id"])) if row else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _patch_to_dict(patch: DraftPatch) -> dict[str, object]:
    return {
        "patch_id": patch.patch_id,
        "target_section": patch.target_section,
        "base_hash": patch.base_hash,
        "proposed_content": patch.proposed_content,
        "used_claim_ids": list(patch.used_claim_ids),
        "used_evidence_ids": list(patch.used_evidence_ids),
        "reviewer_status": patch.reviewer_status,
        "project_id": patch.project_id,
        "citation_keys": list(patch.citation_keys),
        "contribution_ids": list(patch.contribution_ids),
        "review_report_id": patch.review_report_id,
        "change_summary": patch.change_summary,
        "original_content": patch.original_content,
        "warnings": list(patch.warnings),
        "bibliography_base_hash": patch.bibliography_base_hash,
        "citation_bindings": [
            value.to_dict() if hasattr(value, "to_dict") else value
            for value in patch.citation_bindings
        ],
        "bibliography_changes": [
            value.to_dict() if hasattr(value, "to_dict") else value
            for value in patch.bibliography_changes
        ],
        "citation_requirements": [
            value.__dict__ if hasattr(value, "__dict__") else {
                "evidence_id": value.evidence_id,
                "paper_id": value.paper_id,
                "reason": value.reason,
                "identity_key": getattr(value, "identity_key", None),
            }
            for value in patch.citation_requirements
        ],
        "citation_binding_ids": list(patch.citation_binding_ids),
    }


def _patch_from_dict(value: dict[str, object]) -> DraftPatch:
    return DraftPatch(
        patch_id=str(value["patch_id"]),
        target_section=str(value["target_section"]),
        base_hash=str(value["base_hash"]),
        proposed_content=str(value["proposed_content"]),
        used_claim_ids=tuple(str(item) for item in value.get("used_claim_ids", [])),
        used_evidence_ids=tuple(str(item) for item in value.get("used_evidence_ids", [])),
        reviewer_status=str(value.get("reviewer_status", "pending")),
        project_id=str(value["project_id"]) if value.get("project_id") is not None else None,
        citation_keys=tuple(str(item) for item in value.get("citation_keys", [])),
        contribution_ids=tuple(str(item) for item in value.get("contribution_ids", [])),
        review_report_id=str(value["review_report_id"]) if value.get("review_report_id") is not None else None,
        change_summary=str(value.get("change_summary", "")),
        original_content=str(value.get("original_content", "")),
        warnings=tuple(str(item) for item in value.get("warnings", [])),
        bibliography_base_hash=(str(value["bibliography_base_hash"]) if value.get("bibliography_base_hash") else None),
        citation_bindings=tuple(_citation_binding_from_dict(item) for item in value.get("citation_bindings", []) if isinstance(item, dict)),
        bibliography_changes=tuple(_bibliography_change_from_dict(item) for item in value.get("bibliography_changes", []) if isinstance(item, dict)),
        citation_requirements=tuple(_citation_requirement_from_dict(item) for item in value.get("citation_requirements", []) if isinstance(item, dict)),
        citation_binding_ids=tuple(str(item) for item in value.get("citation_binding_ids", [])),
    )


def _citation_identity_from_dict(value: dict[str, object]) -> object:
    from datetime import date
    from app.scholar.citation.models import CitationIdentity

    publication = value.get("publication_date")
    return CitationIdentity(
        canonical_id=str(value["canonical_id"]) if value.get("canonical_id") else None,
        doi=str(value["doi"]) if value.get("doi") else None,
        arxiv_id=str(value["arxiv_id"]) if value.get("arxiv_id") else None,
        title=str(value.get("title") or ""),
        authors=tuple(str(item) for item in value.get("authors", []) if item),
        publication_date=date.fromisoformat(str(publication)) if publication else None,
        venue=str(value["venue"]) if value.get("venue") else None,
        identity_method=str(value.get("identity_method") or ""),
        provenance=value.get("provenance", {}) if isinstance(value.get("provenance"), dict) else {},
    )


def _citation_binding_from_dict(value: dict[str, object]) -> object:
    from app.scholar.citation.models import CitationBinding

    identity = _citation_identity_from_dict(value.get("citation_identity", {}))
    return CitationBinding(
        binding_id=str(value["binding_id"]),
        project_id=str(value["project_id"]),
        citation_identity=identity,  # type: ignore[arg-type]
        bibkey=str(value["bibkey"]) if value.get("bibkey") else None,
        evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
        status=str(value.get("status") or "INVALID"),
        metadata_hash=str(value.get("metadata_hash") or ""),
        source_type=str(value["source_type"]) if value.get("source_type") else None,
    )


def _bib_entry_candidate_from_dict(value: dict[str, object]) -> object:
    from app.scholar.citation.models import BibEntryCandidate

    return BibEntryCandidate(
        bibkey=str(value["bibkey"]),
        entry_type=str(value.get("entry_type") or "misc"),
        title=str(value["title"]),
        authors=tuple(str(item) for item in value.get("authors", [])),
        year=int(value["year"]) if value.get("year") is not None else None,
        doi=str(value["doi"]) if value.get("doi") else None,
        arxiv_id=str(value["arxiv_id"]) if value.get("arxiv_id") else None,
        canonical_id=str(value["canonical_id"]) if value.get("canonical_id") else None,
        venue=str(value["venue"]) if value.get("venue") else None,
        url=str(value["url"]) if value.get("url") else None,
        metadata_provenance=value.get("metadata_provenance", {}) if isinstance(value.get("metadata_provenance"), dict) else {},
        metadata_hash=str(value.get("metadata_hash") or ""),
    )


def _bibliography_change_from_dict(value: dict[str, object]) -> object:
    from app.scholar.citation.models import BibliographyChange

    return BibliographyChange(
        relative_path=str(value.get("relative_path") or "references.bib"),
        identity_key=str(value["identity_key"]),
        entry=_bib_entry_candidate_from_dict(value["entry"]),  # type: ignore[arg-type]
        action=str(value.get("action") or "ADD"),  # type: ignore[arg-type]
    )


def _citation_requirement_from_dict(value: dict[str, object]) -> object:
    from app.scholar.citation.models import CitationRequirement

    return CitationRequirement(
        evidence_id=str(value.get("evidence_id") or ""),
        paper_id=str(value["paper_id"]) if value.get("paper_id") else None,
        reason=str(value.get("reason") or "verified evidence has no stable BibKey in the current bibliography"),
        identity_key=str(value["identity_key"]) if value.get("identity_key") else None,
    )


def _review_to_dict(report: ReviewReport | None) -> dict[str, object] | None:
    if report is None:
        return None
    return {
        "valid": report.valid,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "claim_id": issue.claim_id,
            }
            for issue in report.issues
        ],
        "revision_round": report.revision_round,
        "report_id": report.report_id,
    }


def _review_from_json(value: str | None) -> ReviewReport | None:
    if not value:
        return None
    payload = json.loads(value)
    return ReviewReport(
        valid=bool(payload["valid"]),
        issues=tuple(ReviewIssue(**issue) for issue in payload.get("issues", [])),
        revision_round=int(payload.get("revision_round", 0)),
        report_id=str(payload.get("report_id", "")),
    )


def _stored_patch_from_row(row: sqlite3.Row) -> object:
    from app.scholar.approval.models import StoredPatch

    return StoredPatch(
        patch=_patch_from_dict(json.loads(row["patch_json"])),
        status=str(row["status"]),
        review_report=_review_from_json(row["review_report_json"]),
        source_session_id=row["source_session_id"],
        source_run_id=row["source_run_id"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        result=json.loads(row["result_json"]),
    )
