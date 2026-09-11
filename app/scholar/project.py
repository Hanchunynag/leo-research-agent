"""Project-level Manuscript Facts 与 Contribution Registry。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.scholar.models import Contribution, ManuscriptFact


class ScholarProjectStore:
    """事实与创新点的唯一项目级持久化入口。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.directory = self.project_root / ".scholar"
        self.database_path = self.directory / "project.db"
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
                """
            )

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
