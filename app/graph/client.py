"""Secret-safe Neo4j driver configuration and health checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from neo4j import GraphDatabase


def _read_env(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, _, value = cleaned.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = field(default="", repr=False)
    database: str = "neo4j"

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "Neo4jSettings":
        file_values = _read_env(env_file)
        def get(key: str, default: str = "") -> str:
            return os.getenv(key, file_values.get(key, default))
        return cls(uri=get("LEO_NEO4J_URI", cls.uri), user=get("LEO_NEO4J_USER", cls.user),
                   password=get("LEO_NEO4J_PASSWORD"),
                   database=get("LEO_GRAPH_DATABASE", cls.database))

    def __post_init__(self) -> None:
        if not self.uri.startswith(("bolt://", "neo4j://", "neo4j+s://", "bolt+s://")):
            raise ValueError("LEO_NEO4J_URI must be a Neo4j/Bolt URI")
        if not self.user or not self.password:
            raise ValueError("LEO_NEO4J_USER and LEO_NEO4J_PASSWORD are required")


class Neo4jClient:
    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self.driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))

    def close(self) -> None:
        self.driver.close()

    def status(self) -> dict[str, object]:
        self.driver.verify_connectivity()
        with self.driver.session(database=self.settings.database) as session:
            component = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"
            ).data()
        return {"connected": True, "database": self.settings.database,
                "server": dict(component) if component else {}, "node_counts": counts}

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
