"""Environment configuration for index and GraphRAG planes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _dotenv(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#") and "=" in cleaned:
            key, _, value = cleaned.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class GraphRAGConfig:
    project_root: Path
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = field(default="", repr=False)
    graph_database: str = "neo4j"
    index_registry_path: Path | None = None
    lexical_index_path: Path | None = None
    graph_extraction_model: str = ""
    graph_extraction_concurrency: int = 2
    graph_extraction_prompt_version: str = "1.0"
    graph_ontology_version: str = "1.0"
    community_prompt_version: str = "1.0"
    max_query_variants: int = 5
    max_graph_hops: int = 2
    max_graph_paths: int = 20
    max_cross_query_candidates: int = 40

    @classmethod
    def from_environment(cls, project_root: Path,
                         env_file: Path | None = None) -> "GraphRAGConfig":
        root = project_root.expanduser().resolve()
        values = _dotenv(env_file or root / ".env")
        def get(key: str, default: str = "") -> str:
            return os.getenv(key, values.get(key, default))

        def path(key: str, default: Path) -> Path:
            return Path(get(key, str(default))).expanduser().resolve()
        return cls(project_root=root,
            neo4j_uri=get("LEO_NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=get("LEO_NEO4J_USER", "neo4j"),
            neo4j_password=get("LEO_NEO4J_PASSWORD"),
            graph_database=get("LEO_GRAPH_DATABASE", "neo4j"),
            index_registry_path=path("LEO_INDEX_REGISTRY_PATH", root / "data/index/index_registry.sqlite3"),
            lexical_index_path=path("LEO_LEXICAL_INDEX_PATH", root / "data/index/lexical.sqlite3"),
            graph_extraction_model=get("LEO_GRAPH_EXTRACTION_MODEL"),
            graph_extraction_concurrency=int(get("LEO_GRAPH_EXTRACTION_CONCURRENCY", "2")),
            graph_extraction_prompt_version=get("LEO_GRAPH_EXTRACTION_PROMPT_VERSION", "1.0"),
            graph_ontology_version=get("LEO_GRAPH_ONTOLOGY_VERSION", "1.0"),
            community_prompt_version=get("LEO_COMMUNITY_PROMPT_VERSION", "1.0"),
            max_query_variants=int(get("LEO_MAX_QUERY_VARIANTS", "5")),
            max_graph_hops=int(get("LEO_MAX_GRAPH_HOPS", "2")),
            max_graph_paths=int(get("LEO_MAX_GRAPH_PATHS", "20")),
            max_cross_query_candidates=int(get("LEO_MAX_CROSS_QUERY_CANDIDATES", "40")))

    def __post_init__(self) -> None:
        if not 1 <= self.graph_extraction_concurrency <= 32:
            raise ValueError("graph extraction concurrency must be 1..32")
        if not 1 <= self.max_query_variants <= 5:
            raise ValueError("max query variants must be 1..5")
        if self.max_graph_hops not in {1, 2}:
            raise ValueError("max graph hops must be 1 or 2")
        if not 1 <= self.max_graph_paths <= 100:
            raise ValueError("max graph paths must be 1..100")
        if not 1 <= self.max_cross_query_candidates <= 100:
            raise ValueError("max cross-query candidates must be 1..100")
