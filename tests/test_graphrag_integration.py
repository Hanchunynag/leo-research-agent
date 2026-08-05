from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.graph.client import Neo4jClient, Neo4jSettings
from app.graph.schema import ensure_schema


@pytest.mark.neo4j
def test_neo4j_schema_and_connectivity() -> None:
    if os.getenv("LEO_RUN_NEO4J_TESTS") != "1":
        pytest.skip("set LEO_RUN_NEO4J_TESTS=1 with docker-compose.graph.yml running")
    settings = Neo4jSettings.from_environment(Path(".env"))
    with Neo4jClient(settings) as client:
        report = ensure_schema(client.driver, settings.database)
        assert report == {"constraints": 7, "indexes": 6}
        assert client.status()["connected"] is True


@pytest.mark.qdrant
def test_qdrant_marker_is_available() -> None:
    # Local Qdrant behavior and no-change embedding are covered by
    # test_graphrag_core; this marker lets CI select the backend suite.
    assert True
