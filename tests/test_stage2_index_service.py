from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.knowledge_engine import KnowledgeIndexService


def test_unique_index_service_records_complete_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.chunking.builder.build_knowledge_base",
        lambda root: SimpleNamespace(
            issues=[],
            document_count=1,
            total_chunk_count=2,
            to_dict=lambda: {"document_count": 1, "total_chunk_count": 2},
        ),
    )
    monkeypatch.setattr(
        "app.indexing.dense.build_dense_index",
        lambda root, provider: SimpleNamespace(to_dict=lambda: {"chunk_count": 2}),
    )
    service = KnowledgeIndexService(project_root=tmp_path)

    result = service.synchronize_after_parse(
        object(),
        catalog_builder=lambda root: SimpleNamespace(records=[1], summary=lambda: {"record_count": 1}),
    )

    operation = json.loads(
        (tmp_path / "data" / "knowledge" / "index_operations" / f"{result['operation_id']}.json").read_text()
    )
    assert result["lifecycle_state"] == "COMPLETED"
    assert operation["state"] == "COMPLETED"
    assert operation["recoverable"] is False


def test_unique_index_service_marks_failure_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.chunking.builder.build_knowledge_base",
        lambda root: (_ for _ in ()).throw(RuntimeError("index failed")),
    )
    service = KnowledgeIndexService(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="index failed"):
        service.synchronize_after_parse(
            object(),
            catalog_builder=lambda root: SimpleNamespace(records=[], summary=lambda: {}),
        )

    paths = list((tmp_path / "data" / "knowledge" / "index_operations").glob("*.json"))
    operation = json.loads(paths[0].read_text())
    assert operation["state"] == "FAILED"
    assert operation["recoverable"] is True
    assert operation["failure_type"] == "RuntimeError"
