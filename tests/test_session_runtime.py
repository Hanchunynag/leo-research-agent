from __future__ import annotations

from pathlib import Path

from app.application import ResearchApplicationFacade
from app.session import SessionManager


class FakeAgentService:
    def answer(self, query: str, **_: object) -> dict[str, object]:
        return {
            "answerable": True,
            "answer": f"Answer for {query}.",
            "citations": [{"evidence_id": "E1"}],
            "selected_evidence": [{"chunk_id": "C1", "content": "evidence"}],
            "workflow": "direct_qa",
            "diagnostics": {"harness": {"run_id": "RR_test"}},
        }


def test_session_manager_isolates_private_databases(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    first = manager.create("A", session_id="session-a")
    second = manager.create("B", session_id="session-b")

    manager.open(first.session_id).append_message("user", "AAA")
    manager.open(second.session_id).append_message("user", "BBB")

    assert [item["content"] for item in manager.open("session-a").recent_messages()] == ["AAA"]
    assert [item["content"] for item in manager.open("session-b").recent_messages()] == ["BBB"]
    assert manager.open("session-a").database_path != manager.open("session-b").database_path


def test_application_facade_creates_run_and_normalizes_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    facade = ResearchApplicationFacade(FakeAgentService(), manager)

    result = facade.research_topic("What is supported?", session_id="scholar")

    assert result["session_id"] == "scholar"
    assert str(result["run_id"]).startswith("RUN_")
    assert result["status"] == "success"
    assert result["citations"] == [{"evidence_id": "E1"}]
    runs = manager.open("scholar").list_runs()
    assert len(runs) == 1
    assert runs[0].status == "COMPLETED"
    assert [item["role"] for item in manager.open("scholar").recent_messages()] == [
        "user",
        "assistant",
    ]


def test_archive_and_soft_delete_preserve_session_directory(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create("Archive", session_id="archive")
    manager.archive("archive")
    assert manager.get("archive").status == "ARCHIVED"
    manager.delete("archive")
    assert manager.get("archive").status == "DELETED"
    assert (tmp_path / "data" / "runtime" / "sessions" / "archive" / "session.db").is_file()
