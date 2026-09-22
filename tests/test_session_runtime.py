from __future__ import annotations

from pathlib import Path

from app.session import SessionManager


def test_session_manager_isolates_private_databases(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    first = manager.create("A", session_id="session-a")
    second = manager.create("B", session_id="session-b")

    manager.open(first.session_id).append_message("user", "AAA")
    manager.open(second.session_id).append_message("user", "BBB")

    assert [item["content"] for item in manager.open("session-a").recent_messages()] == ["AAA"]
    assert [item["content"] for item in manager.open("session-b").recent_messages()] == ["BBB"]
    assert manager.open("session-a").database_path != manager.open("session-b").database_path


def test_archive_and_soft_delete_preserve_session_directory(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create("Archive", session_id="archive")
    manager.archive("archive")
    assert manager.get("archive").status == "ARCHIVED"
    manager.delete("archive")
    assert manager.get("archive").status == "DELETED"
    assert (tmp_path / "data" / "runtime" / "sessions" / "archive" / "session.db").is_file()
