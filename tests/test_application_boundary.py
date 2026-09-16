from __future__ import annotations

from pathlib import Path

import pytest

from app.application import ResearchApplicationFacade
from app.application import LegacySessionAdapter
from app.agentic.store import AgenticSessionStore
from app.session import SessionManager


class CapturingAgentService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def answer(self, query: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"query": query, **kwargs})
        return {
            "answerable": True,
            "answer": "grounded answer",
            "citations": [],
            "selected_evidence": [],
            "diagnostics": {"harness": {"run_id": "ENGINE_TRACE"}},
        }


def test_facade_keeps_application_ids_distinct_and_passes_thread_config(
    tmp_path: Path,
) -> None:
    service = CapturingAgentService()
    facade = ResearchApplicationFacade(service, SessionManager(tmp_path))

    result = facade.research_topic(
        "query",
        session_id="s1",
        project_id="PROJECT_1",
        job_id="JOB_1",
    )

    call = service.calls[0]
    thread_id = call["langchain_config"]["configurable"]["thread_id"]  # type: ignore[index]
    assert result["run_id"] != result["metadata"]["trace_id"]
    assert result["run_id"] != thread_id
    assert result["metadata"]["trace_id"] != thread_id
    assert result["metadata"]["job_id"] == "JOB_1"
    assert call["run_id"] == result["run_id"]
    assert call["trace_id"] == result["metadata"]["trace_id"]
    run = facade.sessions.open("s1").list_runs()[0]
    assert run.trace_id == result["metadata"]["trace_id"]
    assert run.job_id == "JOB_1"
    assert run.project_id == "PROJECT_1"
    assert facade.sessions.get("s1").active_run_id is None


def test_facade_persists_refusal_reason_when_answer_has_no_claims(
    tmp_path: Path,
) -> None:
    class RefusingAgentService:
        def answer(self, query: str, **kwargs: object) -> dict[str, object]:
            return {
                "answerable": False,
                "answer": "",
                "claims": [],
                "refusal_reason": "回答模型不可用，暂时无法生成带引用答案。",
                "citations": [],
                "selected_evidence": [],
                "diagnostics": {},
            }

    facade = ResearchApplicationFacade(RefusingAgentService(), SessionManager(tmp_path))

    result = facade.research_topic("测试拒答持久化", session_id="refusal")

    assert result["answerable"] is False
    assert result["answer"] == "回答模型不可用，暂时无法生成带引用答案。"
    messages = facade.sessions.open("refusal").list_messages()
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == result["answer"]
    assert facade.sessions.open("refusal").list_runs()[0].status == "COMPLETED"


def test_project_id_can_be_shared_but_cannot_cross_attach_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    first = manager.resolve("s1", title="one", project_id="PROJECT_1")
    second = manager.resolve("s2", title="two", project_id="PROJECT_1")

    assert first.project_id == second.project_id == "PROJECT_1"
    with pytest.raises(ValueError, match="其他 Project"):
        manager.resolve("s1", title="one", project_id="PROJECT_2")


def test_orphan_running_run_is_marked_interrupted(tmp_path: Path) -> None:
    first_manager = SessionManager(tmp_path)
    first_manager.create("one", session_id="s1")
    runtime = first_manager.open("s1")
    runtime.create_run("query", run_id="RUN_1", thread_id="THREAD_1")
    runtime.start_run("RUN_1", worker_id="old-worker")

    second_manager = SessionManager(tmp_path)
    assert second_manager.recover_orphans() == 1
    assert second_manager.open("s1").get_run("RUN_1").status == "INTERRUPTED"


def test_legacy_session_is_migrated_once_and_new_writes_use_session_runtime(
    tmp_path: Path,
) -> None:
    legacy = AgenticSessionStore(tmp_path, tmp_path / "legacy.sqlite3")
    legacy.create_session("legacy title", session_id="legacy-1")
    topic = legacy.create_topic(
        "legacy-1",
        relation="same_topic",
        topic_summary="old topic",
        user_goal="old goal",
        entities=(),
    )
    legacy.append_event(
        "legacy-1",
        str(topic["topic_id"]),
        "user_query",
        {"query": "old question"},
    )

    manager = SessionManager(tmp_path)
    facade = ResearchApplicationFacade(
        CapturingAgentService(),
        manager,
        legacy_adapter=LegacySessionAdapter(legacy),
    )
    facade.research_topic("new question", session_id="legacy-1")

    messages = manager.open("legacy-1").list_messages()
    assert [value["content"] for value in messages[:2]] == [
        "old question",
        "new question",
    ]
    assert len(manager.open("legacy-1").list_runs()) == 1
