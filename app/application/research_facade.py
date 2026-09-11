"""把 Session Runtime 与现有 Research Engine 连接起来的应用边界。"""

from __future__ import annotations

import secrets
import inspect
from uuid import uuid4
from typing import Any, Mapping

from app.session.context import ConversationContextBuilder
from app.session.manager import SessionManager


class ResearchApplicationFacade:
    """不暴露 LangGraph、RAG 或具体 Store 的 Research 应用门面。"""

    def __init__(
        self,
        agent_service: Any,
        session_manager: SessionManager,
        *,
        context_builder: ConversationContextBuilder | None = None,
        legacy_adapter: Any | None = None,
    ) -> None:
        if not callable(getattr(agent_service, "answer", None)):
            raise TypeError("agent_service 必须提供 answer(query, ...)。")
        self.agent_service = agent_service
        self.sessions = session_manager
        self.context_builder = context_builder or ConversationContextBuilder()
        self.legacy_adapter = legacy_adapter

    def research_topic(
        self,
        query: str,
        session_id: str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query 不能为空。")
        project_id = options.pop("project_id", None)
        job_id = options.pop("job_id", None)
        trace_id = f"TRACE_{uuid4().hex}"
        session = self._resolve_session(
            cleaned,
            session_id=session_id,
            project_id=str(project_id) if project_id else None,
        )
        with self.sessions.lock(session.session_id):
            runtime = self.sessions.open(session.session_id)
            run_id = f"RUN_{secrets.token_hex(8)}"
            thread_id = f"THREAD_{secrets.token_hex(8)}"
            runtime.create_run(
                cleaned,
                run_id=run_id,
                thread_id=thread_id,
                trace_id=trace_id,
                job_id=job_id,
                project_id=session.project_id,
            )
            self.sessions.set_active_run(session.session_id, run_id)
            runtime.start_run(run_id, worker_id=self.sessions.worker_id)
            runtime.append_message("user", cleaned, run_id=run_id)
            context = self.context_builder.build(
                runtime,
                cleaned,
                project_id=session.project_id,
            )
            try:
                options.setdefault(
                    "langchain_config",
                    {"configurable": {"thread_id": thread_id}},
                )
                options.update(
                    {
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "job_id": job_id,
                        "project_id": session.project_id,
                        "recent_conversation": tuple(context["recent_messages"]),
                    }
                )
                result = self._invoke_agent_service(
                    cleaned,
                    session_id=session.session_id,
                    options=options,
                )
                normalized = self._normalize(
                    result,
                    session_id=session.session_id,
                    run_id=run_id,
                    thread_id=thread_id,
                    trace_id=trace_id,
                    job_id=job_id,
                    context=context,
                )
                runtime.append_message(
                    "assistant",
                    normalized["answer"],
                    run_id=run_id,
                    metadata={"status": normalized["status"]},
                )
                runtime.complete_run(
                    run_id,
                    # A deterministic refusal is still a completed Run; FAILED
                    # is reserved for exceptions and unavailable results.
                    status="COMPLETED",
                    answer=normalized["answer"],
                    citations=normalized["citations"],
                    evidence=normalized["evidence"],
                    metadata=normalized["metadata"],
                )
                self.sessions.set_active_run(session.session_id, None)
                return normalized
            except Exception as error:
                runtime.fail_run(run_id, f"{type(error).__name__}: {error}")
                self.sessions.set_active_run(session.session_id, None)
                raise

    def _resolve_session(
        self,
        title: str,
        *,
        session_id: str | None,
        project_id: str | None,
    ) -> Any:
        if session_id is not None and self.legacy_adapter is not None:
            try:
                self.sessions.get(session_id)
            except KeyError:
                snapshot = self.legacy_adapter.snapshot(session_id)
                if snapshot is not None:
                    legacy_session = snapshot.get("session")
                    legacy_title = (
                        str(legacy_session.get("title") or title)
                        if isinstance(legacy_session, Mapping)
                        else title
                    )
                    session = self.sessions.create(
                        legacy_title,
                        session_id=session_id,
                        project_id=project_id,
                    )
                    self.legacy_adapter.migrate_messages(
                        session_id,
                        self.sessions.open(session_id),
                    )
                    return session
        return self.sessions.resolve(
            session_id,
            title=title,
            project_id=project_id,
        )

    def _invoke_agent_service(
        self,
        query: str,
        *,
        session_id: str,
        options: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """调用新 Service 的扩展参数，同时兼容旧显式签名 Service。"""

        answer = self.agent_service.answer
        parameters = inspect.signature(answer).parameters.values()
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        selected = dict(options)
        if not accepts_kwargs:
            names = set(inspect.signature(answer).parameters)
            selected = {key: value for key, value in selected.items() if key in names}
        return answer(query, session_id=session_id, **selected)

    @staticmethod
    def _normalize(
        result: Mapping[str, Any],
        *,
        session_id: str,
        run_id: str,
        thread_id: str,
        trace_id: str,
        job_id: str | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        answer = str(result.get("answer") or "")
        if not answer:
            claims = result.get("claims")
            if isinstance(claims, list):
                answer = "\n".join(
                    str(item.get("text") or "")
                    for item in claims
                    if isinstance(item, Mapping) and item.get("text")
                )
        answerable = bool(result.get("answerable", bool(answer)))
        diagnostics = result.get("diagnostics")
        diagnostics_mapping = diagnostics if isinstance(diagnostics, Mapping) else {}
        harness = diagnostics_mapping.get("harness")
        harness_mapping = harness if isinstance(harness, Mapping) else {}
        citations = result.get("citations")
        evidence = result.get("selected_evidence") or result.get("evidence")
        citations_list = [dict(item) for item in citations if isinstance(item, Mapping)] if isinstance(citations, list) else []
        evidence_list = [dict(item) for item in evidence if isinstance(item, Mapping)] if isinstance(evidence, list) else []
        normalized = {
            **dict(result),
            "session_id": session_id,
            "run_id": run_id,
            "status": "success" if answerable else "refused",
            "answer": answer,
            "citations": citations_list,
            "evidence": evidence_list,
            "metadata": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "job_id": job_id,
                "engine_trace_id": harness_mapping.get("run_id"),
                "context_message_count": len(context.get("recent_messages", [])),
                "termination_reason": harness_mapping.get("termination_reason"),
            },
        }
        return normalized
