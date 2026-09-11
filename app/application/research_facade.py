"""把 Session Runtime 与现有 Research Engine 连接起来的应用边界。"""

from __future__ import annotations

import secrets
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
    ) -> None:
        if not callable(getattr(agent_service, "answer", None)):
            raise TypeError("agent_service 必须提供 answer(query, ...)。")
        self.agent_service = agent_service
        self.sessions = session_manager
        self.context_builder = context_builder or ConversationContextBuilder()

    def research_topic(
        self,
        query: str,
        session_id: str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query 不能为空。")
        session = self.sessions.resolve(session_id, title=cleaned)
        with self.sessions.lock(session.session_id):
            runtime = self.sessions.open(session.session_id)
            run_id = f"RUN_{secrets.token_hex(8)}"
            thread_id = f"THREAD_{secrets.token_hex(8)}"
            runtime.create_run(cleaned, run_id=run_id, thread_id=thread_id)
            runtime.start_run(run_id, worker_id=self.sessions.worker_id)
            runtime.append_message("user", cleaned, run_id=run_id)
            context = self.context_builder.build(runtime, cleaned)
            try:
                result = self.agent_service.answer(
                    cleaned,
                    session_id=session.session_id,
                    **options,
                )
                normalized = self._normalize(
                    result,
                    session_id=session.session_id,
                    run_id=run_id,
                    thread_id=thread_id,
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
                return normalized
            except Exception as error:
                runtime.fail_run(run_id, f"{type(error).__name__}: {error}")
                raise

    @staticmethod
    def _normalize(
        result: Mapping[str, Any],
        *,
        session_id: str,
        run_id: str,
        thread_id: str,
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
        return {
            "session_id": session_id,
            "run_id": run_id,
            "status": "success" if answerable else "refused",
            "answer": answer,
            "citations": citations_list,
            "evidence": evidence_list,
            "metadata": {
                "thread_id": thread_id,
                "workflow": result.get("workflow"),
                "trace_id": harness_mapping.get("run_id"),
                "context_message_count": len(context.get("recent_messages", [])),
                "termination_reason": harness_mapping.get("termination_reason"),
            },
        }
