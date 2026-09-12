from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pydantic import BaseModel

from app.langchain_agent.provider import OpenAICompatibleChatModel
from app.scholar.harness import ScholarHarnessService
from app.langchain_agent.checkpoint_factory import open_checkpointer
from app.scholar.models import EvidencePack
from app.scholar.writing.harness import ScholarSkillRuntime
from app.scholar.writing.runtime import SkillResult
from app.scholar.writing.support import ClaimSupportResult


class _Provider:
    model_name = "fixture"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, messages: list[dict[str, Any]], *, max_tokens: int, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, **kwargs})
        return {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_project_context", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }


def test_langchain_1x_chat_model_adapter_binds_tools_without_json_mode() -> None:
    provider = _Provider()
    model = OpenAICompatibleChatModel(provider=provider, model_name="fixture")
    bound = model.bind_tools([{"type": "function", "function": {"name": "get_project_context", "parameters": {"type": "object"}}}])

    response = bound.invoke([{"role": "user", "content": "context"}])

    assert response.tool_calls[0]["name"] == "get_project_context"
    assert "tools" in provider.calls[0]
    assert "response_format" not in provider.calls[0]


def test_langchain_1x_structured_output_stays_at_provider_adapter_boundary() -> None:
    class _StructuredProvider(_Provider):
        def chat_completion(self, messages: list[dict[str, Any]], *, max_tokens: int, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"messages": messages, "max_tokens": max_tokens, **kwargs})
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "structured-1",
                            "type": "function",
                            "function": {"name": "Decision", "arguments": '{"answer":"ok"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            }

    class Decision(BaseModel):
        answer: str

    provider = _StructuredProvider()
    model = OpenAICompatibleChatModel(provider=provider, model_name="fixture")
    result = model.with_structured_output(Decision, method="function_calling").invoke("structured")

    assert result == Decision(answer="ok")
    assert provider.calls[0]["tools"][0]["function"]["name"] == "Decision"


class _ScriptedModel(BaseChatModel):
    model_name: str = "fixture"
    responses: tuple[dict[str, Any], ...] = ()
    state: dict[str, Any] = Field(default_factory=lambda: {"calls": 0, "bound_tools": [], "system_prompts": []})

    @property
    def _llm_type(self) -> str:
        return "fixture"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **_: Any) -> "_ScriptedModel":
        self.state["bound_tools"].append(tuple(str(tool.name) for tool in tools))
        return self

    def _generate(self, messages: list[Any], stop: list[str] | None = None, **_: Any) -> ChatResult:
        system = next((str(message.content) for message in messages if getattr(message, "type", "") == "system"), "")
        self.state["system_prompts"].append(system)
        response = self.responses[min(self.state["calls"], len(self.responses) - 1)]
        self.state["calls"] += 1
        if response.get("content") is not None:
            message = AIMessage(content=str(response["content"]))
        else:
            message = AIMessage(
                content="",
                tool_calls=[{
                    "name": str(response["tool"]),
                    "args": dict(response.get("args") or {}),
                    "id": f"call-{self.state['calls']}",
                    "type": "tool_call",
                }],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Research:
    workspace_id = "default"
    scope_version = 1

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def research(self, request: Any, *, allow_web: bool, harness: Any) -> EvidencePack:
        self.calls.append((request, allow_web))
        return EvidencePack(
            request_id=request.request_id,
            query=request.query,
            claims=({"status": "unresolved", "evidence_ids": ()},),
        )


def _runtime(tmp_path: Path) -> ScholarSkillRuntime:
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nText.\\end{document}",
        encoding="utf-8",
    )
    shutil.copytree(Path(__file__).parents[1] / "skills", tmp_path / "skills")
    runtime = ScholarSkillRuntime(tmp_path)
    runtime.execute = lambda request, writer=None: SkillResult(  # type: ignore[method-assign]
        "SUPPORT_CLAIM" if hasattr(request, "claim") else request.task_type,
        "support-claim" if hasattr(request, "claim") else "write-conclusion",
        "READY",
        ClaimSupportResult("claim", "claim", (), "INSUFFICIENT_EVIDENCE")
        if hasattr(request, "claim")
        else {"draft": "domain-owned"},
    )
    return runtime


def test_support_claim_delegates_research_and_returns_domain_result(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    research = _Research()
    model = _ScriptedModel(responses=(
        {"tool": "get_project_context"},
        {"tool": "task", "args": {"description": "Research this claim", "subagent_type": "research"}},
        {"tool": "research_evidence", "args": {"query": "claim", "target_claim": "claim"}},
        {"content": "research complete"},
        {"tool": "execute_scholar_skill", "args": {"task_type": "SUPPORT_CLAIM", "instruction": "claim"}},
    ))
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=model,
        research=research,
    )

    result = service.scholar_request("support claim", runtime.project_store.project_id, task_type="SUPPORT_CLAIM")

    assert result.result_type == "ClaimSupportResult"
    assert len(research.calls) == 1
    assert result.metadata["selected_skill"] == "support-claim"
    assert "research" in result.metadata["subagent_names"]
    assert result.metadata["unexpected_tool_calls"] == []


def test_synthesis_harness_has_no_research_subagent_or_write_filesystem(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    model = _ScriptedModel(responses=(
        {"tool": "get_project_context"},
        {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "write conclusion"}},
    ))
    service = ScholarHarnessService(
        tmp_path,
        skill_runtime=runtime,
        model=model,
        writers={"WRITE_CONCLUSION": object()},
    )

    result = service.scholar_request("write conclusion", runtime.project_store.project_id, task_type="WRITE_CONCLUSION")

    assert result.metadata["subagent_names"] == ["reviewer"]
    assert all(
        tool_name not in {"write_file", "edit_file", "delete", "execute"}
        for tools in model.state["bound_tools"]
        for tool_name in tools
    )
    assert result.metadata["skill_metadata_loaded_progressively"] is True
    assert any("Available Skills" in prompt for prompt in model.state["system_prompts"])


def test_deep_agent_checkpoint_resume_survives_service_rebuild(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    database = tmp_path / "scholar-harness.db"
    first_model = _ScriptedModel(responses=(
        {"tool": "get_project_context"},
        {"tool": "execute_scholar_skill", "args": {"task_type": "WRITE_CONCLUSION", "instruction": "write conclusion"}},
    ))
    with open_checkpointer("sqlite", database_path=database) as checkpointer:
        first = ScholarHarnessService(
            tmp_path,
            skill_runtime=runtime,
            model=first_model,
            writers={"WRITE_CONCLUSION": object()},
            checkpointer=checkpointer,
            interrupt_on={"execute_scholar_skill": True},
        )
        interrupted = first.scholar_request(
            "write conclusion",
            runtime.project_store.project_id,
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_RESTART",
            thread_id="THREAD_RESTART",
        )
        assert interrupted.status == "INTERRUPTED"
        assert interrupted.metadata["trace"]["termination_reason"] == "checkpoint_interrupt"

    # A new model/service/runtime object resumes the persisted Deep Agents
    # working state.  The resume decision is framework HIL data, never Patch
    # Approval, and the Domain Result remains produced by the existing Skill.
    rebuilt_runtime = ScholarSkillRuntime(tmp_path)
    rebuilt_runtime.execute = lambda request, writer=None: SkillResult(
        request.task_type,
        "write-conclusion",
        "READY",
        {"draft": "domain-owned-after-restart"},
    )  # type: ignore[method-assign]
    rebuilt_model = _ScriptedModel(responses=({"content": "completed after restart"},))
    with open_checkpointer("sqlite", database_path=database) as checkpointer:
        rebuilt = ScholarHarnessService(
            tmp_path,
            skill_runtime=rebuilt_runtime,
            model=rebuilt_model,
            writers={"WRITE_CONCLUSION": object()},
            checkpointer=checkpointer,
            interrupt_on={"execute_scholar_skill": True},
        )
        resumed = rebuilt.resume(
            "THREAD_RESTART",
            {"decisions": [{"type": "approve"}]},
            rebuilt_runtime.project_store.project_id,
            instruction="write conclusion",
            task_type="WRITE_CONCLUSION",
            session_id="SESSION_RESTART",
        )

    assert resumed.status == "COMPLETED"
    assert resumed.metadata["resumed"] is True
    assert resumed.result_type == "dict"
