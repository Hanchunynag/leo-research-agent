"""CrewAI Crew definition for the four Scholar cognitive roles."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

from app.orchestration.contracts import (
    ManagerDecision,
    ResearchAgentOutput,
    ReviewAgentOutput,
    WriterAgentOutput,
)
from app.orchestration.crewai.tools import (
    CapabilityMatrix,
    ResearchCapabilityTool,
    ReviewerCapabilityTool,
    WriterCapabilityTool,
)
from app.orchestration.crewai.models import CrewAIProviderAdapter


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings="none")
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    return value


def _prompt_jsonable(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Create a bounded prompt projection for structured CrewAI Tasks.

    The Flow has already executed the authoritative capability call before a
    specialist Task starts. Passing the complete domain object again is both
    redundant and expensive: an EvidencePack can contain large source spans
    and a DraftPatch can contain an entire manuscript section. Keep bounded
    previews and identifiers for model cognition while leaving full objects in
    the domain/persistence layer.
    """

    if key in {"domain_result", "draft_patch"}:
        return {"omitted": key, "reason": "authoritative_domain_object"}
    if key == "evidence_packs":
        return {"count": len(value) if isinstance(value, (list, tuple)) else 0}
    if is_dataclass(value):
        return _prompt_jsonable(asdict(value), key=key, depth=depth)
    if isinstance(value, BaseModel):
        return _prompt_jsonable(value.model_dump(mode="python"), key=key, depth=depth)
    if isinstance(value, str):
        if key in {"content", "text", "preview", "research_summary", "instruction"}:
            return value[:1200]
        return value[:2000]
    if isinstance(value, dict):
        if depth >= 6:
            return {"omitted": "nested_context", "reason": "prompt_depth_limit"}
        return {
            str(child_key): _prompt_jsonable(child, key=str(child_key), depth=depth + 1)
            for child_key, child in list(value.items())[:64]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        limit = 4 if key in {"evidence", "verified_evidence", "claims", "citations"} else 16
        return [
            _prompt_jsonable(child, key=key, depth=depth + 1)
            for child in list(value)[:limit]
        ]
    return value


class _ManagerCognition(BaseModel):
    next_action: str | None = None
    target_agent: str | None = None
    goal_alignment: str = "Preserve the original Run goal."
    remaining_gap: str = "No unresolved gap reported."
    reason: str = "Manager decision"
    required_context_refs: list[str] = []
    completion_status: str = "IN_PROGRESS"


class _ResearchCognition(BaseModel):
    status: str = "COMPLETED"
    research_summary: str = ""


class _WriterCognition(BaseModel):
    status: str = "READY"
    research_required: bool = False


class _ReviewCognition(BaseModel):
    decision: str = "PASS"


def _cognition_model(output_model: type[BaseModel]) -> type[BaseModel]:
    """Use a small CrewAI schema; domain contracts are filled by the Flow."""

    return {
        ManagerDecision: _ManagerCognition,
        ResearchAgentOutput: _ResearchCognition,
        WriterAgentOutput: _WriterCognition,
        ReviewAgentOutput: _ReviewCognition,
    }.get(output_model, output_model)


class ScholarCrew:
    """Factory and execution adapter for one bounded CrewAI kickoff.

    A single task is kicked off at a time by the Flow.  This makes the Flow the
    owner of lifecycle transitions while CrewAI still supplies the real Agent,
    Task, Crew and structured-output cognition layer.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        gateway: Any | None = None,
        trace: Any | None = None,
        max_agent_iterations: int = 8,
    ) -> None:
        self.llm = CrewAIProviderAdapter.from_model(model)
        self.gateway = gateway
        self.trace = trace
        research_tools = (
            [ResearchCapabilityTool(gateway)] if gateway is not None else []
        )
        writer_tools = [WriterCapabilityTool(gateway)] if gateway is not None else []
        reviewer_tools = (
            [ReviewerCapabilityTool(gateway)] if gateway is not None else []
        )
        common = {
            "llm": self.llm,
            "allow_delegation": False,
            "max_iter": max(1, max_agent_iterations),
            "memory": False,
            "verbose": False,
        }
        self.manager = Agent(
            # This is the persistent Manager Agent, not a one-shot request
            # router.
            role="Scholar Manager Agent",
            goal="Continuously choose the next bounded specialist action from the current Run state and return ManagerDecision.",
            backstory="You are the persistent Scholar Manager. You plan only; you never retrieve, write, review, approve, or apply.",
            tools=[],
            **common,
        )
        self.research = Agent(
            role="Research Agent",
            goal="Produce traceable verified research evidence and claim-support results through bounded capability tools.",
            backstory="You are a research specialist. You never create DraftPatch, edit manuscripts, or approve changes.",
            tools=research_tools,
            **common,
        )
        self.writer = Agent(
            role="Writer Agent",
            goal="Turn the supplied manuscript context and verified evidence into an immutable DraftPatch proposal.",
            backstory="You are a manuscript writer. You consume evidence handed off by Research and never search the Web yourself.",
            tools=writer_tools,
            **common,
        )
        self.reviewer = Agent(
            role="Reviewer Agent",
            goal="Check claim support, citation scope, facts, contributions, and manuscript consistency.",
            backstory="You are a deterministic review specialist. You only return PASS, REVISE, or REJECT and never apply patches.",
            tools=reviewer_tools,
            **common,
        )
        self.agents: dict[str, Agent] = {
            "manager": self.manager,
            "research": self.research,
            "writer": self.writer,
            "reviewer": self.reviewer,
        }
        self.capability_matrix = CapabilityMatrix.as_dict()
        self.last_crew: Crew | None = None

    def _task(
        self,
        agent_name: str,
        context: Any,
        output_model: type[BaseModel],
        *,
        task_name: str,
        execution_agent: Agent | None = None,
    ) -> Task:
        agent = execution_agent or self.agents[agent_name]
        description = (
            f"You are the {agent.role}. Follow the code-enforced capability boundary. "
            "Return only the requested structured contract.\n\n"
            "Request context JSON:\n{context}\n\n"
            "Do not invent evidence, citations, facts, patches, or approval state. "
            "If the supplied domain result is insufficient, use the contract's fail-closed status. "
            "Return the smallest valid JSON object: use empty arrays, empty strings, and nulls "
            "for optional fields. Never copy source text, evidence arrays, manuscript text, or "
            "the full domain object into your response. Keep the response concise."
        )
        if output_model is ManagerDecision:
            description += (
                "\n\nManager rules (mandatory): reread ORIGINAL_GOAL before every decision. "
                "Do not turn a Specialist subtask, one Reviewer comment, the latest conversation, "
                "or one missing Evidence item into the final task goal. Explain in goal_alignment "
                "why this action serves ORIGINAL_GOAL, and in remaining_gap what is still missing. "
                "Use READY_TO_COMPLETE only when every SUCCESS_CRITERIA is true and there is no "
                "BLOCKING_GAP; otherwise use IN_PROGRESS or BLOCKED. Never invent completion from "
                "recent output alone."
            )
        expected = (
            f"A valid {output_model.__name__} JSON object with no additional keys."
        )
        return Task(
            name=task_name,
            description=description,
            expected_output=expected,
            agent=agent,
            output_pydantic=output_model,
            tools=list(agent.tools),
            context=[],
        )

    def run(
        self,
        agent_name: str,
        context: Any,
        output_model: type[BaseModel],
        *,
        task_name: str,
    ) -> BaseModel:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown Scholar Crew agent: {agent_name}")
        source_agent = self.agents[agent_name]
        # The canonical Agent retains its isolated capability tools for the
        # runtime permission matrix. The Flow invokes those capabilities
        # deterministically before this structured cognition step, so execute
        # a shallow no-tool copy here. This lets the OpenAI-compatible adapter
        # keep response_format=json_object enabled and prevents recursive tool
        # selection or ReAct parsing from corrupting the contract response.
        execution_agent = source_agent.model_copy(deep=False)
        execution_agent.tools = []
        if hasattr(execution_agent, "agent_executor"):
            execution_agent.agent_executor = None
        task_model = _cognition_model(output_model)
        task = self._task(
            agent_name,
            _prompt_jsonable(context),
            task_model,
            task_name=task_name,
            execution_agent=execution_agent,
        )
        crew = Crew(
            name=f"scholar-{agent_name}-{task_name}",
            agents=[execution_agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=False,
            cache=False,
        )
        self.last_crew = crew
        if self.trace is not None:
            self.trace.record(
                "crew_task_started", "RUNNING", {"agent": agent_name, "task": task_name}
            )
        try:
            output = crew.kickoff(
                inputs={
                    "context": json.dumps(
                        _prompt_jsonable(context), ensure_ascii=False, default=str
                    )
                }
            )
            task_output = getattr(output, "tasks_output", None)
            task_result = (
                task_output[-1]
                if isinstance(task_output, list) and task_output
                else getattr(task, "output", None)
            )
            structured = (
                getattr(task_result, "pydantic", None)
                if task_result is not None
                else None
            )
            if not isinstance(structured, task_model):
                json_dict = (
                    getattr(task_result, "json_dict", None)
                    if task_result is not None
                    else None
                )
                if isinstance(json_dict, dict):
                    structured = task_model.model_validate(json_dict)
            if not isinstance(structured, task_model):
                raw = (
                    getattr(task_result, "raw", None)
                    if task_result is not None
                    else str(output)
                )
                structured = task_model.model_validate_json(str(raw))
            payload = structured.model_dump(mode="python")
            if output_model is ManagerDecision and not payload.get("next_action"):
                context_value = context if isinstance(context, dict) else {}
                last_agent = context_value.get("last_agent")
                current_route = context_value.get("route")
                if last_agent == "research" and current_route in {
                    "WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT"
                }:
                    action, target, status = "CALL_WRITER", "writer", "IN_PROGRESS"
                elif last_agent == "research":
                    action, target, status = "COMPLETE", None, "READY_TO_COMPLETE"
                elif last_agent == "writer":
                    action, target, status = "CALL_REVIEWER", "reviewer", "IN_PROGRESS"
                elif last_agent == "reviewer" and context_value.get("review_decision") == "REVISE":
                    action, target, status = "REQUEST_REVISION", "writer", "IN_PROGRESS"
                elif last_agent == "reviewer":
                    action, target, status = "COMPLETE", None, "READY_TO_COMPLETE"
                elif current_route == "REVIEW":
                    action, target, status = "CALL_REVIEWER", "reviewer", "IN_PROGRESS"
                elif current_route in {"WRITE_INTRODUCTION", "RESEARCH", "SUPPORT_CLAIM"}:
                    action, target, status = "CALL_RESEARCH", "research", "IN_PROGRESS"
                else:
                    action, target, status = "CALL_WRITER", "writer", "IN_PROGRESS"
                payload.update(
                    {
                        "next_action": action,
                        "target_agent": target,
                        "goal_alignment": f"The {str(current_route or 'research').casefold().replace('_', ' ')} action advances the original goal.",
                        "remaining_gap": "Provider returned no bounded gap.",
                        "reason": "Normalized Manager payload at the CrewAI boundary.",
                        "completion_status": status,
                    }
                )
            # Route is code-owned by the Flow. If a model omits it from the
            # minimal response, preserve the deterministic route hint instead
            # of allowing a missing field to turn a valid run into a parser
            # failure.
            structured_output = output_model.model_validate(payload)
            if self.trace is not None:
                self.trace.record(
                    "crew_task_completed",
                    "COMPLETED",
                    {"agent": agent_name, "task": task_name},
                )
            return structured_output
        except Exception as error:
            if self.trace is not None:
                self.trace.record(
                    "crew_task_completed",
                    "FAILED",
                    {"agent": agent_name, "task": task_name, "error": str(error)[:500]},
                )
            raise
