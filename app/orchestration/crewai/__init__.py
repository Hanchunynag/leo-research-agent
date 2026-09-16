"""CrewAI adapter layer for Scholar orchestration.

The public names are lazy so importing the optional backend does not eagerly
load CrewAI's LLM/event modules or mutate the host process environment.
"""

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def _without_process_dotenv() -> Iterator[None]:
    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        dotenv.load_dotenv = original


def __getattr__(name: str) -> Any:
    with _without_process_dotenv():
        if name == "ScholarCrew":
            from app.orchestration.crewai.crew import ScholarCrew

            return ScholarCrew
        if name in {"CrewAIOrchestrationFlow", "FlowState"}:
            from app.orchestration.crewai.flow import CrewAIOrchestrationFlow, FlowState

            return {
                "CrewAIOrchestrationFlow": CrewAIOrchestrationFlow,
                "FlowState": FlowState,
            }[name]
        if name == "CrewAIProviderAdapter":
            from app.orchestration.crewai.models import CrewAIProviderAdapter

            return CrewAIProviderAdapter
        if name in {
            "CapabilityMatrix",
            "ResearchCapabilityTool",
            "ReviewerCapabilityTool",
            "WriterCapabilityTool",
        }:
            from app.orchestration.crewai.tools import (
                CapabilityMatrix,
                ResearchCapabilityTool,
                ReviewerCapabilityTool,
                WriterCapabilityTool,
            )

            return {
                "CapabilityMatrix": CapabilityMatrix,
                "ResearchCapabilityTool": ResearchCapabilityTool,
                "ReviewerCapabilityTool": ReviewerCapabilityTool,
                "WriterCapabilityTool": WriterCapabilityTool,
            }[name]
    raise AttributeError(name)

__all__ = [
    "CapabilityMatrix",
    "CrewAIOrchestrationFlow",
    "CrewAIProviderAdapter",
    "FlowState",
    "ResearchCapabilityTool",
    "ReviewerCapabilityTool",
    "ScholarCrew",
    "WriterCapabilityTool",
]
