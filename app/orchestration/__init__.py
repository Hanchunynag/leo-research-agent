"""Top-level Scholar orchestration backends.

The package deliberately keeps CrewAI at the application boundary.  Existing
RAG, writing, review, approval and session services remain the domain owners.
"""

from app.orchestration.contracts import (
    AgentStatus,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationStatus,
    ResearchAgentOutput,
    ReviewAgentOutput,
    Route,
    SupervisorResult,
    WriterAgentOutput,
)
from app.orchestration.evaluation import (
    OrchestrationEvaluationCase,
    OrchestrationEvaluationReport,
    compare_reports,
    default_orchestration_cases,
    evaluate_backend,
)
from app.orchestration.service import ScholarOrchestrationService

__all__ = [
    "AgentStatus",
    "OrchestrationRequest",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestrationEvaluationCase",
    "OrchestrationEvaluationReport",
    "ResearchAgentOutput",
    "ReviewAgentOutput",
    "Route",
    "ScholarOrchestrationService",
    "SupervisorResult",
    "WriterAgentOutput",
    "compare_reports",
    "default_orchestration_cases",
    "evaluate_backend",
]
