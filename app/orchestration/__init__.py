"""Top-level Scholar orchestration backends.

The package deliberately keeps CrewAI at the application boundary.  Existing
RAG, writing, review, approval and session services remain the domain owners.
"""

from app.orchestration.contracts import (
    AgentStatus,
    DomainResultReference,
    ManagerDecision,
    RunBudget,
    RunGoal,
    RunState,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationStatus,
    ResearchAgentOutput,
    ReviewAgentOutput,
    Route,
    WriterAgentOutput,
)
from app.orchestration.evaluation import (
    OrchestrationEvaluationCase,
    OrchestrationEvaluationReport,
    default_orchestration_cases,
    evaluate_backend,
)
from app.orchestration.manager import (
    DecisionValidationResult,
    ManagerDecisionRejected,
    ManagerDecisionValidator,
)
from app.orchestration.service import ScholarOrchestrationService

__all__ = [
    "AgentStatus",
    "DomainResultReference",
    "ManagerDecision",
    "ManagerDecisionRejected",
    "ManagerDecisionValidator",
    "DecisionValidationResult",
    "RunBudget",
    "RunGoal",
    "RunState",
    "OrchestrationRequest",
    "OrchestrationResult",
    "OrchestrationStatus",
    "OrchestrationEvaluationCase",
    "OrchestrationEvaluationReport",
    "ResearchAgentOutput",
    "ReviewAgentOutput",
    "Route",
    "ScholarOrchestrationService",
    "WriterAgentOutput",
    "default_orchestration_cases",
    "evaluate_backend",
]
