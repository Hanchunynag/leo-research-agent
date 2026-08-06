"""阶段三：后端无关 Research Harness、Context 与工作流。"""

from app.research.adapters import AgenticReasoningGeneratorAdapter
from app.research.context import PhaseContextPack, ResearchContextManager
from app.research.coverage import (
    DimensionCoverageAnalyzer,
    DimensionCoverageReport,
    QueryFrame,
    QueryFrameBuilder,
)
from app.research.harness import (
    BudgetExceeded,
    HarnessState,
    RecoveryLevel,
    ResearchBudgetPolicy,
    ResearchRunHarness,
    RunTraceStore,
)
from app.research.memory import ResearchStateStore
from app.research.providers import (
    AcademicLiteratureProviderAdapter,
    BootstrapProviderComposition,
    CandidateKnowledgeRepository,
    CanonicalDocumentParseProviderAdapter,
    build_bootstrap_provider_composition,
)
from app.research.runtime import (
    HarnessAgentService,
    ResearchRuntime,
    WorkflowRequest,
    build_harness_agent_service,
    build_research_runtime,
    unified_tool_handlers,
)
from app.research.tools import ToolGatewayRegistry, ToolSpec, build_default_gateway
from app.research.validation import ClaimEvidenceValidator
from app.research.workflows import (
    DeepResearchWorkflow,
    DirectQAWorkflow,
    RelationReasoningWorkflow,
    ResearchBootstrapWorkflow,
    WorkflowName,
)

__all__ = [
    "BudgetExceeded",
    "AgenticReasoningGeneratorAdapter",
    "AcademicLiteratureProviderAdapter",
    "BootstrapProviderComposition",
    "CandidateKnowledgeRepository",
    "CanonicalDocumentParseProviderAdapter",
    "ClaimEvidenceValidator",
    "DeepResearchWorkflow",
    "DimensionCoverageAnalyzer",
    "DimensionCoverageReport",
    "DirectQAWorkflow",
    "HarnessAgentService",
    "HarnessState",
    "PhaseContextPack",
    "QueryFrame",
    "QueryFrameBuilder",
    "RecoveryLevel",
    "RelationReasoningWorkflow",
    "ResearchBootstrapWorkflow",
    "ResearchBudgetPolicy",
    "ResearchContextManager",
    "ResearchRunHarness",
    "ResearchRuntime",
    "ResearchStateStore",
    "RunTraceStore",
    "ToolGatewayRegistry",
    "ToolSpec",
    "WorkflowName",
    "WorkflowRequest",
    "build_default_gateway",
    "build_bootstrap_provider_composition",
    "build_harness_agent_service",
    "build_research_runtime",
    "unified_tool_handlers",
]
