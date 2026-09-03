"""LangChain 生产 Agent 入口。"""

from app.langchain_agent.service import (
    LangChainHarnessAgentService,
    build_langchain_agent_service,
)
from app.langchain_agent.skills import (
    AnswerGenerationSkill,
    BilingualRetrievalSkill,
    ClaimValidationSkill,
    ScopeReadSkill,
    TranslationSkill,
)
from app.langchain_agent.graph import (
    LLMActionDecider,
    LangGraphResearchRuntime,
    ReferenceResolver,
    ResearchAgentState,
    resolve_references,
)
from app.langchain_agent.planning import (
    COMPARISON_DIMENSIONS,
    build_research_plan,
    classify_research_task,
)
from app.langchain_agent.tools import build_research_tools

__all__ = [
    "LangChainHarnessAgentService",
    "AnswerGenerationSkill",
    "BilingualRetrievalSkill",
    "ClaimValidationSkill",
    "ScopeReadSkill",
    "TranslationSkill",
    "build_langchain_agent_service",
    "LangGraphResearchRuntime",
    "ResearchAgentState",
    "resolve_references",
    "ReferenceResolver",
    "LLMActionDecider",
    "build_research_tools",
    "COMPARISON_DIMENSIONS",
    "build_research_plan",
    "classify_research_task",
]
