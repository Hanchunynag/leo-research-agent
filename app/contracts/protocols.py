"""统一知识服务和 Research Harness 的稳定 Protocol。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from app.contracts.domain import (
    AgentRun,
    CandidateEvidence,
    ContextPack,
    EvidenceRequest,
    IndexGeneration,
    IndexProfile,
    Document,
    SelectedEvidence,
    VerifiedEvidenceBundle,
)


@runtime_checkable
class KnowledgeEngine(Protocol):
    """唯一允许上层发起知识检索的后端无关入口。"""

    def retrieve(self, request: EvidenceRequest) -> Sequence[CandidateEvidence]: ...

    def retrieve_candidates(
        self, request: EvidenceRequest
    ) -> Sequence[CandidateEvidence]: ...

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]: ...

    def update_documents(
        self,
        documents: Sequence[Document],
        *,
        generation: IndexGeneration,
        profile: IndexProfile,
    ) -> Mapping[str, Any]: ...

    def delete_documents(
        self,
        document_ids: Sequence[str],
        *,
        generation: IndexGeneration,
    ) -> Mapping[str, Any]: ...

    def get_status(self) -> Mapping[str, Any]: ...


@runtime_checkable
class EvidenceIntelligenceService(Protocol):
    """负责 Candidate -> Verified -> Selected 的证据治理。"""

    def verify(
        self,
        request: EvidenceRequest,
        candidates: Sequence[CandidateEvidence],
    ) -> VerifiedEvidenceBundle: ...

    def select(
        self,
        request: EvidenceRequest,
        bundle: VerifiedEvidenceBundle,
    ) -> Sequence[SelectedEvidence]: ...


@runtime_checkable
class ContextBuilder(Protocol):
    def build(
        self,
        request: EvidenceRequest,
        evidence: Sequence[SelectedEvidence],
        *,
        token_budget: int,
    ) -> ContextPack: ...


@runtime_checkable
class ToolGateway(Protocol):
    """Agent 调用工具的唯一门面；参数中不得出现具体存储客户端。"""

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class ResearchHarness(Protocol):
    """只编排请求、Context、Budget、State、Evaluation 和 Recovery。"""

    def run(
        self,
        request: EvidenceRequest,
        *,
        context_pack: ContextPack | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> AgentRun: ...
