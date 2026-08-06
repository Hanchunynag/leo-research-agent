"""阶段一冻结的统一知识服务领域契约。

这些类型只表达跨模块语义，不依赖 Qdrant、Neo4j、FTS、GraphRAG 或
Agentic Service。阶段一不要求现有业务对象立即迁移到这些类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


RunState = Literal["created", "running", "completed", "failed", "cancelled"]
GenerationState = Literal[
    "pending",
    "building",
    "validating",
    "active",
    "retired",
    "failed",
    # 阶段一兼容值；新代码不再创建 superseded generation。
    "superseded",
]
EvidenceState = Literal[
    "retrieved",
    "filtered",
    "ranked",
    "backfilled",
    "verified",
    "selected",
    "rejected",
]
EvidenceDirectness = Literal["direct", "indirect", "inferred"]
EvidenceGrade = Literal[
    "primary",
    "candidate",
    "analogy",
    "graph_inference",
]


def _required(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空。")


def _positive(value: int, field_name: str) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} 必须为正整数。")


@dataclass(frozen=True, slots=True)
class Work:
    """可由多个 PDF 版本共同指向的逻辑学术作品。"""

    work_id: str
    title: str
    document_ids: tuple[str, ...] = ()
    doi: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.work_id, "work_id")
        _required(self.title, "title")


@dataclass(frozen=True, slots=True)
class Document:
    """Canonical Corpus 中一个具体 PDF 文档版本。"""

    document_id: str
    paper_id: str
    work_id: str | None
    source_sha256: str
    canonical_path: str
    section_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "document_id",
            "paper_id",
            "source_sha256",
            "canonical_path",
        ):
            _required(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ResearchWorkspace:
    """带单调 scope_version 的课题隔离边界。"""

    workspace_id: str
    scope_version: int
    name: str
    description: str = ""
    direction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.workspace_id, "workspace_id")
        _required(self.name, "name")
        _positive(self.scope_version, "scope_version")


@dataclass(frozen=True, slots=True)
class ResearchScope:
    """Workspace 某一版本的不可变检索边界。"""

    workspace_id: str
    scope_version: int
    included_document_ids: tuple[str, ...] = ()
    excluded_document_ids: tuple[str, ...] = ()
    included_direction_ids: tuple[str, ...] = ()
    excluded_direction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.workspace_id, "workspace_id")
        _positive(self.scope_version, "scope_version")
        if set(self.included_document_ids) & set(self.excluded_document_ids):
            raise ValueError("同一 document_id 不能同时 included 和 excluded。")
        if set(self.included_direction_ids) & set(self.excluded_direction_ids):
            raise ValueError("同一 direction_id 不能同时 included 和 excluded。")


@dataclass(frozen=True, slots=True)
class WorkspaceDocument:
    """Workspace 与 Canonical Document 的有版本成员关系。"""

    workspace_id: str
    document_id: str
    added_in_scope_version: int
    removed_in_scope_version: int | None = None
    tags: tuple[str, ...] = ()
    evidence_grade: EvidenceGrade = "primary"

    def __post_init__(self) -> None:
        _required(self.workspace_id, "workspace_id")
        _required(self.document_id, "document_id")
        _positive(self.added_in_scope_version, "added_in_scope_version")
        if self.removed_in_scope_version is not None:
            _positive(self.removed_in_scope_version, "removed_in_scope_version")
            if self.removed_in_scope_version <= self.added_in_scope_version:
                raise ValueError("removed_in_scope_version 必须大于 added_in_scope_version。")


@dataclass(frozen=True, slots=True)
class ResearchDirection:
    """Workspace 内可独立规划和评估的研究方向。"""

    direction_id: str
    workspace_id: str
    title: str
    objective: str
    constraints: tuple[str, ...] = ()
    excluded: bool = False

    def __post_init__(self) -> None:
        for name in ("direction_id", "workspace_id", "title", "objective"):
            _required(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """所有新检索入口必须接收的显式 Workspace 请求。"""

    request_id: str
    workspace_id: str
    scope_version: int
    query: str
    top_k: int = 10
    work_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    direction_id: str | None = None
    retrieval_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("request_id", "workspace_id", "query"):
            _required(getattr(self, name), name)
        _positive(self.scope_version, "scope_version")
        _positive(self.top_k, "top_k")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """后端召回和融合后、尚未通过来源校验的候选证据。"""

    candidate_id: str
    request_id: str
    workspace_id: str
    scope_version: int
    content: str
    score: float
    retrieval_source: str
    work_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    section_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    block_ids: tuple[str, ...] = ()
    relation_path: tuple[str, ...] = ()
    state: EvidenceState = "retrieved"
    evidence_grade: EvidenceGrade = "candidate"
    directness: EvidenceDirectness | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "request_id",
            "workspace_id",
            "content",
            "retrieval_source",
        ):
            _required(getattr(self, name), name)
        _positive(self.scope_version, "scope_version")

    @property
    def evidence_id(self) -> str:
        """阶段二名称；candidate_id 保留以兼容阶段一调用方。"""

        return self.candidate_id

    @property
    def source_type(self) -> str:
        return self.retrieval_source

    @property
    def retrieval_score(self) -> float:
        return self.score


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    """已确认能映射回 Canonical Document/Chunk/Block 的证据。"""

    evidence_id: str
    candidate_id: str
    request_id: str
    workspace_id: str
    scope_version: int
    content: str
    work_id: str
    document_id: str
    chunk_id: str
    page_start: int
    page_end: int
    block_ids: tuple[str, ...]
    verification_method: str
    content_hash: str
    relation_path: tuple[str, ...] = ()
    state: EvidenceState = "verified"
    evidence_grade: EvidenceGrade = "primary"
    directness: EvidenceDirectness = "direct"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "candidate_id",
            "request_id",
            "workspace_id",
            "content",
            "work_id",
            "document_id",
            "chunk_id",
            "verification_method",
            "content_hash",
        ):
            _required(getattr(self, name), name)
        _positive(self.scope_version, "scope_version")
        _positive(self.page_start, "page_start")
        _positive(self.page_end, "page_end")
        if self.page_end < self.page_start:
            raise ValueError("page_end 不能小于 page_start。")
        if not self.block_ids:
            raise ValueError("VerifiedEvidence.block_ids 不能为空。")


@dataclass(frozen=True, slots=True)
class SelectedEvidence:
    """经覆盖率、预算和多样性约束选入最终上下文的证据。"""

    evidence: VerifiedEvidence
    selection_rank: int
    selection_reason: str
    token_count: int

    def __post_init__(self) -> None:
        _positive(self.selection_rank, "selection_rank")
        _required(self.selection_reason, "selection_reason")
        if self.token_count < 0:
            raise ValueError("token_count 不能为负数。")


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceBundle:
    """一次请求的来源校验结果及可审计失败原因。"""

    request_id: str
    workspace_id: str
    scope_version: int
    evidence: tuple[VerifiedEvidence, ...]
    rejected_candidate_ids: tuple[str, ...] = ()
    verification_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _required(self.workspace_id, "workspace_id")
        _positive(self.scope_version, "scope_version")


@dataclass(frozen=True, slots=True)
class ContextPack:
    """Harness 可消费、与具体检索后端无关的最终上下文。"""

    context_id: str
    request_id: str
    workspace_id: str
    scope_version: int
    selected_evidence: tuple[SelectedEvidence, ...]
    rendered_context: str
    token_count: int
    token_budget: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("context_id", "request_id", "workspace_id"):
            _required(getattr(self, name), name)
        _positive(self.scope_version, "scope_version")
        if self.token_count < 0:
            raise ValueError("token_count 不能为负数。")
        _positive(self.token_budget, "token_budget")
        if self.token_count > self.token_budget:
            raise ValueError("ContextPack 超出 token_budget。")


@dataclass(frozen=True, slots=True)
class AgentRun:
    """ResearchHarness 的后端无关运行记录。"""

    run_id: str
    request_id: str
    workspace_id: str
    scope_version: int
    state: RunState
    answer: str = ""
    answerable: bool | None = None
    context_id: str | None = None
    llm_call_count: int | None = None
    token_usage: int | None = None
    elapsed_ms: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "request_id", "workspace_id"):
            _required(getattr(self, name), name)
        _positive(self.scope_version, "scope_version")
        if self.llm_call_count is not None and self.llm_call_count < 0:
            raise ValueError("llm_call_count 不能为负数。")
        if self.token_usage is not None and self.token_usage < 0:
            raise ValueError("token_usage 不能为负数。")


@dataclass(frozen=True, slots=True)
class IndexGeneration:
    """一次跨索引构建代次；不等同于任何具体后端的 collection。"""

    generation_id: str
    corpus_version: str
    scope_version: int
    state: GenerationState
    document_count: int
    chunk_count: int
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    previous_generation_id: str | None = None
    failure_type: str | None = None
    workspace_id: str = "default"
    index_profile_id: str = "default"
    created_at: str | None = None
    activated_at: str | None = None

    def __post_init__(self) -> None:
        _required(self.generation_id, "generation_id")
        _required(self.corpus_version, "corpus_version")
        _required(self.workspace_id, "workspace_id")
        _required(self.index_profile_id, "index_profile_id")
        _positive(self.scope_version, "scope_version")
        if self.document_count < 0 or self.chunk_count < 0:
            raise ValueError("索引计数不能为负数。")


@dataclass(frozen=True, slots=True)
class IndexProfile:
    """索引和查询阶段共同引用的显式配置快照。"""

    profile_id: str
    engine: str = "lightrag"
    query_mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int = 10
    chunk_top_k: int = 20
    max_entity_tokens: int = 4_000
    max_relation_tokens: int = 4_000
    max_total_tokens: int = 16_000
    embedding_model: str | None = None
    llm_model: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.profile_id, "profile_id")
        _required(self.engine, "engine")
        for name in (
            "top_k",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
        ):
            _positive(getattr(self, name), name)
