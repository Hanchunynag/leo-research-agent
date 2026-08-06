# 目标架构

## 冻结原则

目标系统只有一个 `Unified Knowledge Service`。它是 Agent、Web/CLI Workflow 与 BM25/Dense/Graph 后端之间的唯一边界；上层不得导入 Qdrant、Neo4j、FTS 或具体 RAG runtime。

```text
CLI / Web API / Research Workflow
              |
       ResearchHarness
       |      |       |
 ToolGateway ContextBuilder Evaluation/Recovery
       |      |
 EvidenceIntelligenceService
       |
 Unified Knowledge Service (KnowledgeEngine contract)
       |
  backend adapters: legacy BM25/Dense, legacy GraphRAG, future implementations
       |
 Canonical Corpus + Index Generations
```

## 边界职责

### Canonical Corpus

`paper.json` 及派生的 Section/Chunk 必须成为 PDF、Section、Chunk、Block、Citation locator 的唯一事实源。索引只保存可重建投影，不得反向覆盖 canonical 内容。现有事实生成点是 `parse_paper`、`build_structure` 和 `build_chunks`；阶段二先收口读取接口，不改其文件格式。

### Research Workspace

Workspace 是检索授权和课题隔离边界。所有新 `EvidenceRequest` 强制携带 `workspace_id` 和正整数 `scope_version`。`scope_version` 对应一次确定的 WorkspaceDocument 集合；运行中不得悄悄读取更新后的集合。阶段一只建立内存契约，不新增数据库表。

### Unified Knowledge Service

对上提供 `KnowledgeEngine.retrieve(EvidenceRequest)`。它负责解析 workspace scope、路由后端、汇总诊断和返回 `CandidateEvidence`；具体 Adapter 才能访问旧 `RetrievalRuntime`、`GraphRAGRetrievalRuntime`、Qdrant、Neo4j 或 FTS。

### Evidence Intelligence

结果必须单向流动：

1. `CandidateEvidence`：召回/融合结果，允许缺少完整 locator，但不得直接给生成模型。
2. `VerifiedEvidence`：已回查 Canonical Corpus，确认 document/chunk/page/block/content hash。
3. `SelectedEvidence`：基于覆盖、直接性、多样性与 token budget 选入 Context。

失败候选记录在 `VerifiedEvidenceBundle.rejected_candidate_ids` 和 issues 中。当前 `LegacyEvidenceMapper` 只用旧 locator 字段做结构验证，因此验证方法明确标为 `legacy_locator_fields`，不冒充 canonical 内容回查。

### Research Harness

Harness 只感知 Workflow、Step/Stage、Tool、Context、Budget、State、Evaluation 和 Recovery。`is_graphrag`、Qdrant collection、Neo4j path、FTS route 等都必须由 Knowledge Service/Adapter 消化。Harness 通过 `ToolGateway` 触发工具，通过 `ContextBuilder` 获取 `ContextPack`。

### Index Generation

跨后端索引更新以 `IndexGeneration` 为领域概念。当前 GraphRAG `index_epochs` 可由 Adapter 投影为它；旧 BM25/Dense manifest 暂时投影为 legacy generation。只有全部要求的投影完成并校验后才激活新 generation；失败继续读旧 generation。

## 阶段一落点

- [`app/contracts/domain.py`](../../app/contracts/domain.py)：纯领域 dataclass，不依赖数据库或后端。
- [`app/contracts/protocols.py`](../../app/contracts/protocols.py)：五个稳定 Protocol。
- [`app/contracts/adapters.py`](../../app/contracts/adapters.py)：旧检索、Evidence dict、Agent runtime 兼容 Adapter。
- 旧业务入口保持不变；没有把 Web/CLI 切换到新接口，也没有数据库 migration。

## 非目标

阶段一不接入 LightRAG，不删除 GraphRAG，不改前端，不修改现有 SQLite/Neo4j schema，不进行全仓命名重写，也不改变任何现有 API 路由或响应。

## 阶段二实现状态

目标正式知识引擎已由 [`LightRAGKnowledgeEngine`](../../app/knowledge_engine/lightrag_engine.py) 实现，统一边界是 [`UnifiedKnowledgeService`](../../app/knowledge_engine/unified_service.py)。Canonical 与 scope 分别由 [`CanonicalCorpusService`](../../app/corpus/service.py) 和 [`WorkspaceService`](../../app/workspaces/service.py) 强制执行；Evidence 流程由 [`EvidenceIntelligencePipeline`](../../app/evidence/service.py) 收口。

当前 LightRAG `active` generation 仍是 shadow active，不是 answer-serving active。正式回答继续走 legacy Adapter；这一保守状态由真实 Q001 nDCG 仍低于 legacy 的结果支持，详见 [`stage2_shadow_comparison.md`](stage2_shadow_comparison.md)。旧 Dense/GraphRAG 全部保留，Harness 和前端未改写。
