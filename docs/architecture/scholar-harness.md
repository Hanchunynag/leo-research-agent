# ScholarHarness 当前实现边界

本文描述当前 Phase 3A 的 Scholar Harness 边界。Deep Agents 已接入进程内，但仍只是 Harness；Scholar Domain Contract、Research Capability、Writing Runtime、Citation Lifecycle 和 Human Approval 仍由既有领域服务拥有。

当前已实现：

- `app/session/`：Session Catalog、Session-per-DB、Conversation Message、Agent Run metadata、结果 Projection 和单机 Session Lock。
- `app/application/research_facade.py`：`ResearchApplicationFacade.research_topic(...)`，通过注入的现有 Agent Service 执行 Research Engine；CLI Agentic answer 和 Web `/api/answers` 已接入该 Facade。
- `app/scholar/`：Manuscript State Contract、read-on-request LaTeX hash 同步、base-hash Patch Guard、Manuscript Facts 和 Contribution Registry。
- `app/langchain_agent/checkpoint_factory.py`：官方 LangGraph 1.x `SqliteSaver` Factory（当前锁定 `langgraph-checkpoint-sqlite==3.1.1`），同时保留 `InMemorySaver` 测试模式。
- Application correlation：`run_id`、`trace_id`、`thread_id`、`job_id` 和 `project_id` 作为独立字段保存和传递。
- `LegacySessionAdapter`：旧 Session 首次由新 Facade 打开时只读迁移可安全映射的 Conversation Message；新请求不向旧 Store 写入等价业务状态。
- Citation Lifecycle：Project bibliography sync、Citation Identity/BibKey binding、BibEntry proposal 和受 Human Approval 保护的 `.bib` change。

- `app/scholar/harness.py`：进程内 Deep Agents Harness、显式 Skill 路由、渐进式 Skill Loading、隔离的 Research/Reviewer Subagent、高层 Domain Tool 与 Checkpoint resume。

明确不属于 Harness 权限的能力：任意文件系统、Shell/HTTP、Qdrant/底层数据库、Facts/Contributions 写入、`.tex/.bib` 直接写入、Patch Accept/Reject/Apply。

## Research Capability Layer（Phase 1C）

当前已实现 `app.scholar.research.ResearchCapabilityService`，作为 Scholar 上层未来调用本地
文献能力的稳定边界。它提供 `search_papers()`、`search_sections()` 和 `read_evidence()`，
复用 UnifiedKnowledgeService 的既有 hierarchical retrieval、CanonicalCorpusService 的
canonical locator 以及 EvidenceIntelligencePipeline 的验证/选择，不新增 RAG 实现。正式的
`ResearchRequest`、Budget、Paper/Section Candidate、EvidenceRef/Source 和错误模型见
[`research-capability-contract.md`](research-capability-contract.md)。Phase 1C 的基线只使用本地
Knowledge；当前 Phase 2D 的受控 Web 扩展见 [`web-literature-freshness.md`](web-literature-freshness.md)。

## Introduction Vertical（Phase 2A）

当前已实现 framework-agnostic `ScholarWritingService` / `ManuscriptSupervisor`，仅支持
Introduction 的 WRITE/REVISE。它读取最新 Introduction hash、Manuscript Facts 和用户确认的
Contribution，通过 `write-introduction` Skill 拆分 ResearchNeed，委托现有
`ResearchCapabilityService`，构造 ClaimPlan 和 Evidence-first SectionDraft，执行最多两轮
结构化 Reviewer，并返回未应用的 DraftPatch。它不会调用 `apply_patch()`，不会新增确认
Contribution，也不会写入 Fact。Introduction 的外部文献仍必须经由 ResearchCapabilityService
的 Freshness Policy 和 Evidence Contract；Skill 不直接访问 Web。详细边界见
[`writing-vertical.md`](writing-vertical.md)。

## Human Approval & LaTeX Loop（Phase 2B）

当前已实现 Project-owned Patch persistence、`PatchApprovalService` 和最小 Build Bridge：

- `DraftPatch` 内容 immutable，写入当前 Project 的 `.scholar/project.db`，状态从
  `AWAITING_APPROVAL` 开始；Session/Run 只保存 source correlation。
- Accept/Reject 必须由 human-facing client 提交明确的 `patch_id`、`project_id`、
  `expected_base_hash` 和 `actor`。没有 `force`、`skip_approval` 或客户端 supplied content
  入口；Agent capability 不包含审批权限。
- Apply 前复读真实 `.tex` 并通过 base hash 和 `original_content` Guard，使用现有
  `ManuscriptSynchronizer` 的临时文件 + fsync + atomic replace。Apply 后重新扫描文件，保存真实
  hash、版本和审计记录；冲突不覆盖用户编辑，重复 Apply 幂等。
- Backend 只保存/返回 `BUILD_TRIGGERED`、`SUCCESS`、`FAILED` 等 Build 状态；实际编译由
  VS Code LaTeX Workshop command 执行，Backend 不实现 `latexmk`/`pdflatex` fallback。

完整 Patch lifecycle、HTTP/CLI/VS Code 边界和 Diagnostics 语义见
[`human-approval-latex-loop.md`](human-approval-latex-loop.md)。

## Skill Runtime（Phase 2C）

当前已增加共享 `SkillRegistry`、`TaskRouter`、`CapabilityProfile`、`SkillExecutionContext`
和 `SkillResult`。`support-claim` 返回 `ClaimSupportResult`，可在其 Profile 允许时经统一
Research Runtime 使用受控 Web；Conclusion 和
Abstract 复用同一个 Manuscript Synthesis Runtime、Reviewer、DraftPatch 和 Human Approval，
并关闭 Research/Citation。具体 Profile、FactSet、stale policy 和 Result boundary 见
[`skill-runtime.md`](skill-runtime.md)。

约束：Research Engine、RAG、Evidence/Citation Pipeline、Qdrant、BM25、BGE-M3 和 Reranker 仍由现有模块负责；ScholarHarness 只能通过 Application Facade 或稳定 Research Contract 访问它们。

## Web Literature & Freshness（Phase 2D）

当前已实现 `FreshnessPolicy`、`WebLiteratureAdapter`、Discovery/Metadata
Cache 和 External Evidence Contract。`LOCAL_ONLY`、`LOCAL_FIRST`、
`FRESH_REQUIRED` 的决定在 Research Runtime 中完成；Web Provider 只返回
Discovery Candidate，不能直接生成 Citation。Canonical dedup 优先 DOI、
versionless arXiv ID 和已有 identity 基础设施；命中本地 Work 时复用本地
全文，不写入 Workspace Scope 或 Canonical Corpus。完整的 Candidate →
Verified Evidence、外部 locator、缓存、失败闭环和 Harness Trace 说明见
[`web-literature-freshness.md`](web-literature-freshness.md)。

## Citation Lifecycle（Phase 2E）

当前已实现 `CitationResolutionService` 和 `BibliographySynchronizer`。实际
`references.bib` 是唯一 bibliography authority；Project DB 只保存 Citation Registry
与 External Evidence audit projection。Local/Web Verified Evidence 先解析为
`CitationIdentity`，再复用现有 BibKey 或生成待审批 `BibEntryCandidate`。写作 Patch
携带 bibliography hash、CitationBinding 与 BibliographyChange，只有 Human ACCEPT 的
既有 Approval Service 才能按序安全写入 `.bib` 和 `.tex`。详情见
[`citation-lifecycle.md`](citation-lifecycle.md)。

## Framework Runtime（Phase 3A）

依赖锁定为 LangChain `1.4.0`、LangChain Core `1.6.3`、LangGraph `1.2.11`、Checkpoint `4.2.0`、SQLite Checkpoint `3.1.1` 和 Deep Agents `0.7.13`。`app/langchain_agent/provider.py` 是唯一 Provider/Message/Tool Binding 适配层；既有 Research Graph 继续使用 LangGraph 1.x，既有 `ResearchAgentState` 和 Domain Contract 不携带框架类型。

`ScholarHarnessService.scholar_request(...)` 返回 `ScholarHarnessResult`，而不是 Message List。Deep Agent 只选择 Skill、读取最小 Context、委托既有 Research/Reviewer 能力并返回 `DraftPatch` 或 `ClaimSupportResult`。`ScholarHarnessService.resume(...)` 恢复同一 `thread_id` 的 Working State；Checkpointer 不是 Session、Project、Evidence 或 Patch Store。

## Production Scholar Runtime（Phase 3B）

`ScholarRuntimeFactory` 是唯一完整的生产组合根。Web 和 CLI 的 Scholar-level
请求都通过同一个 Factory 创建的 runtime-scoped `ScholarHarnessService` 进入
`scholar_request()`、`resume()` 或只读 `status()`；Factory 只组装依赖，不复制
TaskRouter、Research Planner 或 Writing 业务决策。测试可以显式 override domain
components，生产路径不能由 Factory 静默降级为 `InMemorySaver`。

运行模式、资源关闭、Session DB/Checkpoint/Project DB 的权威边界、orphan run
启动恢复、Context Budget、Harness Evaluation 和稳定 termination reason 见
[`production-scholar-runtime.md`](production-scholar-runtime.md)。
