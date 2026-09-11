# ScholarHarness 当前实现边界

本文只描述当前已实现的最小基础，不把未来 Supervisor、Research Agent、Reviewer Agent、Writing Skill 或 Deep Agents 描述为已完成能力。

当前已实现：

- `app/session/`：Session Catalog、Session-per-DB、Conversation Message、Agent Run metadata、结果 Projection 和单机 Session Lock。
- `app/application/research_facade.py`：`ResearchApplicationFacade.research_topic(...)`，通过注入的现有 Agent Service 执行 Research Engine；CLI Agentic answer 和 Web `/api/answers` 已接入该 Facade。
- `app/scholar/`：Manuscript State Contract、read-on-request LaTeX hash 同步、base-hash Patch Guard、Manuscript Facts 和 Contribution Registry。
- `app/langchain_agent/checkpoint_factory.py`：官方 `SqliteSaver` Factory，使用已验证的 `langgraph-checkpoint-sqlite==2.0.0`，同时保留 `InMemorySaver` 模式。
- Application correlation：`run_id`、`trace_id`、`thread_id`、`job_id` 和 `project_id` 作为独立字段保存和传递。
- `LegacySessionAdapter`：旧 Session 首次由新 Facade 打开时只读迁移可安全映射的 Conversation Message；新请求不向旧 Store 写入等价业务状态。

当前未实现：

- LangGraphResearchRuntime 的生产 Checkpointer 全量切换和 Process Restart Resume 集成；当前已完成官方 Factory 和独立 Spike。
- Deep Agents、Manuscript Supervisor、Research Agent、Reviewer Agent。
- Introduction 或其他 Writing Skill。
- VS Code / LaTeX Workshop Bridge、旧 Evidence 的完整迁移和 Web Session 管理命令切换。

约束：Research Engine、RAG、Evidence/Citation Pipeline、Qdrant、BM25、BGE-M3 和 Reranker 仍由现有模块负责；ScholarHarness 只能通过 Application Facade 或稳定 Research Contract 访问它们。
