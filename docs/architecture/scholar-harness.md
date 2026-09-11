# ScholarHarness 当前实现边界

本文只描述当前已实现的最小基础，不把未来 Supervisor、Research Agent、Reviewer Agent、Writing Skill 或 Deep Agents 描述为已完成能力。

当前已实现：

- `app/session/`：Session Catalog、Session-per-DB、Conversation Message、Agent Run metadata、结果 Projection 和单机 Session Lock。
- `app/application/research_facade.py`：`ResearchApplicationFacade.research_topic(...)`，通过注入的现有 Agent Service 执行 Research Engine。
- `app/scholar/`：Manuscript State Contract、read-on-request LaTeX hash 同步、base-hash Patch Guard、Manuscript Facts 和 Contribution Registry。

当前未实现：

- 官方兼容的 LangGraph Persistent Checkpointer 和 Process Restart Resume。
- Deep Agents、Manuscript Supervisor、Research Agent、Reviewer Agent。
- Introduction 或其他 Writing Skill。
- VS Code / LaTeX Workshop Bridge、Web/CLI Facade 全量切换和 Legacy Store Migration。

约束：Research Engine、RAG、Evidence/Citation Pipeline、Qdrant、BM25、BGE-M3 和 Reranker 仍由现有模块负责；ScholarHarness 只能通过 Application Facade 或稳定 Research Contract 访问它们。
