# Phase 3A Framework Migration

## Dependency boundary

主工程直接使用稳定的 LangChain/LangGraph 1.x 生态：`langchain==1.4.0`、`langchain-core==1.6.3`、`langgraph==1.2.11`、`langgraph-checkpoint==4.2.0`、`langgraph-checkpoint-sqlite==3.1.1`、`deepagents==0.7.13`。项目不再保留 0.3 生产兼容分支。

`OpenAICompatibleChatModel` 和 `ScholarChatModelAdapter` 只负责 Provider、Message、Tool Call、Async 和 Stream 转换。LangChain 类型停留在 `app/langchain_agent/` 和 Harness 边界，不进入 `ResearchRequest`、`EvidencePack`、`WritingRequest`、`DraftPatch` 等领域 Contract。

## Graph and checkpoint

既有 Research Graph 使用 LangGraph 1.x 的 `StateGraph`/compiled graph、`RunnableConfig.configurable.thread_id` 和 `Command` resume。`open_checkpointer("memory")` 用于快速测试，`open_checkpointer("sqlite")` 使用生命周期受控的官方 `SqliteSaver`。`session_id`、`run_id`、`thread_id`、`trace_id` 和 `job_id` 仍各自独立。

Deep Agents 的 Checkpoint 只保存 Harness Working State。`ScholarHarnessService.resume(...)` 是框架恢复边界；它不接受或执行 Patch Approval，Human Accept/Reject/Apply 仍只能经现有 Approval Service/API/CLI/VS Code 完成。

## Harness boundary

Deep Agents 只提供用户级路由、计划、渐进式 Skill 文档加载和 Context Isolation。它通过三类高层路径工作：主 Agent 执行已选择的 Skill，Research Subagent 只调用 `ResearchCapabilityService`，Reviewer Subagent 只调用共享 Reviewer Adapter。`write-introduction`、`support-claim`、`write-conclusion`、`write-abstract` 仍由 `SkillRegistry` 和 `CapabilityProfile` 定义；没有 IntroductionAgent、CitationAgent 或新的 Research Pipeline。

Builtin filesystem 被限制为 `skills/` 下的只读 `read_file`；写文件、Shell、任意 HTTP、底层检索、Facts/Contributions 修改和 Approval 工具不在 Agent Tool Visibility 中。Conclusion/Abstract 的 Profile 不创建 Research Subagent。

## Source of truth

Deep Agent Memory/Checkpoint 是临时 Harness 状态；SessionRuntime、Project DB、Manuscript Facts、Contribution Registry、Evidence/Citation Registry 和 DraftPatch Store 仍是业务事实来源。外层 Harness 可调用底层成熟 Research Graph，但不能重写 BM25、BGE-M3、Qdrant、Evidence Validation、Citation Lifecycle 或 Human Approval。
