# 阶段三工作流

四类 Workflow 都通过 [`BaseWorkflow`](../../app/research/workflows.py) 调用 Tool Gateway，通过 [`ResearchContextManager.generator_pack`](../../app/research/context.py) 接收 Selected Evidence。共同的 Claim 校验和安全提交由 [`ResearchRuntime.run`](../../app/research/runtime.py) 执行。

## DirectQAWorkflow

```text
SCOPE_CHECK -> RETRIEVE -> EVIDENCE_EVALUATE -> GENERATE
            -> DETERMINISTIC_CITATION_CHECK -> COMPLETE
```

[`DirectQAWorkflow.execute`](../../app/research/workflows.py) 只读一次 Scope、执行一次主要检索、最多执行一次 Generator 调用。Workflow 路由是 [`ResearchRuntime.route`](../../app/research/runtime.py) 的确定性规则，不默认调用 Planner、Query Expansion、Coverage LLM 或 Repair LLM。`DETERMINISTIC_CITATION_CHECK` 由顶层 `CLAIM_VALIDATE` Step 实现；修复只删除或收窄已有 Claim，不调用第二次生成。

## RelationReasoningWorkflow

```text
SCOPE_CHECK -> QUERY_FRAME -> RELATIONAL_RETRIEVE -> EVIDENCE_EVALUATE
            -> OPTIONAL_GAP_RETRIEVE (coverage 不足时最多一次)
            -> GENERATE -> CLAIM_VALIDATE -> COMPLETE
```

[`RelationReasoningWorkflow._sufficient`](../../app/research/workflows.py) 以至少两条证据且包含 direct/indirect 证据作为保守覆盖代理。第一次不足时允许一次限定补检索；`max_retrieval_rounds=2` 是硬上限。

## DeepResearchWorkflow

[`DeepResearchWorkflow.execute`](../../app/research/workflows.py) 只有 `WorkflowRequest.explicit_deep_research=True` 才运行，否则进入安全拒绝。当前实现执行问题与反例两路本地检索，证据不足时最多一次外部 `literature.search`，再用 Selected Evidence 生成一次答案。更细的多代理分解和外部候选正式入库没有隐式启用。

## ResearchBootstrapWorkflow

```text
KNOWLEDGE_READINESS_CHECK -> TOPIC_NORMALIZATION -> PROVISIONAL_SCOPE
-> LITERATURE_DISCOVERY -> CANDIDATE_SCREENING -> LIGHTWEIGHT_PARSE
-> SCOPE_PROPOSAL -> USER_CONFIRMATION | LIMITED_AUTONOMY
-> FORMAL_INGESTION -> SCOPE_VERSION_ADVANCE -> RESUME_ORIGINAL_QUESTION
```

[`ResearchBootstrapWorkflow.execute`](../../app/research/workflows.py) 最多接收 10 条搜索结果、筛选 5 条、读取 2 条轻量元数据、正式下载并解析 1 条。未确认时只返回 Scope proposal。确认后 `workspace.update_scope` 使用乐观 Scope 版本检查，Workflow 改用返回的新 `scope_version` 读取 Scope并恢复原问题，避免入库后仍查询旧快照。

生产 Web/CLI 尚未配置 `literature.*`、`document.parse` 的外部 provider，因此空 Workspace 自动触发 Bootstrap 时会 fail closed；测试通过注入 handler 验证了限量入库和原问题恢复。接入真实 provider 是已知后续任务，不允许绕过 Tool Gateway。
