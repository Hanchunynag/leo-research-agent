# Stage 4 Bootstrap Provider

## 生产链路

`app.research.providers.AcademicLiteratureProviderAdapter` 适配现有
`app.academic_mcp.service.AcademicDiscoveryService`。后者使用
`CrossrefProvider`、`OpenAlexProvider` 和 `ArxivProvider`。Workflow 只看到 Tool Gateway 中的：

- `literature.search`
- `literature.get_metadata`
- `literature.download`
- `document.parse`
- `job.get_status`

搜索最多返回 10 条；`ResearchBootstrapWorkflow` 最多筛选 5 条、轻量读取 2 条并正式处理 1 条。`CandidateKnowledgeRepository` 将外部元数据写入
`data/candidate_knowledge/papers.json`，不写 Canonical Corpus。

下载和解析由 `build_bootstrap_provider_composition` 提交到持久化 Job。
`CanonicalDocumentParseProviderAdapter` 仅通过
`app.parsing.pipeline.parse_paper` 进入 Canonical Corpus。未确认时 Workflow 只返回 Scope Proposal；确认后才调用 `workspace.update_scope`，且该 Tool 使用 compare-and-set 的 `scope_version`。

当 Job 尚未完成时，`ResearchBootstrapWorkflow._resolve_job_result` 返回
`pending_job_id` 和 `pending_stage`。同一请求再次执行时，幂等 Job 会继续下载、解析、更新 Scope，最后进入 `RESUME_ORIGINAL_QUESTION`。

## 组合位置

- Web：`LocalRAGWebRuntime._ensure_bootstrap_providers` 启动持久化 Bootstrap Worker；
- CLI：`agentic_service_from_args` 注入相同 Tool，`main.py jobs work` 执行持久化任务；
- 测试：`tests/test_stage4_bootstrap_e2e.py` 覆盖空 Workspace、无结果、超时、下载/解析失败、未确认、确认入库、Scope 冲突和原问题恢复。

待确认：测试使用可控 Provider 夹具验证真实 Adapter 链路，但本阶段尚未在受控网络环境完成一次 Crossref/OpenAlex/arXiv 下载到 MinerU 的在线端到端验收。
