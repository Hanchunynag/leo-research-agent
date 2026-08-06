# Stage 4 Real Cost Report

## 采集实现

`ResearchRunHarness.record_context` 分别记录 Router、Generator 和 Selected Evidence Token；`record_provider_usage` 保存 Provider 返回的输入、输出、缓存和首 Token 指标。`TieredClaimEvidenceValidator` 单独记录语义验证 Token。

`app.evaluation.costs.RealCostMetricsCollector` 输出：

- Router Context Token；
- Selected Evidence Token；
- Generator 输入/输出 Token；
- Semantic Validator Token；
- Retrieval Tool 与 LLM 调用数；
- 总延迟与首 Token 延迟；
- 缓存命中率；
- 可选货币成本；
- LightRAG 查询 Token 与入库 Token 的独立字段。

Collector 拒绝 `fixture` 和 `control_flow_fixture` 来源。LightRAG 入库 Token 不参与单次问答货币成本。

## 已完成实测

真实 Shadow 运行见 `tests/baselines/stage4_real_shadow_run.json`：

- 29 个查询；
- 51.106 秒；
- 28 次查询 LLM 调用；
- 18,395 输入 Token；
- 1,214 输出 Token；
- 29 次 Embedding 调用、85 条文本；
- Provider 未配置价格，因此货币成本为 `null`。

## 未完成

DirectQA、RelationReasoning、Relation+Gap、Explicit DeepResearch 和 Bootstrap 的真实在线 Answer Provider 五类基准尚未全部运行；首 Token 延迟和货币单价也未配置。`tests/test_stage4_real_cost_metrics.py` 只验证采集和分账逻辑，不作为生产成本数值。阶段三控制流 Token 快照继续保留在
`tests/baselines/stage3_workflow_cost_baseline.json`。
