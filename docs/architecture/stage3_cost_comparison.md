# 阶段三运行成本对比

机器可读快照是 [`tests/baselines/stage3_workflow_cost_baseline.json`](../../tests/baselines/stage3_workflow_cost_baseline.json)，由阶段三测试逐字段锁定。

| 路径 | LLM calls | Tool calls | 检索轮数 | 外部搜索 | Repair | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Agent 历史 | 待确认 | 待确认 | 可多轮 | 可选 | 可调用 LLM | 待确认 |
| 新 DirectQA 确定性夹具 | 1 | 2 | 1 | 0 | 0 | 110 |
| 新 Relation（触发 gap）夹具 | 1 | 3 | 2 | 0 | 0 | 116 |

夹具 Generator 固定报告 20 tokens；Direct 的 Context 为 90 tokens，Relation 为 96 tokens。数值用于检测控制流和计数退化，不代表真实论文问题的 Token 长度或模型价格。

阶段一 [`behavior_baseline.md`](behavior_baseline.md) 明确记录旧 `AgenticRAGService` 没有统一累计 Router、Coverage、Generation、Validation、Repair 等调用，因此旧值是 `null`，不能写成 0。结构上，旧路径允许路由判定、Query Expansion、Coverage、Generation、Semantic Validation 和 Repair；新 DirectQA 默认只保留一次主要 Generation，引用验证为确定性逻辑。

真实 Provider 延迟、缓存 Token、货币成本与 DeepResearch 外部搜索价格仍待在线基准。当前正式知识回答仍由 Unified Service 内部 legacy official engine 提供，阶段二 LightRAG 只做 shadow；其入库 439,569 tokens 与两条 cold query 1,433 tokens 见 [`stage2_shadow_comparison.md`](stage2_shadow_comparison.md)，不应混入阶段三 Generator 的 Run Token。
