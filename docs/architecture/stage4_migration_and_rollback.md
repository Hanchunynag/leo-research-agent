# Stage 4 Migration and Rollback

## 稳定点

阶段三稳定标签：`stage3-stable-20260806`。阶段四分支：
`stage4-production-reliability`。

阶段四提交：

1. `01ee34b` 评估数据集与 qrel Schema；
2. `58d5f46` Shadow Acceptance 与集中门槛；
3. `513700b` Official 配置与 Generation Pin；
4. `d655704` Generation / Legacy 回滚 Smoke；
5. `25ff5c8` 增量添加、更新、删除验收；
6. `6008a2e` Persistent Job Repository；
7. `2a69cb1` Worker、Heartbeat 和重启恢复；
8. `ac65b3c` 长任务提交与取消；
9. `6bfee96` Bootstrap Provider；
10. `aa47f87` QueryFrame 与 Dimension Coverage；
11. `1f832fe` 分级语义验证；
12. `3600817` 真实成本采集；
13. `43aecdc` Web/CLI 状态、生产组合和持久化 Web Job；
14. `95d9707` 保持无 root Web Job 的阶段一兼容基线；
15. `11ef8a0` 全仓 Mypy 类型收口；
16. 本文档提交。

后两项是全量质量门发现的独立兼容/类型修复，不改变用户规定的 14 个阶段四逻辑提交组，也可单独回滚。

## 回滚操作

`EngineCutoverService.switch_to_legacy` 显式返回 Legacy；
`rollback_previous` 恢复审计中的上一份 Engine/Generation 快照，并校验目标 Generation 仍为 `active` 或 `retired`。`tests/test_stage4_generation_rollback.py` 验证 retired Generation 回滚和 Legacy Top-K 恢复。

当前未发生 LightRAG Official 切换，因此生产回滚目标仍是 Legacy 本身。若未来验收通过，切换前必须保留当前审计记录和 Generation 目录；不得删除 Legacy Dense/GraphRAG 代码或索引。

## 增量与删除

`KnowledgeIndexService.build_incremental_generation` 从 active Generation 分叉，只处理 `changed_document_ids`，记录 `unrelated_history_reprocessed_count=0` 和 `full_rebuild=false`。`DocumentVersionRepository` 用
`content_hash + extraction_profile` 去重抽取。

`tests/test_stage4_delete_residual.py` 验证新 Generation 不含被删 Document、旧 Generation 仍保留回滚数据、Canonical PDF/原文不物理删除、其他 Workspace Generation 不受影响。真实 active Generation 尚未执行破坏性删除验收。
