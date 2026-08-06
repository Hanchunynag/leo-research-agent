# Stage 4 Known Limitations

## 阻止正式切换的问题

1. LightRAG DirectQA nDCG@10 仅为 Legacy 的 78.95%，低于 95% 门槛。
2. 8 条关系题全部等待人工确认 qrel，Relation Precision/Coverage 不能作为真值。
3. 真实 active Generation 尚未执行新增、删除和回滚破坏性验收；现有结果来自隔离测试。
4. 因上述问题，Official 仍为 Legacy；LightRAG 仅为 Shadow。

## 生产可靠性待办

- 当前索引仓库有 1 个 active、7 个 failed Generation。失败原因需要单独归档清理，但不得删除回滚所需数据。
- Web 兼容 closure 在重启后会明确失败并要求幂等重提；Bootstrap Provider Job 可以从 payload/checkpoint 恢复。
- Cooperative cancellation 已实现，但通用 OS 级 Worker 进程终止待确认。
- 在线 Crossref/OpenAlex/arXiv → PDF → MinerU → LightRAG 全链路尚未在受控网络完成一次验收。
- 默认本地语义层仍是启发式回退；领域 NLI/Cross-Encoder 与 High-Risk Judge 尚未配置和校准。
- 五类真实回答成本、首 Token 延迟和货币成本未完成，不能引用测试夹具数值。
- 通用 OS 级 Worker 进程终止、领域语义模型校准和在线 Provider 成本仍需后续生产验证；当前全仓 Mypy、Ruff 和 Pytest 已通过。

## 完成条件结论

阶段四尚未完成。已完成配置门禁、回滚机制、持久化 Job、异步提交、维度 Coverage、分级验证和成本采集基础；正式切换、人工关系真值、真实增删验收、在线 Bootstrap 和五类成本基准仍未满足。不得为宣布完成而切换 LightRAG。

当前质量门：全仓 Pytest `236 passed, 1 skipped`；Ruff 通过；Mypy `152 source files` 通过；`git diff --check` 通过；`app/research` 具体后端依赖扫描无命中。
