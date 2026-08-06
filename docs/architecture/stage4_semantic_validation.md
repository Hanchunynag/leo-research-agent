# Stage 4 Semantic Validation

`app.research.validation.TieredClaimEvidenceValidator` 实现三级验证：

1. `ClaimEvidenceValidator` 检查非空 Claim、Selected Evidence、引用存在、词面关联、Scope、冲突、graph inference 和 analogy 语气。
2. `LocalSemanticValidator` 只接收单个 Claim 与最多四条被引用 Evidence，输出 `supports`、`partially_supports`、`contradicts` 或 `unrelated`。
3. `HighRiskSemanticJudge` 仅用于因果、创新/空白、graph inference、analogy、多文献综合、冲突或本地低置信度 Claim；每次答案最多调用一次。

Judge 结果必须提供结构化 `label`、`confidence` 和 `reason`。`rewritten_answer` 等字段不会进入结果。恢复顺序为：删除矛盾/无关 Claim；把部分支持降级为“提示”；Judge 不可用时标记“待验证假设”；无剩余 Claim 时拒绝确定性结论。

`ResearchRuntime` 默认使用分级验证，并在 Trace 中记录
`semantic_judge_calls`、语义输入/输出 Token 和逐 Claim 结果。测试位于
`tests/test_stage4_semantic_validation.py`。

待确认：默认本地实现 `HeuristicLocalSemanticValidator` 是离线保守回退，不是已校准的领域 NLI/Cross-Encoder；生产 High-Risk Judge 尚未配置。因此高风险低置信度 Claim 会降级或拒绝，而不会静默通过。
