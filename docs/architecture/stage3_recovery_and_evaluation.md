# 阶段三恢复与评估

## Claim—Evidence 验证

[`ClaimEvidenceValidator.validate`](../../app/research/validation.py) 只建立 Selected Evidence 注册表，依次检查：Claim 非空、存在 evidence citation、引用未越过 Selected 边界、graph inference 语气、analogy 语气、Claim/证据词面支持、冲突说明和 Scope。无 Evidence 或无 Claim 的确定性答案会 fail closed。

“证据真正支持 Claim”当前采用保守 Token 重叠代理，不等同于完整自然语言蕴含模型。高风险的 graph inference 与 analogy 检查优先于通用重叠检查，使恢复动作能够先降低事实强度。[`deterministic_repair`](../../app/research/validation.py) 只删除不支持 Claim；全部删除后返回“证据不足”，不会新增事实或再次调用 Generator。

## 恢复阶梯

| Level | 实现状态 | 动作 |
|---:|---|---|
| 0 | 已实现 | 参数/格式问题与无新增事实 Claim 修复 |
| 1 | 已实现 | Connection/Timeout 工具错误按 ToolSpec 重试，Trace 进入 `RECOVERING -> EXECUTING` |
| 2 | 已实现 | 可选补检索/外部搜索连续失败时保留已有 Selected Evidence；主检索无证据时安全拒绝 |
| 3 | 已实现 | DeepResearch 外部搜索失败后缩减为本地 Verified Evidence 范围 |
| 4 | 已实现 | Relation coverage 不足时记录并执行一次限定补检索 |
| 5 | 已实现 | 预算耗尽或不可恢复错误后安全终止，不生成无证据结论 |

## Trace 与验收

Trace 包含 workflow、state history、Step/Tool、usage、selected evidence ID、coverage、conflicts、validation issue、recovery action 和 termination reason。阶段三自动化位于 [`tests/test_stage3_research_harness.py`](../../tests/test_stage3_research_harness.py)，覆盖状态、八类预算、Context 隔离、Tool 注册与 Scope 拒绝、四类 Workflow、故障降级、无证据拒答、图推断、长期记忆、生产 handler、外部兼容门面和具体后端依赖扫描。

恢复测试还覆盖 Relation gap backend 失败后使用首轮 Verified Evidence，以及 DeepResearch 外部搜索失败后使用本地 Verified Evidence。当前确定性测试中跨 Workspace handler 调用数为 0，等价于测试夹具泄漏率 0。真实多 Workspace 语料上的长期统计仍需加入离线 Evaluation 数据集；不能用单元测试替代线上泄漏监控。
