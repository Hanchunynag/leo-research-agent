# 阶段三 Context 策略

[`PhaseContextPack`](../../app/research/context.py) 用 `audience` 区分 Router 与 Generator，二者不能复用同一载荷。

## Router

[`ResearchContextManager.router_pack`](../../app/research/context.py) 只包含当前问题、Workspace 摘要和最近最多四条会话消息。单条历史内容截断为 800 字符；不含 Evidence、工具日志、完整 Session 或 Prompt 历史。Router Context Token 同时计入 `context_tokens` 和 `total_tokens`。

## Generator

[`ResearchContextManager.generator_pack`](../../app/research/context.py) 只包含：用户问题、Scope constraints、Selected Evidence、冲突、输出格式。允许的证据字段由 `_EVIDENCE_FIELDS` 固定为 source/evidence/document/chunk/page/block/content/grade/directness 等引用所需字段。任何非 `selected` 状态立即抛出 `ValueError`。

[`AgenticReasoningGeneratorAdapter.generate`](../../app/research/adapters.py) 将该最小 Context 转为结构化生成消息，不传 Session、候选列表或 Tool Trace。生成结果中的 provider `total_tokens` 计入 Run 总预算。

## 长期记忆

[`ResearchStateStore.commit_workspace`](../../app/research/memory.py) 只允许 `confirmed_scope`、`document_roles`、`direction_tree`、`verified_conclusions`、`pending_hypotheses`。隐藏推理、临时猜测、完整工具日志和未验证事实被明确拒绝。Session、Run、Job 使用各自白名单投影，不与 Workspace 长期事实混写。
