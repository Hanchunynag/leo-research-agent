# 阶段三 Tool Gateway 注册表

注册表由 [`default_tool_specs`](../../app/research/tools.py) 定义，调用统一经过 [`ToolGatewayRegistry.invoke`](../../app/research/tools.py)。Gateway 在调用 handler、消耗预算之前检查 Workflow、权限、`workspace_id` 与 `scope_version`；因此错误 Scope 不会触达 Unified Service。

| Tool | 权限 | 超时 | 幂等 | 副作用 | Workflow | 重试 |
|---|---|---:|---|---|---|---|
| `knowledge.retrieve` | `knowledge.read` | 30s | 是 | read | 全部 | Connection/Timeout，最多 2 次 |
| `workspace.read_scope` | `workspace.read` | 5s | 是 | read | 全部 | 1 次 |
| `workspace.update_scope` | `workspace.write` | 10s | 是 | bounded_write | Bootstrap | 1 次，要求 expected scope_version |
| `literature.search` | `literature.search` | 30s | 是 | read | Deep/Bootstrap | Connection/Timeout，最多 2 次 |
| `literature.get_metadata` | `literature.read` | 15s | 是 | read | Deep/Bootstrap | Connection/Timeout，最多 2 次 |
| `literature.download` | `literature.download` | 60s | 是 | external_write | Bootstrap | Connection/Timeout，最多 2 次 |
| `document.parse` | `document.parse` | 300s | 是 | bounded_write | Bootstrap | 1 次 |
| `job.get_status` | `job.read` | 5s | 是 | read | 全部 | 1 次 |

每项同时声明 JSON 风格 input/output Schema。当前轻量 `_validate` 检查 required 字段和基本类型；它不是完整 JSON Schema validator。超时在 handler 返回后通过 elapsed time 判定，不具备抢占式取消能力。模型没有 Shell 或数据库 Tool，且 Gateway 没有注册此类能力。

[`unified_tool_handlers`](../../app/research/runtime.py) 是生产组合边界：`knowledge.retrieve` 显式透传 Workspace/Scope，只接受 `evidence_state=selected` 的结果，并把 Evidence Intelligence 的冲突诊断标准化；Candidate 或 Verified Evidence 不会进入 Workflow。具体知识服务对象只存在于 handler 闭包中。
