# 多用户并发运行时契约

当前实现把并发控制拆成四个边界：

```text
Tenant/Principal
  └─ Project ACL
       ├─ Session lock：同一 Session 同一主体最多一个活动 Run
       ├─ Run/Job scope：状态、幂等键、队列负载按主体隔离
       └─ Project writer fence：Manuscript/Patch Apply 串行 + revision CAS
```

## 请求与身份

- 真实 CrewAI Scholar 请求统一 `202 Accepted` 入队，API 进程不执行 Agent。
- `tenant_id`、`principal_id` 来自受信任的认证解析器；默认 Header 仅用于本地开发，不能作为生产认证。
- Run、Job、Session 都持久化主体字段；同一个客户端幂等键会经过 tenant/principal 命名空间哈希。
- Project API、Approval、Build、Evidence 和 Manuscript API 先检查 Project ACL。外部 IAM 完成授权后调用 `ScholarProjectStore.grant_member()` 配置成员。

## 队列与故障恢复

- Job claim 生成随机 `lease_token`；Heartbeat、终态写入必须带 `worker_id + lease_token`，旧 Worker 不能覆盖新 Worker。
- Job 有 `PLANNED → STARTED → ...` action journal；耗尽重试预算的 `FAILED` Job 可进入 DLQ 查询和人工 `requeue_failed()` 重放。
- `RETRY_PENDING` 支持 `available_at`、指数退避和 jitter；claim 会按主体当前运行数优先选择较空闲的 scope，并可用 `max_running_per_scope` 限流。
- Worker registry heartbeat 与 Job heartbeat 分离，Worker 失联后由 recovery 把 Job 收束为 `INTERRUPTED`，再按 retry policy 重试。

## 文稿写入

- Writer 只产生 immutable `DraftPatch`，人工 Accept/Reject 是唯一副作用入口。
- Accept 全程持有 Project 跨进程写锁；写入前检查 `base_hash`，保存 Manuscript projection 时再做 `expected_version` CAS。
- 因此两个用户可以并发研究或生成候选 Patch，但不能静默覆盖同一份 `.tex`；冲突会进入 `CONFLICT` 并保留审计记录。

## 仍需外部基础设施验证

本地 SQLite 已具备单机多进程一致性和恢复语义。要达到多副本生产规模，还需要把 Jobs/Session/Project/Events 迁移到共享事务数据库，把队列 claim 迁移到共享 broker，并完成真实 OIDC/JWT、配额压测、故障注入和 24 小时 soak test；当前代码不会把本机 SQLite 宣称为跨节点高可用方案。
