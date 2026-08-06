# Stage 4 Job Reliability

## 持久化模型

`app.jobs.repository.PersistentJobRepository` 使用独立数据库
`data/jobs/jobs.sqlite3`，不修改现有业务数据库 Schema。`JobRecord` 保存 `job_id`、类型、Workspace/Scope、Document/Generation 引用、最小 payload、状态、尝试次数、Heartbeat、检查点、错误摘要和结果引用。

状态路径为：

```text
QUEUED -> RUNNING -> SUCCEEDED
RUNNING -> RETRY_PENDING -> RUNNING
RUNNING -> FAILED
RUNNING + restart -> INTERRUPTED -> RETRY_PENDING | FAILED
QUEUED/RUNNING -> CANCELLED | CANCEL_REQUESTED -> CANCELLED
```

相同幂等键只创建一个 Job。Job payload 拒绝 `api_key`、Prompt、PDF bytes、完整模型输出和全文字段，并限制在 64 KiB。完整结果由
`app.research.providers.JobResultStore` 或 `app.web.jobs.JobManager` 保存为文件，任务表仅持久化 `result_reference`。

## Worker 与恢复

`app.jobs.worker.PersistentJobWorker` 按 `job_type` 注入 Handler，使用
`JobExecutionContext.checkpoint` 写 Heartbeat 和安全检查点。启动时
`recover_after_restart` 先标记失联任务为 `INTERRUPTED`，再依据
`attempt < max_attempts` 重试或失败。Worker 只 claim 已注册类型，避免 Bootstrap Worker 误取 Web Job。

`LongTaskSubmitter` 使 `literature.download`、`document.parse`、知识索引、Generation 构建和批量更新/删除立即返回 `job_id`。`ToolGatewayRegistry.ainvoke` 为短工具提供 `asyncio.wait_for` 截止时间。

## 取消与 Web 兼容

运行任务通过持久化 `CANCEL_REQUESTED` 在检查点协作取消；队列任务可直接取消。Web Job 的状态、事件和结果可在重启后查询。旧 Web closure 无法跨进程重建时会明确失败为
`ProcessRestartRequiresResubmission`，不会长期伪装为 RUNNING。

待确认：当前没有通用的 OS 级 Worker 子进程强杀；MinerU 等无检查点的第三方调用仍依赖外部进程退出或超时。相关测试为
`tests/test_stage4_job_recovery.py` 和
`tests/test_stage4_job_cancellation.py`。
