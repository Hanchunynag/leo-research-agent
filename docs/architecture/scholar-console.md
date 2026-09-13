# Scholar Web Console

Scholar Web Console 是 ScholarHarness 的 Agent control and observability
surface，不是 LaTeX IDE。它读取现有 Session Runtime、Project Store、Harness
diagnostics、Evidence/Citation projection 和 Evaluation result；页面不创建第二
份 Workflow State，也不直接访问 Qdrant、Research Engine 或 Project 文件。

## Boundary

```text
Browser
  → FastAPI read/request endpoints
  → ScholarHarnessService / ScholarConsoleProjection
  → Session + Project + Harness stores
```

Web Console 负责 Project/Session/Run 选择、Scholar Request、Run Flow、Evidence
与 Citation、Manuscript State、DraftPatch 状态、Checkpoint/Resume 和 Evaluation
展示。真实正文编辑、Diff、Accept/Reject、LaTeX Build 和 PDF Preview 继续由
VS Code + LaTeX Workshop 负责。Web 的 Accept/Reject 按钮也只能调用已有的人类
审批 API；Agent 没有审批或文件写入权限。

## Run projection and events

`ScholarConsoleProjection.run_snapshot()` 从已持久化的 Run/Result metadata 形成
只读 snapshot。Harness diagnostics 是已脱敏且有界的 trace projection，写入
Session Runtime metadata，因此进程重启后仍可查看历史运行，但不会把 Domain
State 复制进 Deep Agents checkpoint。

`run_events()` 将 snapshot 中的 Harness trace 映射成浏览器需要的事件：

```text
RUN_STARTED → SKILL_SELECTED → RESEARCH/EVIDENCE → DRAFT/REVIEW
→ PATCH_CREATED → WAITING_USER → RUN_COMPLETED/FAILED/INTERRUPTED
```

事件的 `event_id` 使用 `run_id` 前缀和稳定顺序，`cursor` 是从 1 开始的单调
Run-local 序号。SSE 的 `after` 表示客户端最后已消费的 cursor，服务端只发送
更大的 cursor，并以 `event: end` 结束一次 replay。前端收到重复事件时按
`event_id` 去重，连接异常则从最后 cursor 退避重连；旧连接的迟到事件不会污染
新 Run。当前 Scholar Request API 是同步 Domain 调用，因此该 SSE 端点是持久化
trace 的 replay stream，不是绕过 Harness 的第二个 publisher。

## Read models

- Run snapshot：request、selected skill、run/session/thread、result type、usage、
  termination reason。
- Workflow：Supervisor、Skill、Research/Validation、Writing、Reviewer、Patch
  和 Human Approval 的状态、摘要、耗时和有限 metadata。
- Evidence/Citation：只显示 Verified Evidence 或已持久化的 external audit
  projection；校验状态和 content hash 必须匹配，未验证 Discovery Candidate 不会
  被渲染成证据。视图同时提供 claim、source、identity、日期、locator、evidence span、
  validation、CitationBinding/BibKey 和 Patch reference。
- Manuscript State：直接读取 `ManuscriptSynchronizer` 和 Project Store，展示
  section path/hash/version、CURRENT/STALE、dependency reason、review status 和
  最后一个 Patch。
- Evaluation：读取 `ScholarHarnessEvaluationSuite` 的确定性指标，不由前端
  根据文字猜测 PASS/FAIL，也不把文本质量伪装成精确分数。

## API

```text
POST /api/scholar/requests
POST /api/scholar/requests/resume
GET  /api/scholar/runs/{run_id}
GET  /api/scholar/runs/{run_id}/events?after=N
GET  /api/scholar/runs/{run_id}/evaluation
GET  /api/scholar/projects/{project_id}/state
GET  /api/scholar/projects/{project_id}/evidence
GET  /api/scholar/runtime/status
GET  /api/scholar/demo
```

`/api/scholar/demo` 是固定的、不连接 Provider 的截图/演示数据，并在页面显示
`DEMO / FIXTURE DATA`。它只能验证展示 Contract，不能作为 Production E2E 或
Release Gate。
