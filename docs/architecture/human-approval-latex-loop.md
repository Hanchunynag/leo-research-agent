# Phase 2B Human Approval & LaTeX Loop

## Ownership and lifecycle

`DraftPatch` 是 immutable proposal，Owner 是 Scholar Project，不是生成它的 Conversation
Session。Project `.scholar/project.db` 保存 Patch 内容、Review Report、来源 `session_id/run_id`、
生命周期和审批审计；Session/Run 只保留引用关系。

```text
DraftPatch
  -> AWAITING_APPROVAL
  -> REJECTED
  -> APPROVED -> APPLYING -> APPLIED
                         \-> CONFLICT / FAILED
APPLIED -> external Build -> BUILD_TRIGGERED / SUCCESS / FAILED
```

Reject 是终态。已应用 Patch 再次 Accept 只返回历史结果，不重复写文件、版本或审计副作用。
Build 失败不回滚已应用的文件。

## Human authority and Apply guard

只有 human-facing HTTP/CLI/VS Code client 可以调用 Accept/Reject。请求必须携带
`patch_id`、所属 `project_id`、Patch 的 `expected_base_hash` 和 `actor`。Backend 从 Project
Store 读取 immutable proposal，不接受客户端 supplied `proposed_content`，也不存在
`force=true`/`skip_approval=true` 后门。

Accept 前再次读取目标 `.tex`，检查当前 hash 等于 Patch 的 `base_hash`，并检查
`original_content`。Review Report 含 `BLOCKER` 或 `HIGH` 时 Apply 被拒绝。写入继续复用
`ManuscriptSynchronizer` 的 project-root containment、临时文件、flush/fsync 和 atomic replace。
落盘后从真实文件重新扫描并保存新的 hash、section version、project state 和 old/new hash audit。
并发编辑返回 `PATCH_CONFLICT`，不会覆盖用户文件，也不进行三方自动合并。

## HTTP and CLI

现有 FastAPI Server 增加：

```text
GET  /api/scholar/patches/{patch_id}
POST /api/scholar/patches/{patch_id}/accept
POST /api/scholar/patches/{patch_id}/reject
POST /api/scholar/projects/{project_id}/build
GET  /api/scholar/projects/{project_id}/diagnostics
POST /api/scholar/projects/{project_id}/build/report
```

CLI fallback 使用同一服务：

```text
scholar patch show <patch_id>
scholar patch accept <patch_id> --project-id ... --expected-base-hash ... --actor ...
scholar patch reject <patch_id> --project-id ... --expected-base-hash ... --actor ...
```

## VS Code and LaTeX Workshop boundary

`vscode-extension/` 只负责 Patch Preview、审批请求、Build command 和 Diagnostics 展示；它不
持有 Research/Facts/Contribution/Review 状态，也不接受任意目标文件、正文或 shell command。
Preview 使用 VS Code 原生 `vscode.diff`，不实现自定义 HTML merge。

Build 通过 `vscode.commands.executeCommand("latex-workshop.build")` 调用已安装的 LaTeX
Workshop。找不到该 command 时返回 `LATEX_WORKSHOP_UNAVAILABLE`；Backend 的请求状态是
`BUILD_TRIGGERED`，只有 Extension/Bridge 报告真实结果后才记录 `SUCCESS` 或 `FAILED`。当前
没有 `pdflatex`、`latexmk` 或自动错误修复 fallback。

Diagnostics 是 `LatexDiagnostic(file, line, column, severity, message, source)`，只做
`ERROR/WARNING/INFO` normalization 和展示，不由 LLM 解析日志或自动修改 Manuscript。
