# LEO Research Agent

LEO Research Agent 是一个以本地论文库为证据基础、以 CrewAI 多 Agent 为生产
执行核心的学术研究工作台。系统支持 PDF 解析、分层混合检索、证据治理、论文
写作、审阅、人工审批和可恢复运行。

唯一生产架构说明：

[docs/architecture/production_architecture.md](docs/architecture/production_architecture.md)

## 生产链路

```text
Web / CLI
  -> ScholarRunManager
  -> Persistent Job Queue
  -> ScholarRunWorker
  -> Harness
  -> CrewAI Flow
  -> Manager -> Research / Writer / Reviewer
  -> Evidence / DraftPatch / ReviewReport
  -> Human Approval -> PatchApprovalService
```

Web 和 CLI 只创建、查询、恢复和取消持久化 Run；只有 Worker 执行 Agent。项目
只有这一条 CrewAI 生产链路，不保留其他编排框架或问答入口。

## 安装

项目使用 Python 3.11–3.13 和 `uv`：

```bash
uv sync --frozen --all-groups
cp .env.example .env
```

至少配置：

```dotenv
LEO_LLM_BASE_URL=http://127.0.0.1:11434
LEO_LLM_MODEL=qwen3:8b
LEO_SCHOLAR_RUNTIME_MODE=production
```

真实部署还需要可访问的本地模型服务、论文解析环境，以及按需配置 Qdrant 和
MySQL。敏感配置只放在 `.env` 或部署密钥中。

## CLI

解析与建立知识库：

```bash
uv run python main.py parse path/to/paper.pdf
uv run python main.py library rebuild
uv run python main.py knowledge build
uv run python main.py knowledge status
uv run python main.py hierarchical build
```

提交和运维 Scholar Run：

```bash
# 只入队，不在 CLI 进程执行 Agent
uv run python main.py scholar request \
  "比较三篇论文的 Doppler 定位方法并写出引言" \
  --project-id PROJECT_ID --task-type WRITE_INTRODUCTION

uv run python main.py scholar status RUN_ID
uv run python main.py scholar resume --run-id RUN_ID

# Worker 是唯一 Agent 执行进程
uv run python scripts/run_scholar_worker.py --once --max-jobs 10
uv run python scripts/run_scholar_worker.py
```

人工审批和构建桥接：

```bash
uv run python main.py scholar patch show PATCH_ID
uv run python main.py scholar patch accept PATCH_ID \
  --project-id PROJECT_ID --expected-base-hash HASH --actor human
```

## Web

启动 API：

```bash
uv run python main.py web --host 127.0.0.1 --port 8000
```

前端开发：

```bash
cd web
npm install
npm run dev
```

核心 API：

| API | 作用 |
| --- | --- |
| `POST /api/scholar/runs` | 创建并入队 Scholar Run |
| `GET /api/scholar/runs` | 查询当前主体可见的 Runs |
| `GET /api/scholar/runs/{run_id}` | 查询 Run 投影 |
| `GET /api/scholar/runs/{run_id}/events` | SSE 事件回放 |
| `POST /api/scholar/runs/{run_id}/resume` | 从持久化 checkpoint 恢复 |
| `POST /api/scholar/runs/{run_id}/stop` | 协作式停止 |
| `POST /api/papers/upload` | 入队 PDF 解析 Job |
| `GET /api/papers` | 查询论文目录 |
| `GET /api/system/status` | 查询解析/索引状态 |
| `GET /ready` | 查询服务和 Worker 就绪状态 |

论文解析 Job 仍属于 Web 的 Corpus/Index 能力；它不执行 Scholar Agent。

## 目录职责

```text
app/web/                         FastAPI 与 Web 读写控制面
app/scholar/runs.py              Run Manager、Job handler、Worker
app/harness.py                   唯一 CrewAI Harness 与 Runtime 组装根
app/orchestration/crewai/        CrewAI Flow、Agent、Tool、Trace
app/scholar/research/            Research Capability、provider gateway、budget
app/scholar/writing/             Writing、Reviewer、DraftPatch
app/knowledge/                   Corpus、UnifiedKnowledgeService、索引生命周期
app/session/                     Session、Goal、Run、Checkpoint 持久化
app/jobs/                        通用持久化队列、lease、重试和取消
web/src/                         Scholar Console
```

EvidencePack、DraftPatch、ReviewReport 由 domain service/store 持有；CrewAI
Flow 只保存有限引用和状态。`CapabilityBudget` 只限制 Research 工具调用，不
是独立的编排入口。`Harness` 只负责依赖组装、资源生命周期和 CrewAI Runtime
交付，跨 Agent 编排仍由 CrewAI Flow 负责。

## 验证

```bash
.venv/bin/python -m compileall -q app main.py scripts
.venv/bin/pytest -q
cd web && npm run build
```

真实 LLM、外部学术 Provider、Embedding/Reranker 和 Docker 依赖需要在对应环境
中单独验证；离线测试使用 deterministic provider，不将 fixture 结果标记为真实
Provider E2E。
