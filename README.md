# LEO Research Agent

面向低轨（LEO）机会信号定位研究的本地论文解析与 RAG 知识库项目。

项目当前已经完成从 PDF 到本地混合证据检索的可评测 RAG 基线链路：

```text
PDF 入库
  → PDF 预检查
  → MinerU 解析
  → MinerU 数据标准化
  → 每篇论文生成一个 paper.json
  → 全库生成可重建的 papers.jsonl
  → 恢复内容区域和章节层级
  → 生成确定性结构化 Chunk
  → 建立 BM25 与 BGE-M3/Qdrant local 索引
  → 使用 RRF 融合
  → 使用 BGE Cross-Encoder 精排
  → 组装带稳定 SOURCE、页码和 block 的上下文证据包
  → 本地模型生成结构化 claims
  → 逐条校验引用并失败关闭
```

现在支持容错批量入库、论文身份归并、真实标题规范命名、增量知识层构建、
BM25、BGE-M3 单向量召回、Qdrant local、RRF、候选池 Oracle、Cross-Encoder
精排、证据上下文组装、OpenAI-compatible 本地回答模型、claim 级引用校验和
fail-closed 拒答。回答入口同时保留单轮 fast RAG，并新增持久化、多轮、可增量
检索和 Claim-Citation 语义验证的 agentic RAG。

## 核心设计原则

1. **MinerU 是唯一论文解析引擎**，不并行维护多套 PDF 解析器。
2. **只有一个解析入口**，CLI 和 UI 都调用同一个 `parse_paper()`。
3. **两个虚拟环境严格隔离**，主代码不导入 MinerU 环境中的 Python 包。
4. **原始数据不丢失**，PDF、MinerU 原始输出和 `raw_block` 都保留。
5. **文件身份与论文身份分离**：SHA-256 生成稳定 `document_id`，DOI 等规范
   标识生成 `work_id`；兼容目录继续保留 `paper_id`。
6. **RAG 必须依靠元数据区分论文**，不能依赖文件名或让 LLM 自己猜来源。
7. **支持增量维护**，新增一篇论文不应该重新解析和索引全部论文。

## 从零安装

以下步骤用于一台没有现成虚拟环境的新电脑。项目使用 Python 3.11，并通过
`uv.lock` 固定主环境依赖。`.venv`、`.venv-mineru` 和 `data/` 都是本地目录，
不会提交到 Git。

### 1. 安装基础工具

需要预先安装：

- Git；
- [uv](https://docs.astral.sh/uv/)；
- 足够存放 MinerU 依赖和模型的磁盘空间。

macOS 可以使用 Homebrew 安装 uv：

```bash
brew install uv
```

确认安装：

```bash
git --version
uv --version
```

### 2. 克隆项目并创建主环境

```bash
git clone https://github.com/Hanchunynag/leo-research-agent.git
cd leo-research-agent
uv sync --frozen --all-groups
```

`uv sync` 会读取 `.python-version`、`pyproject.toml` 和 `uv.lock`，自动创建
`.venv`，并安装运行依赖及 pytest、Ruff、MyPy 等开发工具。

验证主环境：

```bash
./.venv/bin/python --version
./.venv/bin/python main.py --help
```

### 3. 创建 MinerU 专用环境

当前经过验证的组合是 Python 3.11 和 MinerU 3.4.4：

```bash
uv venv --python 3.11 .venv-mineru
uv pip install \
  --python .venv-mineru/bin/python \
  "mineru[core]==3.4.4"
```

下载默认 `pipeline` backend 所需模型。国内网络优先使用 ModelScope：

```bash
./.venv-mineru/bin/mineru-models-download \
  --source modelscope \
  --model_type pipeline
```

模型下载体积较大，只需在首次安装或主动更换模型时执行。验证 MinerU：

```bash
./.venv-mineru/bin/mineru --version
```

预期输出：

```text
mineru, version 3.4.4
```

Windows 中可执行文件位于 `.venv\Scripts\`，需要把上述 `bin` 路径替换为
`Scripts`，例如 `.venv-mineru\Scripts\mineru.exe`。

### 4. 运行质量检查

```bash
./.venv/bin/pytest -q
./.venv/bin/ruff check app main.py tests
./.venv/bin/mypy app main.py
```

三项检查全部通过后，再解析论文：

```bash
./.venv/bin/python main.py parse "/absolute/path/to/paper.pdf"
```

首次解析会运行 MinerU，耗时取决于论文页数和本机性能。原始 PDF、MinerU
产物和 `paper.json` 会写入本地 `data/`，该目录已被 Git 忽略。

## 当前端到端流程

### 1. PDF 入库

入口：`app/ingestion/ingest.py`

处理内容：

- 验证文件存在且扩展名为 PDF；
- 检查 PDF 文件头；
- 计算完整 SHA-256；
- 使用 SHA-256 前 12 位生成稳定 Paper ID，例如 `P_2e250a42c5f9`；
- 清洗文件名并复制到 `data/raw/<paper_id>/`；
- 相同 PDF 再次导入时复用已有文件。

Paper ID 来自文件内容，而不是标题或文件名。因此即使两篇论文重名，也不会
发生目录冲突；同一篇 PDF 改名后再次导入，仍然得到相同 Paper ID。

### 2. PDF 预检查

入口：`app/parsing/precheck.py`

使用 PyMuPDF 统计：

- 页数、文件大小和文本字符数；
- 有无原生文字层；
- 是否为整页扫描图片；
- `native_text`、`scanned_image`、`scanned_with_ocr` 或 `mixed`；
- 可能的双栏页面；
- 嵌入图片数量。

预检查只生成元数据，不再作为独立人工步骤。MinerU 的默认 `auto` 方法会根据
PDF 类型选择文本或 OCR 路径。

### 3. 调用 MinerU

统一入口：`app/parsing/pipeline.py`

主流程由 `.venv` 中的 Python 运行，但 MinerU 始终通过专用环境的可执行文件
启动：

```text
.venv/bin/python
    └── subprocess → .venv-mineru/bin/mineru
```

默认 MinerU 参数：

- backend：`pipeline`；
- method：`auto`；
- formula：开启；
- table：开启。

若 `data/parsed/<paper_id>/mineru/` 已存在完整的
`content_list_v2.json + middle.json`，流程会直接复用。使用
`--force-mineru` 才会重新运行 MinerU。

MinerU 原始产物保留在 `data/parsed`，包括：

- Markdown；
- `content_list_v2.json`；
- `middle.json`；
- 模型和版面 JSON；
- 公式、图、表裁图；
- layout、span 和 origin PDF；
- pipeline 日志。

这些是底层解析资产，不是未来 RAG 的直接入口。

### 4. 标准化 MinerU 输出

内部适配器：`app/normalization/mineru_adapter.py`

pipeline 按 MinerU 阅读顺序逐页处理 `content_list_v2.json`，统一生成以下
block 类型：

- `title`
- `paragraph`
- `list`
- `equation`
- `figure`
- `table`
- `algorithm`
- `page_metadata`

每个 block 至少包含：

```json
{
  "block_id": "P_xxx_p003_b005",
  "paper_id": "P_xxx",
  "page_number": 3,
  "reading_order": 5,
  "type": "equation",
  "bbox": [0, 0, 100, 100],
  "text": "...",
  "latex": "...",
  "image_path": "...",
  "quality": {},
  "raw_block": {}
}
```

公式顺序和公式编号不是解析主流程的阻塞条件。公式最重要的数据是：

- 属于哪篇论文：`paper_id`；
- 在哪一页：`page_number`；
- 对应哪张原始裁图：`image_path`；
- MinerU 提取出的 LaTeX：`latex`。

### 5. 输出单一 paper.json

下游唯一入口：

```text
data/canonical/<paper_id>/paper.json
```

`paper.json` 包含：

- `schema_version`；
- `paper_id`；
- 标题等论文元数据；
- 原始 PDF 完整 SHA-256 和相对路径；
- PDF 预检查结果；
- MinerU backend、版本和实际/请求参数；
- 页面信息；
- 全部标准化 blocks；
- 公式、图和表的轻量索引；
- 每个 block 的完整 MinerU `raw_block`。

旧版本生成的 `blocks.json`、`formulas.json`、`formula_review.md` 和相关 report
只是历史产物。新流程不会生成或读取它们。

## 两个虚拟环境

项目刻意隔离两个环境：

- `.venv`：运行主脚本、Gradio、PyMuPDF、数据标准化和测试；
- `.venv-mineru`：只安装并运行 MinerU。

pipeline 不会回退到系统 PATH 中寻找 MinerU，防止误用主环境里的同名命令。
默认位置是：

```text
.venv-mineru/bin/mineru
```

也可以显式覆盖：

```bash
export LEO_MINERU_EXECUTABLE=/absolute/path/to/.venv-mineru/bin/mineru
```

## 使用方法

### 解析一篇论文

```bash
./.venv/bin/python main.py parse /path/to/paper.pdf
```

单篇解析成功后会自动重建 `data/knowledge/papers.jsonl`。

强制重新运行 MinerU：

```bash
./.venv/bin/python main.py parse /path/to/paper.pdf --force-mineru
```

查看全部选项：

```bash
./.venv/bin/python main.py parse --help
```

### 批量解析论文

解析目录第一层中的全部 PDF：

```bash
./.venv/bin/python main.py batch "/path/to/papers"
```

递归扫描子目录：

```bash
./.venv/bin/python main.py batch "/path/to/papers" --recursive
```

批处理逐篇调用同一个 `parse_paper()`。单篇失败会记录在报告中，但不会阻止
后续论文。完成后自动重建全局目录，并写入：

```text
data/knowledge/last_batch_report.json
```

只要存在失败论文或 canonical 目录问题，命令会在打印完整 JSON 报告后返回
非零退出码，方便脚本和 CI 识别部分失败。

### 管理论文目录

从全部 `data/canonical/*/paper.json` 原子重建目录：

```bash
./.venv/bin/python main.py library rebuild
```

列出目录记录：

```bash
./.venv/bin/python main.py library list
```

列出按 `work_id` 归并的逻辑论文和 PDF 版本：

```bash
./.venv/bin/python main.py library works
```

检查 raw、MinerU、canonical 和 `papers.jsonl` 是否一致：

```bash
./.venv/bin/python main.py library status
```

`papers.jsonl` 是可重建的派生目录，不能作为单篇论文事实来源手工维护。损坏的
`paper.json` 会作为 issue 报告，其余有效论文仍会进入目录。

`papers.jsonl` 一行对应一个具体 PDF 文档；`works.jsonl` 一行对应一篇逻辑
论文。同一 DOI 的出版社版、预印本或带批注 PDF 拥有不同 `document_id`，但可
归入相同 `work_id`，后续检索可按 `work_id` 去重、按 `document_id + page`
引用具体证据。

### 构建结构化知识层并检索

从全部已核验、具备 `work_id/document_id` 的 canonical 论文增量构建：

```bash
./.venv/bin/python main.py knowledge build
```

默认正文上限为 700 个近似词元。小于 80 词元的父章节引导段会作为独立
`parent_context` 附着到直属子章节；同一章节的连续 Chunk 最多附带 80 词元的
`overlap_context`。三个阈值可分别通过 `--max-tokens`、
`--min-chunk-tokens` 和 `--overlap-tokens` 调整。上下文保留自己的章节、页码和
block ID，不会跨论文、跨内容区域或跨兄弟章节。输入未变化时会根据指纹复用。

本地关键词证据检索：

```bash
./.venv/bin/python main.py search \
  "ephemeris and timing error disambiguation" \
  --limit 5
```

可以使用 `--work-id` 或 `--document-id` 过滤。默认每个 `work_id` 最多返回两个
Chunk，结果包含标题、章节路径、页码、`block_ids`、父章节上下文、重叠上下文
和具体内容。若索引和 `chunks.jsonl` 指纹不一致，检索会拒绝使用旧索引并提示
重新构建。

固定 BGE-M3 revision 并构建可复用的 Qdrant local 索引：

```bash
./.venv/bin/python main.py dense build \
  --revision 5617a9f61b028005a4858fdac845db406aefb181

./.venv/bin/python main.py dense search \
  "Which measurements track LEO ephemerides?" \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only
```

Dense 只使用归一化的 1024 维单向量，不启用 BGE-M3 sparse 或 multi-vector。
Manifest 记录 Chunk 指纹、文本策略、模型 revision、维度、距离和 collection；
任一项不一致时检索会拒绝旧索引。混合检索默认融合 BM25 Top-20 与 Dense
Top-20，使用 `k=60` 的 RRF：

```bash
./.venv/bin/python main.py hybrid search \
  "Which measurements track LEO ephemerides?" \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only
```

固定 BGE Reranker revision，对 RRF Top-20 做精排：

```bash
./.venv/bin/python main.py rerank search \
  "Which measurements track LEO ephemerides?" \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --local-files-only
```

Reranker 只比较 `Query + RRF Candidate` 并输出原始相关性 logits，不与 BM25、
Dense 或 RRF 分数线性相加。结果保留两路召回排名、RRF 排名和 Reranker 分数。

将检索结果组装成 LLM 可消费的证据包：

```bash
# 快速模式：BM25 + Dense + RRF
./.venv/bin/python main.py context build \
  "Which measurements track LEO ephemerides?" \
  --mode fast --token-budget 6000 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only

# 精确模式：RRF Top-20 + Cross-Encoder
./.venv/bin/python main.py context build \
  "Which measurements track LEO ephemerides?" \
  --mode accurate --token-budget 6000 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --local-files-only
```

`ContextBundle` 中每个 `[S1]`、`[S2]` 都绑定唯一 `chunk_id/work_id/document_id`、
章节、页码和 block IDs。同一逻辑论文只采用一个 PDF 版本；跨 document 的
parent/overlap 会被拒绝，重复上下文只保留一次，最后一个证据可在自身来源边界内
按预算截断。

CLI 是一次性进程，仍会在每次执行时加载模型。UI、API 或本地 Agent 应在进程
启动时创建一个 `RetrievalRuntime` 并复用；调用 `warmup()` 后，后续查询不会重复
创建 Dense 或 Reranker 模型实例。

### 生成带逐条引用的回答

先启动支持 OpenAI Chat Completions 接口的本地模型服务，例如 Ollama、LM Studio
或 vLLM，或者准备一个兼容该接口的远程服务。复制本地配置模板：

```bash
cp .env.example .env
```

`.env` 已被 Git 忽略。使用远程 DeepSeek 服务时，在 `.env` 中填写：

```dotenv
LEO_LLM_BASE_URL=https://api.deepseek.com
LEO_LLM_MODEL=服务商提供的模型名
LEO_LLM_API_KEY=新生成的密钥
LEO_LLM_TIMEOUT_SECONDS=120
LEO_LLM_MAX_TOKENS=1200
```

不要把密钥直接写进命令行、Python 文件、日志或 `.env.example`。如果密钥曾经
出现在终端历史或日志中，应先到服务商控制台撤销，再把新密钥写入 `.env`。

配置完成后，命令不再需要携带 API 参数：

```bash
./.venv/bin/python main.py answer \
  "Which measurements track LEO ephemerides?" \
  --mode fast \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only
```

配置优先级为命令行参数、系统环境变量、项目 `.env`。命令行的
`--llm-base-url`、`--llm-model` 等参数只用于临时覆盖；其中 base URL 可以是
服务根地址、`/v1` 地址或完整的 `/v1/chat/completions` 地址。本地 Ollama 可将
地址设为 `http://127.0.0.1:11434`，并将 API Key 留空。`accurate` 模式还需固定
Reranker revision，参数与 `context build` 相同。

回答模型不能直接输出自由文本引用。它必须返回结构化 claims：

```json
{
  "answerable": true,
  "claims": [
    {
      "claim_id": "C1",
      "text": "The observations jointly estimate both errors.",
      "source_ids": ["S1"]
    }
  ],
  "refusal_reason": null
}
```

应用只接受当前 `ContextBundle` 中存在的来源，并确定性渲染为
`claim text [S1]`。每条 citation 同时展开为 title、`work_id`、`document_id`、
章节、页码和 block IDs。空证据、上下文预算或来源完整性异常、无引用 claim、
重复或未知来源、非法模型 JSON 都不会输出部分回答，而是返回
`answerable=false`；CLI 对拒答使用退出码 2。

`answer` 默认输出回答、claims、去重后的 citations、校验结果、耗时、服务端返回
的模型标识和 token usage，不再重复打印完整论文上下文。只有排查检索证据时才
使用 `--include-context` 输出完整 `ContextBundle`。

#### Context Session 与缓存命中实验

普通动态问答保持 `S + Q + C`，每个问题重新检索。需要围绕同一批证据连续提问
时，指定一个安全的 session ID：

```bash
# 首次调用：检索、保存固定 ContextBundle，并使用 S + C + Q
./.venv/bin/python main.py answer \
  "哪些观测量用于估计星历和时钟误差？" \
  --context-session leo_timing \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only

# 后续调用：复用相同证据，不再加载 BGE-M3 或执行检索
./.venv/bin/python main.py answer \
  "其中多普勒观测对应哪些状态量？" \
  --context-session leo_timing
```

快照保存在本地私有目录 `data/runtime/context_sessions/`，包含稳定
`context_hash` 和完整性校验；不会提交 Git。只有研究主题或证据需求发生变化时
才显式刷新：

```bash
./.venv/bin/python main.py answer \
  "为当前问题重新选择证据" \
  --context-session leo_timing \
  --refresh-context-session \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only
```

动态问答默认 `--prompt-layout query_first`，Context Session 默认
`context_first`；两种模式都可以显式覆盖。session 复用状态、原始检索问题、
`context_hash`、完整 Prompt 指纹、稳定前缀指纹和近似 token 数记录在
`diagnostics`。若服务商返回 `prompt_cache_hit_tokens` 与
`prompt_cache_miss_tokens`，还会自动生成：

```json
{
  "cache_diagnostics": {
    "hit_tokens": 2560,
    "miss_tokens": 128,
    "eligible_prompt_tokens": 2688,
    "hit_rate": 0.952381
  }
}
```

命中率是服务端 token 级结果；`prompt_diagnostics` 中的 token 数只是项目统一的
近似计数，用于本地对照。固定 session 能提高缓存复用，但不会自动判断新问题是否
仍与旧证据相关；证据不再适用时必须刷新，不能用命中率替代回答质量评测。

fast 模式的校验保证“引用存在且身份可追溯”，不自动证明 claim 被引用文本语义
蕴含；需要该能力时使用下述 agentic 模式。语义正确率、引用精确率和拒答质量仍
需要在人工标注的生成式评测集中单独测量。

### Agentic Scientific RAG

`--retrieval-mode fast` 保留原有单轮行为和 JSON 必填字段；
`--retrieval-mode agentic` 使用 SQLite 恢复会话，并执行完整的有界流程：

```text
User Query
  → Topic Router + Standalone Query Rewrite
  → Category-aware Query Planner
  → BM25 + BGE-M3 → RRF Top-20
  → BGE Cross-Encoder + directness grade → Top-8
  → Evidence Registry 去重与复用
  → Evidence Coverage Check
      └─ partial/missing → focused query → 补充检索（默认总计最多 2 轮）
  → Context Builder → Top-5
  → Structured Answer Generator
  → Structural Citation Validation
  → Claim-Citation Semantic Entailment
      ├─ retrieve_more 且仍有轮次 → 补充检索并重新生成
      └─ rewrite/drop → 最多一次 Repair 并重新验证
  → append-only Session Events + State Delta
```

首次问答会自动创建 Session；显式 ID 便于独立进程继续同一会话：

```bash
./.venv/bin/python main.py answer \
  "哪些观测量用于估计星历和时钟误差？" \
  --retrieval-mode agentic \
  --session-id leo_error \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --local-files-only

./.venv/bin/python main.py answer \
  "那为什么多普勒能够估计钟漂？" \
  --retrieval-mode agentic \
  --session-id leo_error \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --local-files-only
```

不传 `--session-id` 时，输出的 `session.session_id` 可用于下一次调用。需要显式
隔离证据时使用 `--force-new-topic`。本地模型权重缺失且指定
`--local-files-only` 时，Reranker 会记录 `fallback_used=true` 并保持 RRF 顺序，
不会联网下载或令整个回答崩溃；也可用 `--disable-reranker` 做消融。

#### Agent Harness：状态机、预算和运行轨迹

Agentic 业务流程运行在独立的轻量 Harness 中，而不是让 LLM 自由决定循环和终止。
Harness 与 Router、Retriever、Generator 解耦，只负责五类系统约束：

```text
AgenticRunPolicy
  ├─ max_retrieval_rounds
  ├─ max_structure_repairs
  ├─ max_answer_repairs
  ├─ max_total_latency_ms
  ├─ fail_closed
  └─ allow_model_downloads

AgenticRunHarness
  ├─ finite-state transitions
  ├─ retrieval/repair budget
  ├─ deadline
  ├─ termination reason
  └─ ordered stage trace
```

有限状态机只允许以下主路径和显式回环：

```text
initialized
→ routing → planning
→ retrieving → reranking → coverage_checking
   └──────────────────────────────→ retrieving（证据不足且仍有预算）
→ context_building → compacting? → generating
→ structural_validating → semantic_validating
   ├──────────────────────────────→ retrieving（retrieve_more）
   └→ repairing → structural_validating（最多一次）
→ persisting → completed | refused
```

非法跳转会抛出 `HarnessConstraintError`；检索最多 1–5 轮，结构修复和答案修复
只能配置为 0 或 1。`fail_closed=false` 会在配置阶段直接被拒绝。可选总时限在新
检索、Repair 和生成前作为硬门禁；`--local-files-only` 同时令 Harness Policy 的
`allow_model_downloads=false`。

Agentic 输出的 `diagnostics.harness` 提供可复现实验和面试讲解所需的统一轨迹：

```json
{
  "policy": {
    "max_retrieval_rounds": 2,
    "max_structure_repairs": 1,
    "max_answer_repairs": 1,
    "fail_closed": true
  },
  "state": "completed",
  "termination_reason": "completed",
  "budget": {
    "retrieval_rounds_used": 2,
    "structure_repairs_used": 0,
    "answer_repairs_used": 0
  },
  "trace": [
    {"ordinal": 1, "stage": "routing", "status": "succeeded"},
    {"ordinal": 2, "stage": "planning", "status": "succeeded"},
    {"ordinal": 3, "stage": "retrieving", "attempt": 1, "status": "succeeded"}
  ]
}
```

Trace 只保存阶段、次数、耗时、数量和错误类型，不保存异常正文、API Key 或完整
Prompt。业务级 retrieval diagnostics 和 Harness 控制轨迹分别保留，避免把质量
评测与执行控制混为一层。

#### Topic、Session 与 Evidence Registry

一个 Session 可以包含多个 Topic。Router 同时使用上下文依赖、BGE-M3 语义
相似度、实体重合和轻量检索后的证据重合，默认权重为 `0.40/0.25/0.20/0.15`：

- `same_topic`：继续当前 append-only 消息流，筛选并复用已有证据；
- `related_subtopic`：创建以旧 Topic 为父节点的新分支，不默认继承旧证据；
- `new_topic`：创建完全独立的 Topic，不把旧对话或证据发送给回答模型。

组合分数不小于 `0.75` 倾向同主题，不大于 `0.45` 倾向新主题；中间区间最多
调用一次 temperature=0 的结构化 Router。常见结果如下：

| 当前主题 | 新问题 | relation | standalone_query |
|---|---|---|---|
| LEO 星历与时钟误差观测量 | 那为什么多普勒能够估计钟漂？ | same_topic | 为什么多普勒频率观测能够约束低轨卫星与接收机之间的相对时钟漂移？ |
| 同上 | RRF 中的 k=60 是什么意思？ | related_subtopic | 原问题已独立 |
| 同上 | Python 装饰器是什么？ | new_topic | 原问题已独立 |

Session Store 默认位于 `data/runtime/agentic_sessions.sqlite3`，包含
`sessions/topics/events/evidence_registry`。事件 ordinal 单调递增，类型包括
`user_query/query_analysis/evidence_added/answer/validation/state_delta/compaction`；
旧事件从不修改。每个 Topic 中同一 `chunk_id` 获得稳定的 `E001...`，答案内临时
`S1...` 只负责兼容当前 citation。重复命中的 Chunk 标为 `origin=reused`，不会
再次追加完整正文；新 Chunk 才产生 `evidence_added`。

```bash
./.venv/bin/python main.py session list
./.venv/bin/python main.py session show leo_error
./.venv/bin/python main.py session evidence leo_error
./.venv/bin/python main.py session compact leo_error
```

`session --session-db-path /path/to/sessions.sqlite3 ...` 和 answer 的同名参数可以
覆盖默认数据库。

#### Planner、Coverage、Entailment 与 Repair

Planner 把问题分类为 fact list、definition、mechanism、comparison、method、
numeric result、citation lookup 或 synthesis，同时区分 measurement/observable、
input、prior、state、parameter、method、result、metric、dataset 和 assumption。
例如用户问“观测量”时，预测星历和 SGP4 传播结果会被放入排除类别约束，只能作为
辅助输入或先验说明，不能生成 `category=measurement` 的直接 Claim。

Reranker 不只判断主题相关性，还为每个候选赋予 0–3 的 directness grade：3 表示
直接包含答案，2 表示关键支持，1 表示背景，0 表示不支持。因此只重复“星历误差、
时钟误差”的 Introduction 不会压过明确说明载波相位或多普勒用途的正文。

Coverage 必须逐个 subquestion 返回 `sufficient/partial/missing` 和稳定 Evidence ID。
缺失项产生 focused follow-up query，重新执行 Hybrid → RRF → Reranker，并按
`chunk_id` 合并。达到 `max_retrieval_rounds` 仍不足时系统 fail closed，输出具体
缺失项，而不是让 LLM 用常识补全。

Structural Validation 检查 JSON、Claim、S 编号和 Chunk 映射；Semantic Validation
再逐 Claim 检查 `entailed/partially_entailed/not_entailed/contradicted`、问题对齐、
类别、直接引用、条件扩张和冲突。例如 Claim“预测星历是一种观测量”即使引用了
“载波相位和预测星历作为算法输入”，也会得到 `category_correct=false`、非完全
蕴含和 `rewrite/drop`。Repair 最多一次，之后重新执行两层验证；仍无效时明确
返回 `validation.valid=false`。

#### Append-only Prompt、缓存与 Compaction

每个 Topic 的 Prompt 是固定 System Prompt/Schema 加按 ordinal 稳定序列化的事件
流。第二轮消息严格为第一轮完整消息的前缀再追加 H2；旧消息、工具顺序和证据正文
不重排、不重写。时间戳、request ID、耗时和 API usage 只进入数据库元数据或输出
diagnostics，不进入可缓存前缀。`prompt_cache` 报告稳定 `prefix_hash`、消息数和
Provider 的真实 usage；服务商不返回缓存 token 时对应字段是 `null`，不会伪造。

上下文估算达到模型窗口的 70% 时自动 Compaction，也可手动执行。Compaction
只追加新事件，不删除数据库历史；新消息流保留 Topic summary、目标、confirmed
facts、open questions、Evidence Registry 定位和摘要、未完成任务及最近事件。
由于它创建新前缀，下一轮缓存重新预热属于预期行为。

缓存命中率高只说明大量 Prompt token 的字节前缀被服务商复用，不能说明检索到了
正确证据，也不能证明 Claim 被 citation 蕴含。质量仍需分别看 Retrieval/Reranker
Recall、Evidence Coverage、Citation Precision/Recall、Claim Entailment Accuracy
和 answerable 判断；本地 `app.agentic.evaluation.aggregate_agentic_metrics()`
已经提供平均检索轮数、新增 Evidence、复用率、覆盖率、引用精确率 proxy、缓存率
和延迟的最小聚合接口。

#### Agentic 配置

主要 CLI 参数为 `--candidate-limit 20`、`--rerank-top-k 8`、
`--final-top-k 5`、`--max-retrieval-rounds 2`、
`--max-structure-repairs 1`、`--max-answer-repairs 1`、
`--max-total-latency-ms`、`--rrf-k 60`、
`--disable-reranker`、
`--disable-semantic-validation`、`--same-topic-threshold`、
`--new-topic-threshold`、四个 Router 权重、`--model-context-window` 和
`--context-compaction-threshold`、`--recent-events-after-compaction`。同名配置可
通过 `LEO_AGENTIC_*` 环境变量覆盖，
完整样例见 `.env.example`；CLI 优先于环境默认值。权重之和必须为 1，所有轮次和
Repair 均有硬上限。

API Key 优先读取 `LEO_LLM_API_KEY`，也兼容 `DEEPSEEK_API_KEY`。推荐只放在权限
受控的环境变量或被 Git 忽略的 `.env`；配置对象 repr、异常输出、Session DB、
Prompt、diagnostics 和测试快照不会保存 Key。若旧日志曾出现真实 Key，仅脱敏新
日志不能使旧 Key 失效，必须在服务商控制台撤销并轮换。

当前限制：CLI 是短进程，每次仍可能加载本地 Embedding/Reranker；交互部署应复用
长驻 `RetrievalRuntime`。确定性 Planner/Coverage/Semantic guardrail 偏保守，复杂
跨论文综合仍需要扩充人工评测集。Compaction 的 Evidence 摘要目前是确定性截断，
不是额外的生成式摘要。Agentic 指标接口已就绪，但仓库尚未包含大型多轮标注集。

### 运行检索评测基线

人工标注问题集保存在：

```text
data/evaluation/retrieval_questions.jsonl
```

每道题按最细粒度可用标注判断相关性：优先使用 `relevant_block_ids`，缺失时才
依次退回 `relevant_document_ids` 和 `relevant_work_ids`。父章节与重叠上下文中的
block 也保留独立来源，因此可以参与严格 block 级评测。

运行召回、候选池和精排基线：

```bash
./.venv/bin/python main.py evaluate retrieval
./.venv/bin/python main.py evaluate retrieval --retriever dense \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 --local-files-only
./.venv/bin/python main.py evaluate retrieval --retriever rrf \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 --local-files-only
./.venv/bin/python main.py evaluate retrieval --retriever oracle \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 --local-files-only
./.venv/bin/python main.py evaluate retrieval --retriever reranker \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --local-files-only
```

报告按检索器写入 `data/evaluation/*_baseline.json`，包含 Recall@1/5/10、MRR、
nDCG@10、按问题类型聚合的指标以及每道题的排名。当前 21 道题的首版基线为：

| 指标 | BM25 | Dense | RRF | Reranker |
|---|---:|---:|---:|---:|
| Recall@1 | 0.3571 | 0.2381 | 0.3810 | **0.5714** |
| Recall@5 | 0.5714 | 0.7381 | **0.8095** | 0.7857 |
| Recall@10 | 0.7857 | 0.8333 | 0.8810 | **0.9286** |
| MRR | 0.5137 | 0.4480 | 0.5409 | **0.7110** |
| nDCG@10 | 0.5685 | 0.5441 | 0.6232 | **0.7416** |

联合候选池与 RRF Top-20 的平均 Oracle Recall 均为 0.9286；Reranker 在 Top-10
达到该上限。CPU 下 20 对精排平均 14.01 秒、P95 17.18 秒，吞吐 1.428 pair/s。
质量提升成立，但当前配置不适合交互式在线查询。

### 启动界面

```bash
./.venv/bin/python main.py ui
```

界面包含：

- “单篇入库”：上传并解析一篇 PDF；
- “批量入库”：一次选择多篇 PDF，显示总体进度、新解析数、复用数、逐篇结果
  和失败列表；
- “本地论文库”：查看本地 raw、MinerU 和 canonical 状态。

UI 和 CLI 最终都调用同一个 `parse_paper()`，不存在两套论文解析逻辑。批量
页面使用与 CLI `batch` 相同的容错批处理核心，并写入同一个
`last_batch_report.json`。

## 外部学术发现 MCP

项目提供一个职责独立的 `Academic Discovery MCP`。它只连接外部学术数据源，
不会包装或替代本地的 MinerU 解析、canonical、Chunk、索引和 RAG 流程。

当前暴露四个工具：

- `search_papers`：聚合搜索 Crossref、OpenAlex 和 arXiv；
- `resolve_paper`：根据本地提取的标题返回候选元数据和标题匹配分数；
- `find_fulltext`：根据 DOI、OpenAlex ID 或 arXiv ID 查找开放全文；
- `download_open_pdf`：凭 `find_fulltext` 返回的临时 token，把用户选定的
  开放 PDF 下载到 `data/inbox/`。

下载工具只接受由全文发现步骤登记过的 URL，并检查公网 HTTPS、重定向、文件
体积和 `%PDF-` 文件头。它不会绕过付费墙，也不会自动把下载文件写入 raw、
canonical 或索引。下载后仍由本地 Agent 显式调用现有 `parse_paper()` 入库。

### 启动 stdio MCP

建议配置一个真实联系邮箱。Crossref 和 OpenAlex 会把它用于 polite 请求，
Unpaywall 要求提供邮箱才能查询：

```bash
export LEO_ACADEMIC_CONTACT_EMAIL="researcher@example.com"
./.venv/bin/python main.py academic-mcp
```

通用 MCP 客户端配置示例：

```json
{
  "mcpServers": {
    "leo-academic-discovery": {
      "command": "/absolute/path/to/leo-research-agent/.venv/bin/python",
      "args": [
        "/absolute/path/to/leo-research-agent/main.py",
        "academic-mcp"
      ],
      "env": {
        "LEO_ACADEMIC_CONTACT_EMAIL": "researcher@example.com"
      }
    }
  }
}
```

如需本机 Streamable HTTP：

```bash
./.venv/bin/python -m app.academic_mcp.server \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000
```

默认单个 PDF 最大 100 MiB，可通过正整数环境变量
`LEO_ACADEMIC_MAX_PDF_BYTES` 调整。MCP 的网络搜索结果是外部候选证据，最终
是否接受标题、作者、摘要和 DOI，仍由本地 Agent 的元数据合并规则决定。

### 本地 Agent 核验论文元数据

只通过 stdio MCP 查询候选、不修改本地文件：

```bash
./.venv/bin/python main.py metadata resolve P_2e250a42c5f9
```

使用严格自动规则核验并合并元数据：

```bash
./.venv/bin/python main.py metadata enrich P_2e250a42c5f9
```

自动接受必须同时满足：标题匹配分数至少 `0.98`、至少两个独立数据源、前两名
分差至少 `0.05`，并且作者、年份和 DOI 都存在。不满足时不会改写
`paper.json`，但会把候选与原因保存到：

```text
data/knowledge/metadata_reviews/<paper_id>.json
```

用户或本地 Agent 审核候选后，可以显式选择从 0 开始的候选序号：

```bash
./.venv/bin/python main.py metadata enrich \
  P_2e250a42c5f9 \
  --candidate-index 0
```

接受后会保存 `parser_title`、规范标题、作者、摘要、年份、DOI、venue、外部 ID
和核验来源，并原子重建 `papers.jsonl`。相同 SHA 的 PDF 以后重新运行 MinerU
时会保留已核验元数据。

已有核验元数据的论文可以不访问网络，直接补齐 `work_id/document_id` 并规范化
raw PDF 文件名：

```bash
./.venv/bin/python main.py metadata normalize P_2e250a42c5f9
```

文件名只采用已核验的真实标题，执行 Unicode NFKC 归一化，移除 Windows、
macOS 和 Linux 路径中的非法/控制字符，合并连续空白，固定 `.pdf` 后缀，并将
UTF-8 文件名限制在 220 字节内。原上传名称保留在 `original_filename` 和
`filename_history` 中。

## 数据目录

```text
data/
├── inbox/             # 外部 MCP 下载、等待用户确认入库的开放 PDF
├── raw/
│   └── <paper_id>/
│       └── <verified-title>.pdf
├── parsed/
│   └── <paper_id>/
│       └── mineru/
├── canonical/
│   └── <paper_id>/
│       └── paper.json
├── knowledge/
│   ├── papers.jsonl
│   ├── works.jsonl
│   ├── chunks.jsonl
│   ├── structures/
│   │   └── <document_id>.structure.json
│   ├── chunks/
│   │   └── <document_id>.chunks.json
│   ├── metadata_reviews/
│   ├── last_knowledge_build.json
│   └── last_batch_report.json
├── index/
│   └── bm25.json
└── evaluation/      # 后续自动评测数据
```

## 几十篇论文时，paper.json 能否让 LLM 分清论文？

### 结论

**每篇目录中的文件都叫 `paper.json` 没有问题。LLM 是否能分清论文，与文件名
无关，取决于检索层是否把来源元数据放进每一个 Chunk 和提示词。**

LLM 通常不会直接浏览 `data/canonical` 目录。它只会看到检索器返回的若干段
文本。如果只把正文文本交给 LLM，不附带来源，那么无论文件叫什么，LLM 都
可能把不同论文的内容混在一起。

因此当前每个 Chunk 都包含：

```json
{
  "chunk_id": "D_xxx_cp02_c000123",
  "work_id": "W_xxx",
  "document_id": "D_xxx",
  "paper_id": "P_xxx",
  "title": "论文标题",
  "page_start": 3,
  "page_end": 4,
  "block_ids": ["P_xxx_p003_b005"],
  "section_path": ["3. Framework", "State Transition"],
  "content": "...",
  "content_types": ["paragraph", "equation"]
}
```

向 LLM 组装上下文时，应明确标记来源：

```text
[SOURCE]
paper_id: P_2e250a42c5f9
title: Ephemeris Error Correction for Tracking ...
page: 3
block_ids: P_2e250a42c5f9_p003_b005

正文或公式内容……
[/SOURCE]
```

回答时要求模型引用 `document_id + page`，同时保留兼容 `paper_id`。这样即使一次检索命中十篇论文，LLM
仍能区分每条证据属于哪篇论文。

### 是否需要把文件改名为 `<paper_id>.json`？

在当前目录结构中不需要：

```text
data/canonical/P_aaa/paper.json
data/canonical/P_bbb/paper.json
```

完整路径已经唯一，固定名称也方便程序查找。只有在把多个 JSON 导出到同一个
平面目录时，才应命名为：

```text
P_aaa.paper.json
P_bbb.paper.json
```

## 多论文 RAG 流程

### 阶段 1：建立全局论文目录（v0.2 已完成）

从每个 `paper.json` 提取一条论文级记录，写入：

```text
data/knowledge/papers.jsonl
```

当前记录字段：

- `catalog_schema_version`
- `paper_id`
- `sha256`
- `title`
- `authors`
- `year`
- `doi`
- `page_count`
- `canonical_path`
- `schema_version`
- `parser_name`
- `parser_version`
- `parse_status`
- `quality_issue_count`
- `updated_at`

`paper.json` 是单篇论文事实来源，`papers.jsonl` 是全库目录。

### 阶段 2：恢复内容区域和章节结构（已完成）

结构恢复在 Chunk 之前执行，负责：

- 识别 abstract、main body、appendix、references、biography 等内容区域；
- 恢复罗马数字、数字编号、字母子标题和无编号标题的章节路径；
- 排除页眉页脚、目录、参考文献、致谢和作者简介；
- 纠正被 MinerU 标为标题的 Figure、Table、Algorithm 和 Equation；
- 为公式、图、表记录同章节内最近的上下文 block；
- 使用已核验摘要只辅助识别 PDF 中的摘要 block，不把外部摘要伪装成本地证据。

### 阶段 3：结构化 Chunk（已完成）

当前规则优先保持章节和 block 边界，必要时才按句子拆分超长内容。每个 Chunk
携带 `work_id`、`document_id`、兼容 `paper_id`、章节路径、页码和 block ID；
小父章节和章节内重叠分别存入 `parent_contexts` 与 `overlap_context`，不混入
主证据字段。Chunk ID、顺序和输入指纹都是确定性的。全库产物位于：

```text
data/knowledge/chunks.jsonl
```

### 阶段 4：建立索引（BM25 与 Dense 已完成）

BM25 无需外部服务，标题和章节字段有适度权重。Dense 使用 BGE-M3 单向量和
Qdrant local。两类索引都校验 Chunk 集合指纹，并保存：

```text
chunk_id → work_id → document_id → page/block_ids
```

### 阶段 5：检索与精排（已完成）

当前已支持关键词检索、语义检索、RRF、BGE Cross-Encoder、候选池 Oracle、
`work_id/document_id` 过滤、按 `work_id` 控制证据数量和带页码/block 的结果。
证据上下文组装也已完成。下一步是引用校验、AnswerProvider 和拒答机制，而不是
Agent 或复杂 Query Rewrite。

检索评测器已经独立于具体召回器实现。BM25、Dense、RRF 和 Reranker
必须使用同一问题集、同一 block qrels 和同一指标函数，才能进行有效比较。
Dense 业务层只依赖 `EmbeddingProvider` 协议，不直接导入具体模型 SDK 或 API
客户端；本地 BGE 模型和远程 Embedding API 都必须通过这一接口接入。

当前及后续精排流程：

```text
用户问题
  → 可选元数据过滤
  → BGE-M3 Dense Top-20 + BM25 Top-20
  → RRF 合并去重（已完成）
  → BGE Cross-Encoder 重排（已完成）
  → 按 work_id 去重并控制证据多样性
  → 组装带 SOURCE 标记的 ContextBundle（已完成）
```

“按 work_id 去重并控制证据多样性”既能防止同一论文的多个 PDF 版本重复占满
上下文，也能避免一个长论文挤掉其他论文证据。

### 后续：自动分类和知识抽取

根据项目 taxonomy 自动提取 research task、method family、algorithm、观测量、
卫星星座、实验设置和主要结论。分类结果必须带来源身份，不能生成脱离论文的
全局标签文本。

### 阶段 6：生成带引用的回答

生成模型只能基于检索证据回答，并输出：

- 结论；
- 对应论文标题；
- `paper_id`；
- 页码；
- 必要时的 block 或公式 ID。

如果多篇论文结论不同，应该按论文分别陈述，而不是强行合并成一个结论。

### 阶段 7：自动评测

至少评测：

- 检索是否找到正确论文；
- 页码和引用是否正确；
- 回答是否混淆多篇论文；
- 公式是否来自正确的 `paper_id`；
- 回答中的数值能否在原始 block 中定位；
- 新增或重新解析论文后，旧问题是否退化。

## 增量维护策略

维护几十篇论文时不应全量重跑。建议根据以下字段判断需要更新的阶段：

| 变化 | 需要重跑 |
|---|---|
| PDF SHA 不变 | 不重新入库 |
| MinerU 版本或参数变化 | 重新解析该论文 |
| `paper.json` schema 变化 | 重新标准化该论文 |
| Chunk 规则变化 | 重新 Chunk，可复用 MinerU |
| Embedding 模型变化 | 重新建立向量索引 |
| taxonomy 变化 | 重新分类，不需要重新 MinerU |
| 单个 PDF 删除 | 按 `document_id` 删除文档、Chunk 和索引记录，再更新对应 `work_id` |

未来批量命令应该遍历 `papers.jsonl`，逐篇判断状态，而不是无条件重新处理全部
PDF。

## 当前完成度

| 阶段 | 状态 |
|---|---|
| PDF 哈希入库 | 已完成 |
| PDF 预检查 | 已完成 |
| MinerU 双 venv 调用 | 已完成 |
| Canonical `paper.json` | 已完成 |
| 公式、图、表资产收集 | 已完成 |
| 可重建 `papers.jsonl` | 已完成 |
| 容错批量论文入库 | 已完成 |
| `library` 管理命令 | 已完成 |
| 外部学术发现 MCP | 已完成 |
| `work_id/document_id` 版本归并 | 已完成 |
| 已核验标题 PDF 规范命名 | 已完成 |
| 内容区域与章节结构恢复 | 已完成 |
| 确定性结构化 Chunk | 已完成 |
| 小父章节吸收与章节内重叠 | 已完成 |
| 本地 BM25 索引 | 已完成 |
| 带页码/block 的关键词证据检索 | 已完成 |
| block 级检索评测集与 BM25 基线 | 已完成 |
| BGE-M3 单向量与 Qdrant local Manifest 索引 | 已完成 |
| Dense 基线与逐题退化分析 | 已完成 |
| BM25 + Dense RRF 混合检索基线 | 已完成 |
| 联合候选池 Oracle Recall | 已完成 |
| BGE Cross-Encoder 精排与性能诊断 | 已完成 |
| fast/accurate 长驻检索运行时 | 已完成 |
| EvidenceItem/ContextBundle 与 token budget | 已完成 |
| AnswerProvider 与 OpenAI-compatible 本地模型 | 已完成 |
| claim 级引用校验与 fail-closed 拒答 | 已完成 |
| Context Session、Prompt 布局与缓存诊断 | 已完成 |
| Agentic Session/Topic 与 append-only 事件 | 已完成 |
| 增量 Evidence Registry、Coverage 与有界补充检索 | 已完成 |
| 类别感知 Planner、语义引用验证与 Repair | 已完成 |
| Cache Prefix 诊断与非破坏性 Compaction | 已完成 |
| Agent Harness Policy、状态机、预算与 Stage Trace | 已完成 |
| 自动分类 | 未实现 |
| 大型多轮生成式人工评测集 | 未实现 |

## 代码入口

- `main.py`：唯一 CLI；
- `app/parsing/pipeline.py`：唯一论文解析流程；
- `app/normalization/mineru_adapter.py`：pipeline 内部 MinerU 数据适配器；
- `app/ingestion/ingest.py`：pipeline 内部 PDF 入库；
- `app/ingestion/batch.py`：容错批量入库与批处理报告；
- `app/knowledge/catalog.py`：全局论文目录重建、读取与一致性检查；
- `app/chunking/structure.py`：内容区域、章节路径和资产关系恢复；
- `app/chunking/chunker.py`：确定性结构化 Chunk；
- `app/chunking/builder.py`：全库增量知识层构建；
- `app/indexing/bm25.py`：本地 BM25 倒排索引；
- `app/indexing/dense.py`：带 Manifest 的 Qdrant local 单向量索引；
- `app/retrieval/search.py`：带过滤、去重和引用的证据检索；
- `app/retrieval/dense.py`：BGE-M3 Dense 证据检索；
- `app/retrieval/hybrid.py`：BM25 与 Dense 的 RRF 融合；
- `app/retrieval/reranked.py`：RRF Top-20 精排、来源保留与延迟记录；
- `app/reranking/base.py`：与模型 SDK 解耦的 Reranker 协议；
- `app/reranking/bge.py`：BGE Reranker v2 M3 Cross-Encoder Provider；
- `app/runtime/retrieval.py`：复用模型实例的 fast/accurate 检索运行时；
- `app/context/models.py`：EvidenceItem 与 ContextBundle 稳定契约；
- `app/context/assembly.py`：来源去重、边界校验和 token budget 组装；
- `app/context/session.py`：固定证据快照、完整性指纹与本地 session 存储；
- `app/generation/base.py`：与具体 LLM 服务解耦的 AnswerProvider 协议；
- `app/generation/openai_compatible.py`：本地 Chat Completions 适配器；
- `app/generation/settings.py`：环境变量与本地 `.env` 的安全配置入口；
- `app/generation/validation.py`：上下文完整性与 claim 级引用校验；
- `app/generation/service.py`：检索、生成、确定性引用渲染和失败关闭；
- `app/agentic/`：Session Store、Topic Router、Planner、Reranker、Coverage、
  结构化 Provider、Prompt/Compaction、语义验证、Repair 和端到端编排；
- `app/agentic/harness.py`：RunPolicy、有限状态机、预算、终止原因与安全 Trace；
- `app/evaluation/retrieval.py`：检索问题校验、qrels 映射和排名指标；
- `app/embeddings/base.py`：与本地模型/API 解耦的 Embedding 协议；
- `app/parsing/precheck.py`：pipeline 内部 PDF 预检查；
- `app/storage.py`：JSON 和 JSONL 原子写入；
- `app/ui/gradio_app.py`：调用同一 pipeline 的界面。

这里的“唯一流程”指只有一个对外入口，不是把所有实现堆在一个超大 Python
文件中。入库、预检查和数据适配仍保持为内部模块，便于测试和维护。

## 测试

GitHub Actions 会在每次推送到 `main` 或创建 Pull Request 时，在全新的
Ubuntu 环境中自动安装主环境并运行以下检查：

```bash
./.venv/bin/pytest -q
./.venv/bin/ruff check app main.py tests
./.venv/bin/mypy app main.py
```

CI 不安装 MinerU，也不下载模型。测试使用小型临时 PDF 和模拟 MinerU 输出，
用来验证入库、命令构造、失败处理、标准化、结构恢复、Chunk、BM25、Dense
Manifest、Qdrant 过滤、RRF、Cross-Encoder 适配、上下文来源边界和评测逻辑。
CI 不下载真实 BGE-M3 或 Reranker 权重。

v0.2 的批量验收会自动生成 5 份独立 PDF 测试夹具和 1 份损坏 PDF，验证：

- 5 篇有效论文全部进入 `papers.jsonl`；
- 损坏 PDF 不会阻断其他论文；
- 再次批量导入不会生成重复目录记录；
- 重建前后的 `papers.jsonl` 内容一致；
- 中文、空格、子目录和大写 `.PDF` 均可识别。
- Gradio 批量页面能正确展示进度、成功数、复用数和失败列表。

这些夹具只验证工程流程，不代替真实 LEO 论文语料验收。

## License

项目代码使用 [MIT License](LICENSE)。该许可证不覆盖用户导入的论文、论文
图片、MinerU 解析出的论文内容、第三方模型或第三方依赖。
