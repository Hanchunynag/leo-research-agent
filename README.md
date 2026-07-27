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
  → 使用 BGE Cross-Encoder 精排并返回页码/block 证据
```

现在支持容错批量入库、论文身份归并、真实标题规范命名、增量知识层构建、
BM25、BGE-M3 单向量召回、Qdrant local、RRF、候选池 Oracle 和 Cross-Encoder
精排。LLM 上下文组装、证据引用、拒答、分类和生成式回答评测仍是下一阶段。

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
下一步是上下文组装、证据引用和拒答机制，而不是 Agent 或复杂 Query Rewrite。

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
  → 组装带 SOURCE 标记的上下文
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
| 自动分类 | 未实现 |
| LLM 回答和引用 | 未实现 |
| 生成式回答自动评测 | 未实现 |

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
Manifest、Qdrant 过滤、RRF、Cross-Encoder 适配和评测逻辑。CI 不下载真实
BGE-M3 或 Reranker 权重。

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
