# LEO Research Agent

面向低轨（LEO）机会信号定位研究的本地论文解析与 RAG 知识库项目。

项目当前已经完成“论文进入知识库之前”的基础链路：

```text
PDF 入库
  → PDF 预检查
  → MinerU 解析
  → MinerU 数据标准化
  → 每篇论文生成一个 paper.json
```

分类、Chunk、索引、检索、重排、LLM 回答和自动评测是下一阶段。本文档同时
说明已经实现的流程和后续维护几十篇论文时应遵守的数据设计。

## 核心设计原则

1. **MinerU 是唯一论文解析引擎**，不并行维护多套 PDF 解析器。
2. **只有一个解析入口**，CLI 和 UI 都调用同一个 `parse_paper()`。
3. **两个虚拟环境严格隔离**，主代码不导入 MinerU 环境中的 Python 包。
4. **原始数据不丢失**，PDF、MinerU 原始输出和 `raw_block` 都保留。
5. **每篇论文独立存储**，由 PDF 内容哈希生成稳定 `paper_id`。
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

强制重新运行 MinerU：

```bash
./.venv/bin/python main.py parse /path/to/paper.pdf --force-mineru
```

查看全部选项：

```bash
./.venv/bin/python main.py parse --help
```

### 启动界面

```bash
./.venv/bin/python main.py ui
```

UI 的“开始解析”和 CLI 调用的是同一个 `parse_paper()`，不存在两套解析逻辑。

## 数据目录

```text
data/
├── raw/
│   └── <paper_id>/
│       └── original.pdf
├── parsed/
│   └── <paper_id>/
│       └── mineru/
├── canonical/
│   └── <paper_id>/
│       └── paper.json
├── knowledge/       # 后续全局论文目录与 chunks
├── index/           # 后续向量、关键词和元数据索引
└── evaluation/      # 后续自动评测数据
```

## 几十篇论文时，paper.json 能否让 LLM 分清论文？

### 结论

**每篇目录中的文件都叫 `paper.json` 没有问题。LLM 是否能分清论文，与文件名
无关，取决于检索层是否把来源元数据放进每一个 Chunk 和提示词。**

LLM 通常不会直接浏览 `data/canonical` 目录。它只会看到检索器返回的若干段
文本。如果只把正文文本交给 LLM，不附带来源，那么无论文件叫什么，LLM 都
可能把不同论文的内容混在一起。

因此后续每个 Chunk 必须包含：

```json
{
  "chunk_id": "P_xxx_c000123",
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

回答时要求模型引用 `paper_id + page`。这样即使一次检索命中十篇论文，LLM
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

## 后续多论文 RAG 流程

### 阶段 1：建立全局论文目录

从每个 `paper.json` 提取一条论文级记录，写入：

```text
data/knowledge/papers.jsonl
```

建议字段：

- `paper_id`
- `sha256`
- `title`
- `authors`
- `year`
- `doi`
- `canonical_path`
- `parser_version`
- `schema_version`
- `indexed_at`

`paper.json` 是单篇论文事实来源，`papers.jsonl` 是全库目录。

### 阶段 2：自动分类和知识抽取

根据项目 taxonomy 自动提取：

- research task；
- method family；
- algorithm；
- observation type；
- 卫星星座；
- 信号和观测量；
- 数据集、仿真和实验设置；
- 主要结论。

分类结果必须带 `paper_id`，不能生成脱离来源的全局标签文本。

### 阶段 3：结构化 Chunk

Chunk 不应简单按固定字符数切分。建议：

- 优先按标题和段落边界；
- 保留 section path；
- 连续短段可以合并；
- 公式与解释段绑定在同一 Chunk 或建立引用关系；
- 图表与 caption 绑定；
- 参考文献和 Biography 默认不进入正文索引；
- 每个 Chunk 始终携带 `paper_id` 和页码。

全库 Chunk 可保存为：

```text
data/knowledge/chunks.jsonl
```

### 阶段 4：建立多种索引

建议至少保留三种能力：

1. 向量索引：语义问题；
2. BM25/全文索引：术语、公式变量、卫星名称和精确关键词；
3. 元数据过滤：`paper_id`、年份、任务类别、算法和星座。

索引中的每条记录必须保存：

```text
chunk_id → paper_id → page/block_ids → canonical_path
```

不要只存 embedding 和正文，否则无法稳定引用或删除单篇论文。

### 阶段 5：检索与重排

推荐流程：

```text
用户问题
  → 可选元数据过滤
  → 向量召回 + BM25 召回
  → 合并去重
  → reranker 重排
  → 按 paper_id 控制证据多样性
  → 组装带 SOURCE 标记的上下文
```

“按 paper_id 控制证据多样性”可以防止一个长论文占满全部上下文，也能避免
LLM 把来自不同论文的结论误认为同一实验结果。

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
| 单篇论文删除 | 按 `paper_id` 删除目录、Chunk 和索引记录 |

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
| 批量论文目录 | 未实现 |
| 自动分类 | 未实现 |
| Chunk | 未实现 |
| 向量/BM25 索引 | 未实现 |
| 检索与重排 | 未实现 |
| LLM 回答和引用 | 未实现 |
| 自动评测 | 未实现 |

## 代码入口

- `main.py`：唯一 CLI；
- `app/parsing/pipeline.py`：唯一论文解析流程；
- `app/normalization/mineru_adapter.py`：pipeline 内部 MinerU 数据适配器；
- `app/ingestion/ingest.py`：pipeline 内部 PDF 入库；
- `app/parsing/precheck.py`：pipeline 内部 PDF 预检查；
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
用来验证入库、命令构造、失败处理和标准化逻辑。真实论文的 MinerU 解析仍在
本地专用环境中运行。

## License

项目代码使用 [MIT License](LICENSE)。该许可证不覆盖用户导入的论文、论文
图片、MinerU 解析出的论文内容、第三方模型或第三方依赖。
