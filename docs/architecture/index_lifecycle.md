# 索引生命周期

## 当前两套生命周期

### Legacy 全量路径

```text
paper.json
  -> build_knowledge_base
     -> structure files
     -> per-document chunks
     -> chunks.jsonl (全量原子重写)
     -> bm25.json (全量原子重写)
  -> build_dense_index
     -> temporary Qdrant directory
     -> qdrant_dense replacement + dense_manifest.json
```

实现分别位于 [`app/chunking/builder.py`](../../app/chunking/builder.py)、[`app/indexing/bm25.py`](../../app/indexing/bm25.py) 和 [`app/indexing/dense.py`](../../app/indexing/dense.py)。BM25 读取时用 chunks digest 拒绝 stale index；Dense 用 manifest 校验模型、revision、text policy 和 chunks digest。

### GraphRAG 增量路径

```text
chunks.jsonl
  -> create pending epoch
  -> diff stable chunk keys
  -> enqueue lexical/dense/graph operations
  -> FTS5 sync
  -> chunk + entity Qdrant sync
  -> graph extraction + Neo4j persistence
  -> aggregate relations
  -> communities + reports + embeddings
  -> source validation
  -> activate epoch (commit point)
```

实现位于 [`app/index_registry/coordinator.py`](../../app/index_registry/coordinator.py) 和 [`app/index_registry/store.py`](../../app/index_registry/store.py)。pending 失败后标记 failed，旧 active epoch 保持可读。Outbox 存在并可重试，但 `KnowledgeSyncService.sync` 当前在同一进程串行执行全部后端。

## 操作矩阵

| 操作 | Canonical | papers.jsonl | Chunk/BM25 | 旧 Dense | FTS5/增量 Dense/Graph/社区 |
|---|---:|---:|---:|---:|---:|
| CLI `parse` | 是 | 是 | 否 | 否 | 否 |
| CLI `batch` | 是 | 是 | 否 | 否 | 否 |
| Web upload | 是 | 是 | 是 | 是 | 否 |
| `knowledge build` | 读 | 否 | 是 | 否 | 否 |
| `dense build` | 否 | 否 | 读 Chunk | 是 | 否 |
| `knowledge sync` | 否 | 否 | 读 Chunk | 否 | 是 |
| metadata enrich | 更新 | 否 | 否 | 否 | 否 |
| delete paper | 不支持 | 不支持 | 不支持 | 不支持 | 仅底层 diff 能表达删除 |

因此当前没有一个“新增/更新/删除都原子传播到全部索引”的生命周期。Web 上传只同步自身使用的 legacy 查询栈。

## 目标状态机

1. `CorpusChanged`：Canonical Corpus 已通过原子写入；记录 corpus version 和受影响 document。
2. `GenerationPending`：建立 `IndexGeneration`，绑定 workspace scope/corpus version。
3. `ProjectionBuilding`：各 Adapter 构建 lexical/dense/graph 投影，报告计数和 checksum。
4. `Validation`：验证 document/chunk 数、locator 可回查、抽样检索和后端健康。
5. `Active`：统一提交 generation；查询固定读取 request.scope_version 对应 generation。
6. `Failed`：保留旧 active generation，失败 generation 可重试或回收。
7. `Superseded`：超过保留策略后清理后端投影；Canonical 数据删除需要单独、显式、可审计授权。

## 新增、更新、删除的目标规则

- 新增：先 canonical，再创建 generation；只有 generation active 后 Workspace scope 才可引用文档。
- 更新：PDF 内容变化必须产生新 Document；元数据/Chunk policy 变化产生新 corpus/generation，不就地改活跃索引。
- 删除：阶段一保持“不支持”。阶段二先实现 Workspace 解绑（可恢复）与后端失效，再单独设计物理删除；不得把 Neo4j/Qdrant 删除当作 canonical 删除成功。
- 幂等：同一 generation/object/hash 的操作必须可安全重放。现有 `make_operation` 和 `INSERT OR IGNORE` 可继续作为 Adapter 内机制。

## 阶段二校验门槛

- generation 中 document/chunk 数与 scope snapshot 一致；
- 每个检索结果可回查 `document_id + chunk_id + block_ids`；
- 所有必需后端完成且 source validation 通过；
- active 切换失败不影响旧 generation；
- Web/CLI 只能调用统一 lifecycle service，不直接调用各 builder。

## 阶段二落地状态

[`KnowledgeIndexService`](../../app/knowledge_engine/index_service.py) 已成为 Web 解析后的唯一下游编排入口；[`LocalRAGWebRuntime.parse_pdf`](../../app/web/runtime.py) 不再自行拼接 catalog/Chunk/Dense builder。操作状态写入 `data/knowledge/index_operations/*.json`，失败保留 `FAILED` 记录。

LightRAG generation 由 [`IndexGenerationRepository`](../../app/knowledge_engine/generations.py) 持久化为 `building -> validating -> active/failed`，旧 active 激活新代际时转 `retired`，可调用 [`KnowledgeIndexService.rollback`](../../app/knowledge_engine/index_service.py) 恢复。当前 `IG_419c4e228f0dd1a8` 为 active shadow；7 个历史失败代际保留审计，0 个悬空 validating。增量 update/delete 的 `full_rebuild=False` 由 [`tests/test_stage2_lightrag_engine.py`](../../tests/test_stage2_lightrag_engine.py) 冻结。
