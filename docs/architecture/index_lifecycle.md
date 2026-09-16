# Document-level incremental indexing

## 设计约束

Level-2 Content KB 的一篇论文就是一个独立 indexing unit；Level-1 Paper KB 则是有意维护的全局 Paper 语义空间。两层的 Chunk 生命周期和 Dense 生命周期分开管理。

```text
Paper P
  Canonical Parse
    -> Level-1 metadata chunk（恰好 1 个）
    -> Level-2 content chunks（0..N 个）
    -> BM25 lexical projection
    -> BGE-M3 embedding
    -> vector database incremental upsert
```

## 增量判定

`app/chunking/builder.py` 为每篇文档单独检查 structure/chunk fingerprint，输出 `changed_paper_ids`。Content Dense manifest 将每篇论文的 embedding input digest 存在 `paper_digests` 中：

- 新 `paper_id` 只加入自己的 Level-2 Content vectors；
- 变化 `paper_id` 只删除并重写自己的 Level-2 Content vectors；
- 不再存在的 `paper_id` 只删除自己的 Level-2 Content vectors，不调用 embedding provider；
- digest 未变化的论文保留原 Level-2 vectors，embedding provider 不会收到它们的 Content 文本。

Paper Dense 不采用上述单 Paper 增量策略：每篇只生成一个 `chunk_id=<paper_id>_metadata` 的 Level-1 vector，但只要 Paper metadata corpus 的 `papers_digest` 发生变化（例如新增 P004、删除 Paper 或 metadata 更新），就对当前全部 Paper metadata 文本重新调用 BGE-M3。只有 Paper corpus 完全不变时才复用 Level-1 vectors。

## 允许触发旧论文重处理的条件

1. Level-1 Paper metadata corpus 新增、删除或字段变化；
2. Level-2 原 PDF 或 Canonical content 变化；
3. `CHUNK_POLICY_VERSION` 变化；
4. Embedding model、revision、artifact fingerprint、归一化或 text policy 变化；
5. 管理员显式传入 `force=True`。

Manifest schema、tokenizer policy 和 collection 配置也进入兼容性门禁。旧 manifest 如果缺少按论文 `paper_digests`，系统拒绝静默全量重算，要求管理员显式执行 force rebuild。

## BM25 与 Dense 分离

`data/index/bm25.json` 和 `data/index/paper_bm25.json` 可基于当前投影重建全局 postings/IDF。全局 BM25 重算不等于重新 Chunk，也不允许调用 Dense provider 重新计算旧论文向量。Dense 只有受影响 Paper 的文本进入 `embed_documents()`。

## 删除与更新

Qdrant 使用统一 collection，但 Point ID 分别由 `paper_id:chunk_id` 和 `paper_id` 稳定派生，避免不同论文都拥有 `C001` 时发生覆盖。删除操作按 payload `paper_id` 清除该论文 Point，再对仍存在的该论文执行 upsert；其他论文的 Point 不变。

所有 Point payload 都保留：

```text
paper_id / chunk_id / level / section_path
content_hash / embedding_model / embedding_revision
```

以及可回查的 document、page、block、content zone 和 text hash。

## 查询生命周期

```text
Stage 1: Query -> Paper BM25 + Paper BGE-M3 -> RRF -> Top-K paper_id
Stage 2: Query -> filtered Content BM25 + Content BGE-M3 -> RRF
         -> Cross-Encoder -> Verified Evidence -> Citation Validation
```

Stage 2 的 `paper_id` 过滤在 BM25 评分和 Qdrant 查询阶段执行。默认入口不会在全部论文的全部细粒度 Chunk 上直接启动 Dense Retrieval。

## 运维命令

```bash
uv run python main.py hierarchical build
uv run python main.py hierarchical build --force
uv run python main.py hierarchical search "你的问题"
```

`--force` 只用于管理员明确要求的 rebuild；正常导入 PDF 会复用未变化论文的 structure、Chunk 和 vectors。
