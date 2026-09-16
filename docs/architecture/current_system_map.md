# 当前系统架构

Level-2 Content KB 以一篇论文为最小索引生命周期单位；Level-1 Paper KB 是有意的全局 Paper 语义空间。全库文件只承担可重建投影和 BM25 的全局统计，新增/删除 Paper 时 Level-1 metadata vectors 会整体重算，但不会触碰已有论文的 Level-2 Chunk 或 Content Embedding。

## PDF 到 Canonical

```text
PDF -> ingest / SHA-256 去重 -> PDF precheck -> MinerU parse
    -> normalization -> data/canonical/<paper_id>/paper.json
```

`paper.json` 是下游事实源，保留 `source.sha256`、`paper_id`、`identity.document_id/work_id`、metadata、页级 blocks、解析质量和原始定位信息。相同 PDF 不重复解析；原 PDF、解析产物和 Canonical 不因索引更新而删除。

## 单篇论文索引单元

```text
Canonical Parse
  -> Level-1 metadata/abstract chunk（每篇 1 个）
  -> Level-2 content chunks（只来自该 paper）
  -> BM25 lexical projection
  -> BGE-M3 embedding
  -> Qdrant incremental upsert
```

结构恢复和 Chunk 构建位于 `app/chunking/structure.py`、`app/chunking/chunker.py`。`build_knowledge_base` 按每篇文档 fingerprint 复用未变化的 structure/chunks，不会因为新增论文而重新切分旧论文。Level-1 Paper Dense 位于 `app/indexing/paper_dense.py`，Level-2 Content Dense 位于 `app/indexing/dense.py`。

## 存储职责

| 数据 | 当前实现 | 生命周期 |
|---|---|---|
| Canonical | `data/canonical/<paper_id>/paper.json` | 单篇事实源 |
| Structure / Document chunks | `data/knowledge/structures/`、`data/knowledge/chunks/` | 单篇 fingerprint 复用 |
| Chunk projection | `data/knowledge/chunks.jsonl` | 原子重建兼容投影 |
| Paper metadata | `data/knowledge/paper_records.jsonl` / MySQL | 一篇一条 Paper 记录 |
| Chunk BM25 | `data/index/bm25.json` | 全局 IDF 可重算，不触发 Embedding |
| Paper BM25 | `data/index/paper_bm25.json` | Paper-level lexical statistics |
| Chunk Dense | `data/index/qdrant_dense` | 统一 collection，按 paper_id 增量维护 |
| Paper Dense | `data/index/qdrant_papers_dense` | 每篇一个 Level-1 vector；Paper metadata corpus 变化时全量重算 |
| Manifests | `data/index/*dense_manifest.json` | model/policy/paper digest 审计 |

Qdrant collection 是物理容器，不是共同向量化边界。每个 Point payload 至少保留 `paper_id`、`chunk_id`、`level`、`section_path`、`content_hash`、`embedding_model`、`embedding_revision`，以及可回查的 document、page、block 和 content zone。

## 查询路径

```text
Query
  -> Paper BM25 + Paper BGE-M3
  -> Paper RRF
  -> Top-K paper_id
  -> 仅在这些 paper_id 内执行 Content BM25 + Content BGE-M3
  -> Content RRF
  -> BGE Cross-Encoder
  -> Canonical evidence verification
  -> citation validation / fail closed
```

实现位于 `app/retrieval/paper.py`、`app/retrieval/hierarchical.py`、`app/retrieval/reranked.py`。Content Stage 的 BM25 和 Qdrant 查询都在检索内部应用 `paper_id` filter，不允许先对全部论文的全部细粒度 Chunk 做默认全局 Dense Retrieval。

## Agent 与应用边界

CrewAI 只负责上层 Supervisor、Research、Writer、Reviewer 的 Flow/Task 编排，代码位于 `app/orchestration/`。RAG、Evidence、Citation、Workspace、Manuscript 和 Approval 由确定性 domain service 与 capability adapter 执行；Agent 不接触 Qdrant、BM25 文件或模型客户端。

主要入口是 `main.py`、`app/web/runtime.py`、`app/knowledge_engine/index_service.py` 和 `app/evidence/service.py`。

## 关键不变量

1. `paper_id` 是 Paper 生命周期边界；`document_id` 是具体 PDF 身份；`work_id` 只用于论文实体归并。
2. 新增 `P004` 只新增 `P004` 的 metadata chunk、content chunks、lexical postings 和 vectors。
3. 修改或删除某篇论文不会触碰其他论文的 Level-2 Chunk 或 Content Embedding。
4. 新增、删除或修改 Paper metadata 时，Level-1 Paper Dense 会对当前全部 Paper metadata 重新 Embedding；Level-2 仍只处理受影响 Paper。
5. Level-2 旧论文只有在 Canonical/PDF、Chunk policy、Embedding model/revision/text policy 变化，或管理员传入 `force=True` 时才重算。
6. 查询证据必须能回查 `document_id + chunk_id + page + block_ids + content_hash`。
