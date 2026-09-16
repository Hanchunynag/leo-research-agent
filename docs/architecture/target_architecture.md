# 当前目标架构

```text
PDF / Canonical Corpus
  -> document-level indexing lifecycle
     -> Paper metadata BM25 + BGE-M3
     -> Paper content BM25 + BGE-M3
  -> hierarchical retrieval
     -> Paper RRF
     -> Top-K paper_id
     -> filtered Content RRF
     -> Cross-Encoder
     -> Evidence Governance
     -> Grounded Answer / Scholar DraftPatch
```

MinerU -> Canonical 是唯一 PDF 事实链。`paper.json` 是事实源；`KnowledgeIndexService` 编排导入后的更新；`build_knowledge_base` 负责按文档复用结构/Chunk 和 BM25 投影；`build_paper_dense_index` 负责 Paper metadata corpus 变化时全量重建 Level-1 Dense，`build_dense_index` 负责按 `paper_id` 增量维护 Level-2 Content Dense。

Paper-level lexical/semantic candidates 先决定论文集合，Content-level lexical/semantic retrieval 只在该集合中进行。RRF 只融合 rank；Cross-Encoder 只处理 Content RRF 候选。Evidence 必须经历 candidate、verified、selected 三阶段，Claim-Citation 验证失败时 fail closed。

CrewAI 位于应用层，编排 Supervisor、Research、Writer、Reviewer 的 bounded Flow。它调用 capability/service contract，不实现索引、不管理 vectors、不绕过 evidence verification 和 human approval。

每个 Dense manifest 记录 model、revision、artifact fingerprint、embedding text policy、chunk policy、tokenizer、vector dimension、collection 和每篇 paper digest。缺少 per-paper digest 的旧 manifest 不会被静默当作增量索引；管理员必须显式 force rebuild。索引失败不会删除 Canonical，也不影响未变化论文的既有 vectors。
