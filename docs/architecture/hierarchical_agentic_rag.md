# Hierarchical Agentic RAG 实施说明

本次升级保留既有 `MinerU → Canonical → Semantic Chunk → BM25 + BGE-M3/Qdrant → RRF → Cross Encoder → Evidence/Claim 校验` 主链，只增加 Paper Knowledge Layer 和结构化审计存储。

## 两层检索

```text
User Query
  → fixed LangGraph preparation nodes
  → Research Intent
  → Paper BM25 + Paper BGE-M3
  → Paper RRF
  → Candidate Paper IDs
  → Chunk BM25 + Chunk BGE-M3 (Qdrant payload filter)
  → Chunk RRF
  → BGE Cross Encoder
  → Evidence validation and generation
```

Chunk 检索的候选论文过滤发生在 BM25 评分阶段和 Qdrant 查询阶段，不会先做全库 Top-K 再事后过滤。Candidate Paper 为空时不会执行全库 Chunk 检索。

## 数据职责

- MySQL：`papers`、`paper_versions`、`paper_sections`、`chunks`、`query_history`、`citation_records`、`temporary_evidence`、`research_jobs`、`index_epochs`。
- Qdrant：Paper/Chunk 向量及必要 payload；Paper 与 Chunk 使用不同 collection。
- JSON/JSONL：可重建兼容投影，MySQL 未启用或暂时不可用时继续工作。
- BM25：由 Paper/Chunk 投影派生并可重建，不把向量写入 MySQL。

## Agent 边界

LangGraph 仍是薄编排层。翻译、引用解析、分类、Planner、Scope、论文解析元数据等是固定节点；LLM 控制器只能请求 `knowledge.retrieve`，不能自由选择工具、索引或检索范围。外部论文搜索只由既有 Coverage/Bootstrap 固定分支触发，摘要证据标记 `abstract_only=true`，全文通过既有后台 Job 异步解析入库。

## 版本与可恢复性

Dense manifest 和 `index_epochs` 记录 source fingerprint、BGE-M3 model/revision、tokenizer/chunker version。Canonical 是事实源，索引和 MySQL 都可从 Canonical 重建；MySQL 失败时不会覆盖或删除现有 JSON/JSONL 产物。
