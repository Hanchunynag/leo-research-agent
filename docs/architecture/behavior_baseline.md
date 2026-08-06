# 阶段一行为基线

可版本化摘要位于 [`tests/baselines/stage1_behavior_baseline.json`](../../tests/baselines/stage1_behavior_baseline.json)。完整的 21 题 Top-10 位于本地 `data/evaluation/{bm25,dense,rrf,reranker}_baseline.json`；这些运行产物被 `.gitignore` 排除，因此摘要记录其 SHA-256，并内嵌代表题 Q001 的四条 Top-10 排名。

## 当前数据快照

- Canonical Document 7；Section 122；searchable Block 797；Chunk 135。
- 旧 BM25 document 135；旧 Dense chunk 135，BGE-M3 1024 维，revision `5617a9...`。
- 当前本地 `index_registry.sqlite3` 没有 epoch/active epoch，因此 GraphRAG 端到端关系查询、图计数和图延迟无法在本次只读基线上实测，标记待确认。
- Reranker 21 题：MRR 0.710979、NDCG@10 0.741598、Recall@10 0.928571；总延迟 mean 14146.983 ms、P50 14346.666 ms、P95 17299.847 ms。

## 引用与生成

本地 session 历史有 8 个 answer/validation event，其中 1 个 answerable；5/8 validation 的 structural.valid 为 true。14 个 claim_results 对应 5 个 CitationRecord，原始比值 0.357143。该历史包含 fail-closed 拒答，不能解释成“成功回答的 citation recall”；唯一 answerable 样本包含 5 个 claim 和 5 个 citation。

LLM 调用次数和 Token 使用量没有跨 routing/coverage/generation/repair/graph/community 统一累计或持久化，因此基线为 `null`（未测量），不是 0。`OpenAICompatibleAnswerProvider` 和 `AgenticReasoningProvider` 能读取单次 response usage，但 `AgenticRAGService` 只把 generation usage 用于 prompt-cache diagnostics。

## 十二项覆盖结论

1. PDF 解析：`test_pipeline*` 自动化。
2. Canonical Chunk：结构/边界/确定性/复用测试自动化。
3. BM25：自动化 + 21 题完整快照。
4. Dense：自动化 + 21 题完整快照。
5. 图关系：direct/inferred/none 分支自动化；真实 Neo4j E2E 待 active epoch 环境。
6. RRF/Reranker：自动化 + 21 题完整快照与 Reranker 延迟。
7. 普通问答：generation/agentic 自动化。
8. 关系问答：Graph 关系分支已冻结；端到端生成待确认。
9. 引用映射：source/evidence/document/chunk/page/block 校验自动化。
10. 新增论文：pipeline 和 Web parse->catalog->Chunk/BM25->Dense 自动化。
11. 删除论文：当前无外部行为，冻结为“不支持”；不能在阶段一新增 API。
12. 重启任务状态：新增测试确认新 `JobManager` 无法恢复旧 job_id，即当前为 volatile。

## 更新基线的规则

只有明确批准外部行为变化时才能更新 Top-K/指标。更新需同时提交：原因、模型 revision、corpus/chunks digest、完整产物 SHA-256、索引计数、延迟环境和测试结果。无法测量的字段保留 `null` 并写明原因。
