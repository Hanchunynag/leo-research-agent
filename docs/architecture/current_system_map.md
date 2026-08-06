# 当前系统地图（阶段一只读审计）

审计日期：2026-08-06。本文只描述仓库中能定位到的实现；无法从代码或当前本地产物确认的内容标为“待确认”。

## 1. PDF 上传、入库与 MinerU 解析

CLI `parse`/`batch` 从 [`main.py`](../../main.py) 分发到 `parse_paper`/`batch_parse_directory`。Web `POST /api/papers/upload` 在 [`app/web/api.py`](../../app/web/api.py) 校验扩展名、200 MB 上限和 `%PDF-` 文件头，写入 `data/runtime/web_uploads/upload_*` 临时目录，再提交给进程内 `JobManager`。

统一业务入口是 [`app/parsing/pipeline.py`](../../app/parsing/pipeline.py) 的 `parse_paper`：

1. `ingest_paper`（[`app/ingestion/ingest.py`](../../app/ingestion/ingest.py)）校验 PDF、计算 SHA-256，以 `P_<sha前12位>` 为 `paper_id`，复制到 `data/raw/<paper_id>/`；相同内容复用已有 PDF。
2. `precheck_pdf`（[`app/parsing/precheck.py`](../../app/parsing/precheck.py)）执行轻量 PDF 预检查。
3. `find_mineru_artifacts` 查找可复用的 `*_content_list_v2.json` 和唯一 `*_middle.json`。无可复用产物时，`run_mineru` 只使用 `.venv-mineru` 或显式配置的 MinerU 可执行文件。每篇文档使用 `FileLock(data/parsed/<paper_id>/mineru/.parse.lock)` 防止并发重复解析。
4. `build_canonical_document`（[`app/normalization/mineru_adapter.py`](../../app/normalization/mineru_adapter.py)）把 MinerU 页/块转换为 canonical blocks，保留 `raw_block`。
5. `parse_paper` 补充 source、identity、precheck、pipeline、formula/table/figure 轻量视图并原子写入 `data/canonical/<paper_id>/paper.json`。

MinerU 原始资产保留在 `data/parsed`，Canonical 下游事实入口是 `paper.json`。重新解析相同 SHA 时，状态为 `verified`/`selected` 的外部元数据会被保留。

## 2. paper.json、papers.jsonl、Section、Chunk、Block

- Block：`convert_block`（[`app/normalization/mineru_adapter.py`](../../app/normalization/mineru_adapter.py)）生成 `P_*_pNNN_bNNN`，包含页码、阅读顺序、bbox、文本/公式/表格/图片、质量状态和完整 `raw_block`。Block 是 `paper.json.blocks` 的一部分。
- `paper.json`：`build_canonical_document` 生成基础对象，`parse_paper` 增补 source/identity/pipeline 后由 `write_json_atomic` 写入。
- `papers.jsonl`：`rebuild_catalog`（[`app/knowledge/catalog.py`](../../app/knowledge/catalog.py)）扫描全部 `data/canonical/*/paper.json` 后原子重写 `data/knowledge/papers.jsonl`。CLI `parse` 的 `main` 包装层、批处理和 Web 上传都会调用它；底层 `parse_paper` 函数自身只写单篇 canonical 文档。
- Work/Document identity：`build_identity`（[`app/knowledge/identity.py`](../../app/knowledge/identity.py)）生成具体 PDF 的 `document_id`；只有可靠元数据满足规则时才有 `work_id`。`rebuild_work_catalog`（[`app/knowledge/works.py`](../../app/knowledge/works.py)）生成 `works.jsonl`。
- Section：`build_structure`（[`app/chunking/structure.py`](../../app/chunking/structure.py)）从 canonical blocks 恢复内容区域、标题层级、章节路径和资产邻接关系，写入 `data/knowledge/structures/<document_id>.structure.json`。
- Chunk：`build_chunks`（[`app/chunking/chunker.py`](../../app/chunking/chunker.py)）只消费 searchable blocks，不跨 Section/内容区，执行小父章节吸收与同章节 overlap，生成 `D_*_cp02_cNNNNNN`。单文档集合写入 `data/knowledge/chunks/*.chunks.json`。
- 全库 Chunk：`build_knowledge_base`（[`app/chunking/builder.py`](../../app/chunking/builder.py)）按 fingerprint 复用或重建结构/Chunk，原子重写 `data/knowledge/chunks.jsonl`，随后重建旧 BM25。

当前存在一个身份风险：`chunk_id` 含序号，会随上游切分变化；增量索引另由 `stable_chunk_key`（[`app/index_registry/diff.py`](../../app/index_registry/diff.py)）基于结构身份生成稳定键。

## 3. BM25、Dense、RRF、Reranker

| 能力 | 创建位置 | 调用位置 |
|---|---|---|
| 旧 BM25 JSON | `build_bm25_index`/`write_bm25_index`，[`app/indexing/bm25.py`](../../app/indexing/bm25.py)；由 `build_knowledge_base` 全量重写 | `search_evidence`，[`app/retrieval/search.py`](../../app/retrieval/search.py) |
| 增量 FTS5 BM25 | `sync_lexical`，[`app/indexing/lexical_fts.py`](../../app/indexing/lexical_fts.py)；由 `KnowledgeSyncService.sync` 调用 | `search_lexical_evidence`，[`app/retrieval/lexical_fts.py`](../../app/retrieval/lexical_fts.py) |
| 旧 Dense Qdrant local | `build_dense_index`，[`app/indexing/dense.py`](../../app/indexing/dense.py)，路径 `data/index/qdrant_dense` | `search_dense_evidence`，[`app/retrieval/dense.py`](../../app/retrieval/dense.py) |
| 增量 Dense Qdrant | `sync_incremental_dense`，[`app/indexing/incremental_dense.py`](../../app/indexing/incremental_dense.py)，路径 `data/index/qdrant_graphrag` | `search_incremental_dense`/`search_community_dense`，[`app/retrieval/incremental_dense.py`](../../app/retrieval/incremental_dense.py) |
| RRF | 无独立持久索引；`reciprocal_rank_fusion` 融合旧 BM25 与旧 Dense | `search_hybrid_evidence`，[`app/retrieval/hybrid.py`](../../app/retrieval/hybrid.py) |
| Cross-Encoder | `BGERerankerProvider`，[`app/reranking/bge.py`](../../app/reranking/bge.py) | `search_reranked_evidence`，[`app/retrieval/reranked.py`](../../app/retrieval/reranked.py)，以及 Agentic `DirectAnswerReranker` |

`RetrievalRuntime`（[`app/runtime/retrieval.py`](../../app/runtime/retrieval.py)）缓存 embedding/reranker provider，`fast` 调 RRF，`accurate` 调 Cross-Encoder。GraphRAG 在线路径不使用这个组合，而由 `GraphRAGRetrievalRuntime` 直接调 FTS5、增量 Dense、GraphRetriever 和社区检索。

## 4. GraphRAG 全流程

入口是 CLI `knowledge sync`/`migrate-to-graphrag`，都调用 [`app/index_registry/coordinator.py`](../../app/index_registry/coordinator.py) 的 `KnowledgeSyncService.sync`；后一个名称当前只是同一路径的兼容入口，不会重新解析 PDF。

1. `IndexRegistryStore.create_epoch` 创建 pending epoch；`IndexCoordinator.plan` 用 `diff_chunks` 分类 added/dense_changed/graph_changed/changed/deleted/unchanged，并写 lexical/dense/graph outbox。
2. `sync_lexical` 更新带 epoch 可见性的 SQLite FTS5。
3. `sync_incremental_dense` 更新版本化 Qdrant named vector；实体向量由 `sync_entity_embeddings` 更新。
4. `extract_chunk_graph`（[`app/graph/extraction.py`](../../app/graph/extraction.py)）调用结构化 LLM，按 graph text hash、模型、prompt、ontology 生成可缓存 extraction。
5. `EntityResolver`（[`app/graph/resolution.py`](../../app/graph/resolution.py)）执行类型感知实体解析；`persist_chunk_graph`（[`app/graph/persistence.py`](../../app/graph/persistence.py)）MERGE Work/Document/Section/Chunk/Entity/RelationClaim 及来源边。
6. `rebuild_aggregate_edges`（[`app/graph/aggregation.py`](../../app/graph/aggregation.py)）重建 Entity 间 RELATED 聚合边。
7. `detect_communities`/`persist_communities`（[`app/graph/communities.py`](../../app/graph/communities.py)）使用固定 seed 42 生成社区；`generate_community_report` 与 `embed_community_reports`（[`app/graph/reports.py`](../../app/graph/reports.py)）生成并向量化报告。
8. `validate_graph_sources` 成功后才 `activate_epoch`；失败则 `fail_epoch`，旧 active epoch 仍可见。
9. 在线 `GraphRAGRetrievalRuntime.retrieve_multi`（[`app/runtime/graphrag.py`](../../app/runtime/graphrag.py)）针对多查询执行 lexical、dense、graph_direct/graph_path、可选 community，最后由 `weighted_rrf` 跨查询/路由融合。

图关系查询由 `GraphRetriever`（[`app/graph/retrieval.py`](../../app/graph/retrieval.py)）提供实体链接、local、direct relationship 和二跳 inferred path。没有关系时返回 none；不会把“分别有定义”当作关系证据。

## 5. Agentic Service

[`app/agentic/service.py`](../../app/agentic/service.py) 的 `AgenticRAGService.answer` 是有界编排：

1. `TopicRouter.route` 路由 same/related/new topic，并可改写依赖上下文的问题。
2. `QueryPlanner.plan` 生成 intent、子问题、证据要求和检索 query。
3. 只有 GraphRAG runtime 执行 `AdaptiveQueryExpander.expand` 和 `QueryDriftValidator.validate`。
4. `_retrieve_queries`：GraphRAG 调 `retrieve_multi`；legacy 逐 query 调 `RetrievalRuntime.retrieve(mode="fast")`。
5. `DirectAnswerReranker.rerank`，随后 `AgenticSessionStore.register_evidence` 以 `chunk_id` 注册稳定 topic-local `E001...`。
6. `_coverage` 先运行 deterministic coverage，再尝试 reasoning provider；不足时最多按 Harness budget 追加检索。
7. `_final_evidence`/`CoverageAwareEvidenceSelector` 执行直接性、覆盖和 MMR 选择，`assemble_context_bundle` 建 Context。
8. reasoning provider 生成结构化 claims；`validate_answer_draft` 做结构/引用校验，`_semantic_validate` 做蕴含、类别和问题对齐判断。
9. 语义校验可消耗剩余检索轮；仍失败时最多一次 answer repair；最终 fail closed。
10. answer、validation、state_delta 追加到 SQLite session store，Harness 记录阶段、预算、终止原因和诊断。

Harness 已能感知 Stage/State/Budget/Recovery，但当前 `AgenticRAGService` 仍直接感知具体 runtime 能力，尚未达到目标隔离。

## 6. 后端判断点

- `main.retrieval_runtime_from_args`：`retrieval_mode == "graphrag"` 时构造 `GraphRAGRetrievalRuntime`，否则构造 `RetrievalRuntime`。
- `main.main` answer 分支：`retrieval_mode in {"graphrag", "agentic"}` 决定是否走 Agentic；GraphRAG 还要求 provider 暴露 `chat_completion`。
- `GraphRAGRetrievalRuntime.is_graphrag = True`。
- `AgenticRAGService` 在 `_retrieve_queries`、query expansion、分路由 Harness trace、关系 none guard、focused query 限制等位置多次使用 `getattr(runtime, "is_graphrag", False)`。
- `LocalRAGWebRuntime._build_retrieval_runtime` 固定构造旧 `RetrievalRuntime`，Web 没有 GraphRAG 选择分支。

这些判断是阶段二改由 `KnowledgeEngine` capability/contract 消化的主要目标。

## 7. 论文新增、更新、删除的索引生命周期

- 新增/更新（CLI parse）：`parse_paper` 写 raw/parsed/canonical，随后 `main` 调用 `rebuild_catalog`；不自动重建 Chunk 或索引。
- 新增/更新（batch）：逐篇 `parse_paper`，结束时重建 `papers.jsonl`；不自动构建 Chunk/Dense/Graph。
- 新增/更新（Web）：`parse_paper` -> `rebuild_catalog` -> `build_knowledge_base` -> `build_dense_index`。它重建旧 BM25 和旧 Dense。
- 增量 GraphRAG：必须另行执行 `knowledge sync`；registry epoch 是 FTS5/Qdrant/Neo4j 的提交点。
- 元数据更新：[`app/knowledge/metadata_enrichment.py`](../../app/knowledge/metadata_enrichment.py) 可更新 `paper.json`，但不会自动触发下游索引。
- 删除：仓库没有论文删除 API、CLI 或领域服务。`diff_chunks`/`KnowledgeSyncService` 能在“chunks 输入已不含旧 chunk”时使后端版本失效，但没有从 raw/parsed/canonical 到 catalog/Chunk 的受控删除入口。当前外部行为基线是“不支持删除”。

## 8. Web 上传是否更新所有索引

否。`LocalRAGWebRuntime.parse_pdf` 更新 canonical、catalog、结构/Chunk、旧 BM25、旧 Dense；不调用 `KnowledgeSyncService.sync`，因此不更新 FTS5、增量 Qdrant、Neo4j 实体关系或社区。并且 Web answer 固定使用旧 `RetrievalRuntime`，所以 Web 内部读写暂时自洽，但与 CLI GraphRAG 形成两套索引世界。

## 9. 后台任务持久化

Web `JobManager`（[`app/web/jobs.py`](../../app/web/jobs.py)）用 `ThreadPoolExecutor` 和内存字典 `_jobs` 保存 queued/running/result/error/events。没有数据库或恢复逻辑；新进程无法读取旧 job_id。Agentic session/event/evidence 与 GraphRAG index epoch/outbox 是 SQLite 持久化的，但它们不是 Web Job 持久化。

## 10. 引用校验和 Evidence 结构

- 检索结果仍是自由形状 `dict`，通常包含 work/document/chunk、页码、block_ids、content 和后端分数。
- `EvidenceItem`/`ContextBundle` 在 [`app/context/models.py`](../../app/context/models.py) 中表达最终上下文，`assemble_context_bundle` 为 evidence 分配 `S1...` source_id。
- Agentic store 的 `evidence_registry` 持久化 `E001...` evidence_id 和原始 evidence JSON；当前 schema 只表达 active/reused，没有 Candidate/Verified/Selected 三态。
- `AnswerClaim` 同时携带 `source_ids` 与 `evidence_ids`。`validate_answer_draft`（[`app/generation/validation.py`](../../app/generation/validation.py)）拒绝未知 source、无引用 claim、非法 locator 和 source/evidence 映射错误，并生成 `CitationRecord`。
- `AgenticRAGService._restrict_draft_to_context` 防止引用未进入本轮 Context 的历史 evidence；`_semantic_validate` 再检查 entailment、query/category alignment 和 citation directness。

待确认：Graph/社区候选映射到 canonical chunk 时，当前依赖 `source_chunk_keys` 和 active registry；尚无独立、可复用的 VerifiedEvidence 持久层。
