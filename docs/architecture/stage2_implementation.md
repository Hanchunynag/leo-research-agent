# 阶段二实现说明

记录日期：2026-08-06。本阶段未修改 `web/` 前端、未改写 [`app/agentic/harness.py`](../../app/agentic/harness.py)，也未删除 [`app/runtime/graphrag.py`](../../app/runtime/graphrag.py) 或任何旧 GraphRAG/Dense 实现。

## 唯一知识服务边界

[`UnifiedKnowledgeService`](../../app/knowledge_engine/unified_service.py) 是 Agent 生产组装可见的知识边界。Web 在 [`LocalRAGWebRuntime._build_service`](../../app/web/runtime.py) 中、CLI 在 [`main.py`](../../main.py) 中先用 `build_legacy_unified_service` 包装旧 runtime，再传给 [`AgenticRAGService`](../../app/agentic/service.py)。Agent 已不再判断 `is_graphrag`，只读取 `supports_advanced_retrieval` capability。

迁移期开关仍是：

```text
official answer = legacy runtime
shadow retrieval = LightRAGKnowledgeEngine（可选）
```

LightRAG generation 的 `active` 只表示 LightRAG 内部可查询，不表示正式回答已切换。影子比较见 [`stage2_shadow_comparison.md`](stage2_shadow_comparison.md)。

## Canonical Corpus 与 Workspace

- [`DocumentRepository`](../../app/corpus/repository.py)、`SectionRepository`、`ChunkRepository`、`CitationRepository` 读取现有 `paper.json`、structure 和 chunks，不修改原 Schema。
- [`CanonicalCorpusService`](../../app/corpus/service.py) 提供 Document 列表、SHA-256 去重检查和 `document_id/chunk_id/page/block/content` 回查。
- [`WorkspaceRepository`](../../app/workspaces/repository.py) 用独立 `data/knowledge/workspaces.json` 保存 scope 快照；没有 ALTER 现有数据库。
- [`WorkspaceService`](../../app/workspaces/service.py) 创建 `default` workspace，将现有 7 篇文献迁入 scope 1，并在过滤时拒绝非成员、excluded document、excluded direction 和错误 scope_version。
- [`LocalRAGWebRuntime.parse_pdf`](../../app/web/runtime.py) 在解析前用 PDF content hash 检查 canonical 重复；解析后只调用 [`KnowledgeIndexService.synchronize_after_parse`](../../app/knowledge_engine/index_service.py) 编排下游索引。

## LightRAG 引擎

[`LightRAGKnowledgeEngine`](../../app/knowledge_engine/lightrag_engine.py) 实现 `KnowledgeEngine` 的 `index_documents`、`update_documents`、`delete_documents`、`retrieve_candidates` 和 `get_status`：

- `ainsert_custom_chunks` 复用 canonical Chunk 边界并使用稳定 `document_id`；
- `canonical_chunk_map.json` 保存 LightRAG internal chunk ID 到 canonical locator 的映射；
- `adelete_by_doc_id` 只删除指定文档；更新先删除指定文档再插入其当前 Chunk；
- [`KnowledgeIndexService.build_incremental_generation`](../../app/knowledge_engine/index_service.py) 通过 `fork_generation` 复制已验证 generation，仅重算受影响文档，不触发全库重建；
- Chunk/Entity/Relation 各自保留候选池，使用源内排名分数，避免不可比分数在 Evidence Intelligence 前互相挤占。

模型桥接位于 [`build_lightrag_client_config`](../../app/knowledge_engine/model_bridge.py)，统一记录 LLM/Embedding 调用与 Token。索引 profile 固定为 `lightrag-1.5.6-bge-m3-deepseek-chat-json-v1`；正式回答模型配置未被覆盖。

## Evidence Intelligence

[`EvidenceIntelligencePipeline.verify`](../../app/evidence/service.py) 严格执行 workspace/scope 过滤、来源校验、去重、文献多样性、重排、图原文回填、直接性、证据等级、覆盖与冲突检测；`select` 最后按 token、Top-K、文献与来源配额选择。所有 graph content 都回填为 [`CanonicalLocator`](../../app/corpus/service.py) 的原文。

找不到直接或间接关系支持时，证据被标记 `graph_inference`，metadata 带“不能表述为论文已证明”的 disclaimer。[`SelectedEvidenceContextBuilder`](../../app/evidence/context.py) 拒绝任何非 `selected` Evidence。

## 索引操作状态

[`KnowledgeIndexService`](../../app/knowledge_engine/index_service.py) 持久化 Web 操作：

```text
REGISTERED -> PARSING -> CANONICAL_READY -> INDEXING
           -> KNOWLEDGE_READY -> COMPLETED
                         \-> FAILED（可恢复记录）
```

[`IndexGenerationRepository`](../../app/knowledge_engine/generations.py) 管理 `building/validating/active/retired/failed`，新 generation 验证通过前不会替换 active，retired 可回滚。当前状态见 [`stage2_migration_report.md`](stage2_migration_report.md)。

## 测试落点

阶段二定向测试为 [`tests/test_stage2_corpus_workspace.py`](../../tests/test_stage2_corpus_workspace.py)、[`tests/test_stage2_lightrag_engine.py`](../../tests/test_stage2_lightrag_engine.py)、[`tests/test_stage2_evidence_intelligence.py`](../../tests/test_stage2_evidence_intelligence.py)、[`tests/test_stage2_unified_service.py`](../../tests/test_stage2_unified_service.py)、[`tests/test_stage2_index_service.py`](../../tests/test_stage2_index_service.py) 和 [`tests/test_stage2_model_bridge.py`](../../tests/test_stage2_model_bridge.py)。真实迁移与查询脚本分别是 [`scripts/migrate_stage2_lightrag.py`](../../scripts/migrate_stage2_lightrag.py) 和 [`scripts/evaluate_stage2_shadow.py`](../../scripts/evaluate_stage2_shadow.py)。
