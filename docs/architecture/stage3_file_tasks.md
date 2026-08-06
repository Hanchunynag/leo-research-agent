# 阶段三精确文件级任务

以下任务按可独立回滚的提交拆分；不得在同一提交同时重写 Agent、索引和前端。

1. **完整影子验收**：扩展 [`scripts/evaluate_stage2_shadow.py`](../../scripts/evaluate_stage2_shadow.py) 读取 `data/evaluation/retrieval_questions.jsonl`，生成 21 题 Recall@K/nDCG/延迟/Token；扩展现有 `tests/baselines/stage2_shadow_acceptance.json`，并新增 `tests/test_stage3_shadow_acceptance.py`。
2. **关系题标注**：新增 `data/evaluation/relation_questions.jsonl`（需人工确认 qrel），扩展 [`app/evaluation/shadow.py`](../../app/evaluation/shadow.py) 计算 relation-path precision/coverage；新增 `tests/test_stage3_relation_acceptance.py`。
3. **正式引擎切换策略**：修改 [`app/knowledge_engine/unified_service.py`](../../app/knowledge_engine/unified_service.py)，加入显式 `official_engine=legacy|lightrag` 和 generation pin；只有验收报告通过才允许 LightRAG serving。新增 `tests/test_stage3_engine_cutover.py`。
4. **回滚 smoke**：扩展 [`app/knowledge_engine/generations.py`](../../app/knowledge_engine/generations.py) 的诊断，不改状态机；在 `tests/test_stage3_engine_cutover.py` 验证 active-answer generation 切换与 retired rollback 后 Top-K 可恢复。
5. **Agent 收口**：修改 [`app/agentic/service.py`](../../app/agentic/service.py)，移除面向旧 fake runtime 的最终兼容分支，只依赖 Unified Service capability；更新 [`tests/test_agentic_rag.py`](../../tests/test_agentic_rag.py) 和 [`tests/test_agentic_harness.py`](../../tests/test_agentic_harness.py)。不重写 [`app/agentic/harness.py`](../../app/agentic/harness.py)。
6. **Web composition**：修改 [`app/web/runtime.py`](../../app/web/runtime.py)，从配置注入 active LightRAG engine/profile/generation，但不改 [`app/web/api.py`](../../app/web/api.py) 的外部响应和 `web/` 前端；更新 [`tests/test_web_api.py`](../../tests/test_web_api.py)。
7. **CLI composition**：修改 [`main.py`](../../main.py)，增加内部 active-engine/profile/generation 选择与状态诊断，保持现有参数和 JSON 字段兼容；更新检索与生成 CLI 测试。
8. **失败 operation 恢复**：扩展 [`app/knowledge_engine/index_service.py`](../../app/knowledge_engine/index_service.py)，实现 FAILED operation 的幂等 resume，以及 add/update/delete 的生产命令；新增 `tests/test_stage3_index_recovery.py`。
9. **删除残留验收**：在 forked generation 上调用 [`LightRAGKnowledgeEngine.delete_documents`](../../app/knowledge_engine/lightrag_engine.py)，验证 mapping、vector、entity/relation 和查询均无目标 Document 残留；不得物理删除 canonical。新增 `tests/test_stage3_delete_residual.py`。
10. **删除 API 决策后实现**：若得到外部行为变更授权，只修改 [`app/web/api.py`](../../app/web/api.py) 增加路由，并且唯一调用 `KnowledgeIndexService.delete_documents`；新增 Web API 测试。未授权前保持“不支持”。
11. **后台任务持久化**：修改 [`app/web/jobs.py`](../../app/web/jobs.py) 或新增 `app/jobs/repository.py`，采用独立存储且不 ALTER 现有 Schema；先确认 restart 时 queued/running 的恢复语义，再新增 `tests/test_job_recovery.py`。
12. **候选排序校准**：修改 [`app/evidence/service.py`](../../app/evidence/service.py) 和 [`app/knowledge_engine/lightrag_engine.py`](../../app/knowledge_engine/lightrag_engine.py)，用 21 题数据校准 source quota/reranker，目标是不牺牲关系回填与隔离；每次变更更新 shadow 快照，不修改阶段一 baseline。
