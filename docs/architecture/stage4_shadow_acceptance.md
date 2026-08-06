# Stage 4 Shadow Acceptance

## 实测范围

2026-08-06 使用 `scripts/evaluate_stage2_shadow.py` 对
`data/evaluation/retrieval_questions.jsonl` 的 21 条基础题和
`data/evaluation/relation_questions.jsonl` 的 8 条关系题执行了真实 Legacy / LightRAG 双检索。机器门槛集中在
`tests/baselines/lightrag_cutover_acceptance.json`，判定入口为
`app.evaluation.acceptance.evaluate_cutover`。精简实测快照保存在
`tests/baselines/stage4_real_shadow_run.json`；完整运行产物
`data/evaluation/stage4_shadow_acceptance.json` 按 `.gitignore` 留在本机。

## 结果

| 指标 | Legacy | LightRAG / 结果 | 门槛 |
|---|---:|---:|---:|
| DirectQA nDCG@10 | 0.621967 | 0.491043 | LightRAG ≥ Legacy × 0.95 |
| 相对 nDCG | 100% | 78.95% | ≥ 95% |
| Source Backfill | — | 100% | ≥ 95% |
| Workspace 泄漏 | — | 0 | 0 |
| Excluded Evidence 泄漏 | — | 0 | 0 |
| 关系真值确认数 | — | 0 / 8 | 8 / 8 |

本次 29 个查询实测耗时 51.106 秒。LightRAG 查询桥
`app.knowledge_engine.model_bridge.build_lightrag_client_config` 记录了 28 次 LLM 调用、18,395 输入 Token、1,214 输出 Token、29 次 Embedding 调用和 85 条 Embedding 文本。

## 判定

`official_cutover_approved=false`。失败分类为：

- dataset：8 条关系题仍为 `pending_human_confirmation`；
- reranking：LightRAG DirectQA nDCG 未达到 Legacy 的 95%；
- relation：关系 Precision/Coverage 无人工真值，不能计算；
- indexing/deletion/rollback：真实评估脚本尚未写入线上操作结果，虽然独立回归测试已通过。

因此 `app.knowledge_engine.serving.EngineCutoverService.switch_to_lightrag` 不得执行。当前正式回答继续使用 Legacy，LightRAG 只以 Shadow 运行。

## 下一轮定位

优先检查 `app.knowledge_engine.lightrag_engine.LightRAGKnowledgeEngine._map_results` 的源内排名、`app.evidence.service.EvidenceIntelligencePipeline` 的跨来源重排和 Chunk qrel 映射。不得通过整体降低门槛绕过当前失败。
