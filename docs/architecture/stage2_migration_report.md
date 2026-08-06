# 阶段二数据迁移报告

## Canonical 与 Workspace

- Canonical Document：7；Canonical Chunk：135。计数与 [`stage1_behavior_baseline.json`](../../tests/baselines/stage1_behavior_baseline.json) 一致。
- default workspace：`workspace_id=default`、`scope_version=1`、成员文档 7。
- 重复 PDF 通过 [`CanonicalCorpusService.duplicate_for_hash`](../../app/corpus/service.py) 检查，相同 SHA-256 不重复解析。
- 没有数据库 Schema migration；Workspace、generation 和 operation 使用独立 JSON 持久化。

## LightRAG active shadow generation

数据来源是 `data/evaluation/stage2_lightrag_migration.json`：

| 字段 | 结果 |
|---|---:|
| generation_id | `IG_419c4e228f0dd1a8` |
| state | `active`（仅 shadow） |
| profile | `lightrag-1.5.6-bge-m3-deepseek-chat-json-v1` |
| document / canonical mapping | 7 / 135 |
| entity / relation | 409 / 532 |
| 构建耗时 | 925.049 s |
| 增量标志 | true |

`active` 由 [`IndexGenerationRepository.activate`](../../app/knowledge_engine/generations.py) 写入 `data/knowledge/index_generations.json`。两个早期单 Chunk smoke `IG_0dac2e047d6f19e3` 和 `IG_0d2a9d43828e6f9c` 因不满足 relation threshold 已保留记录并转为 `failed`；历史网络、传输和中断失败也全部保留。当前 registry 共 8 个 generation：1 active、7 failed、0 validating。

## 入库 Token 与运行成本

完整入库实测：153 次 LLM 调用，354,832 prompt tokens、84,737 completion tokens、439,569 total tokens；134 次 embedding 调用，1,238 个 embedded texts；LLM failure/retry 均为 0。仓库没有配置供应商单价，因此只报告可验证 Token 与耗时，不虚构货币金额。

构建使用 `deepseek-chat` 的独立 IndexProfile；[`load_local_llm_settings`](../../app/generation/settings.py) 的正式回答模型配置没有被改写。迁移脚本的 validation 要求 canonical mapping 完整且 relation_count >= 1，未通过时 [`KnowledgeIndexService.build_generation`](../../app/knowledge_engine/index_service.py) 保留旧 active。

## 增量与删除语义

[`LightRAGKnowledgeEngine.update_documents`](../../app/knowledge_engine/lightrag_engine.py) 和 `delete_documents` 使用指定 `document_id`，返回 `full_rebuild=False`；[`test_incremental_update_and_delete_never_request_full_rebuild`](../../tests/test_stage2_lightrag_engine.py) 冻结该行为。真实 active corpus 没有执行破坏性删除演练，因此“删除残留的生产实测值”仍待阶段三在 forked generation 中完成；当前外部删除 API 仍不存在。

Web 上传已经只通过 [`KnowledgeIndexService.synchronize_after_parse`](../../app/knowledge_engine/index_service.py) 编排，但为保持 legacy 正式回答的既有行为，该兼容方法仍调用旧 `build_knowledge_base`/`build_dense_index`，尚未自动创建 LightRAG incremental generation。因此“新增论文不触发全库重建”已经在 LightRAG API 与测试层成立，但尚未对仍在 serving 的 legacy Web 索引路径成立；这是正式切换前的剩余阻塞项。
