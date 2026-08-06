# Stage 4 Engine Cutover

## 配置与 Pin

`app.knowledge_engine.serving.KnowledgeServingConfig` 定义以下持久化字段：

```yaml
knowledge:
  official_engine: legacy | lightrag
  official_generation_id: IG_xxx | null
  shadow_engine: legacy | lightrag | none
  shadow_generation_id: IG_xxx | null
```

配置由 `KnowledgeServingConfigRepository` 原子写入
`data/knowledge/serving_config.json`，变更追加到
`data/knowledge/serving_audit.jsonl`。`knowledge_serving_status` 是 Web 与 CLI 的共同只读投影。

`EngineCutoverService.switch_to_lightrag` 同时要求：

- Acceptance 的 `passed` 和 `official_cutover_approved` 均为 `true`；
- Generation 存在且状态为 `active` 或 `retired`；
- Generation 使用明确 `index_profile_id`；
- 切换写入包含 `acceptance_passed=true` 的审计记录。

`app.knowledge_engine.unified_service.build_configured_unified_service` 只按持久化 Pin 组装服务。LightRAG Official 缺 Pin、Pin 不存在、状态不可服务或审计不匹配时直接失败，不选择“最新 Generation”。一次查询由 `UnifiedKnowledgeService` 捕获一个固定 Engine/Generation 实例，查询期间配置文件变化不会改变该请求。

## 当前状态

- Official Engine：`legacy`
- Official Generation：`null`
- Shadow Engine：`lightrag`
- Shadow Generation：`IG_419c4e228f0dd1a8`
- Shadow Profile：`lightrag-1.5.6-bge-m3-deepseek-chat-json-v1`

当前 LightRAG 未达到正式门槛，未执行 Cutover。

## 接口兼容

`app.web.runtime.LocalRAGWebRuntime.public_status` 增加 Official/Shadow/Profile/Generation 展示字段；`main.py knowledge status` 在原结果中增加 `knowledge_serving`。回答 API 的既有字段未删除或改名。相关测试位于
`tests/test_stage4_engine_cutover.py`。
