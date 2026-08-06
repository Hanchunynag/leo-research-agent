# 迁移计划

## 阶段一完成物

本阶段只新增领域契约、Protocol、Legacy Adapter、文档和基线测试。没有切换 [`main.py`](../../main.py)、[`app/web/runtime.py`](../../app/web/runtime.py) 或 [`app/agentic/service.py`](../../app/agentic/service.py) 的现有调用，也没有修改任何前端/数据库 Schema。

## 阶段二精确文件级迁移列表

以下顺序按可独立回滚的提交拆分；每一项先加测试再切调用。

### 2.1 Canonical Corpus 只读门面

- 新增 `app/corpus/__init__.py`
- 新增 `app/corpus/repository.py`：按 document/chunk/block 回查 `paper.json`、structure、chunks，计算 corpus version。
- 新增 `tests/test_corpus_repository.py`
- 复用但不改格式：`app/parsing/pipeline.py`、`app/chunking/structure.py`、`app/chunking/chunker.py`

### 2.2 Workspace scope（先文件存储，不改现有 DB Schema）

- 新增 `app/workspaces/__init__.py`
- 新增 `app/workspaces/repository.py`：阶段二采用独立 JSON/SQLite 文件需另行审批；不能修改现有 session/index registry schema。
- 新增 `app/workspaces/service.py`
- 新增 `tests/test_workspaces.py`
- 暂不改 `app/web/api.py`，直到兼容策略明确。

### 2.3 Unified Knowledge Service

- 新增 `app/knowledge_service/__init__.py`
- 新增 `app/knowledge_service/service.py`：实现 `KnowledgeEngine`
- 新增 `app/knowledge_service/scope.py`：解析 workspace_id/scope_version
- 新增 `app/knowledge_service/legacy_backend.py`：组合 `LegacyKnowledgeEngineAdapter`
- 新增 `app/knowledge_service/graphrag_backend.py`：包装 `GraphRAGRetrievalRuntime`
- 新增 `tests/test_unified_knowledge_service.py`
- 保留 `app/runtime/retrieval.py`、`app/runtime/graphrag.py` 原文件，不删除。

### 2.4 Evidence 三阶段

- 新增 `app/evidence/__init__.py`
- 新增 `app/evidence/verification.py`：Canonical locator/content hash 回查
- 新增 `app/evidence/selection.py`：迁入覆盖/MMR 策略
- 新增 `app/evidence/service.py`：实现 `EvidenceIntelligenceService`
- 新增 `tests/test_evidence_verification.py`
- 新增 `tests/test_evidence_selection_contract.py`
- 复用并逐步委托：`app/agentic/selection.py`、`app/generation/validation.py`

### 2.5 ContextBuilder

- 新增 `app/context/builder.py`：实现新 `ContextBuilder`
- 修改 `app/context/assembly.py`：保留旧函数，内部可委托 builder；旧返回结构不变
- 新增 `tests/test_context_builder_contract.py`
- 保留 `app/context/models.py` 兼容对象。

### 2.6 Agent 与后端解耦

- 修改 `app/agentic/service.py`：构造参数改为 `KnowledgeEngine`、`EvidenceIntelligenceService`、`ContextBuilder`、`ToolGateway`；移除所有 `is_graphrag` 判断。
- 修改 `app/agentic/harness.py`：只增加契约级 trace/预算字段，不接触后端。
- 新增 `app/tools/gateway.py`
- 新增 `tests/test_tool_gateway.py`
- 更新 `tests/test_agentic_rag.py`、`tests/test_agentic_harness.py`
- 旧 `RetrievalRuntime`/`GraphRAGRetrievalRuntime` 仍由 Adapter 使用。

### 2.7 统一 IndexGeneration

- 新增 `app/index_lifecycle/__init__.py`
- 新增 `app/index_lifecycle/service.py`
- 新增 `app/index_lifecycle/legacy_adapter.py`
- 新增 `app/index_lifecycle/registry_adapter.py`：把 `index_epochs` 投影为 `IndexGeneration`
- 修改 `app/index_registry/coordinator.py`：只在 Adapter 后调用，不改 schema
- 新增 `tests/test_index_lifecycle.py`
- 保留 `app/indexing/*`、`app/graph/*` 全部旧实现。

### 2.8 CLI 切换（独立提交）

- 修改 `main.py` 的 search/dense/hybrid/rerank/context/answer 构造路径，统一通过 service factory。
- 新增 `app/bootstrap.py` 作为 composition root。
- 更新 CLI 快照测试：`tests/test_knowledge_retrieval.py`、`tests/test_dense_retrieval.py`、`tests/test_reranking.py`、`tests/test_generation.py`、`tests/test_agentic_rag.py`。
- 必须保持参数、退出码和 JSON 字段兼容。

### 2.9 Web 切换（与 Agent/索引提交分离）

- 修改 `app/web/runtime.py`，只调用 Unified Knowledge Service 和 index lifecycle；不直接调用 `build_knowledge_base`/`build_dense_index`。
- `app/web/api.py` 仅在明确 workspace 兼容策略后修改；所有旧 endpoint/response 保持兼容。
- 更新 `tests/test_web_api.py`。
- 本项不得与 Agent 或索引实现修改放在同一提交，不修改 `web/`。

### 2.10 持久 Job 与删除语义（需独立决策）

- 待确认存储和恢复语义后，再新增 `app/jobs/repository.py`、`app/jobs/service.py` 和 `tests/test_job_recovery.py`。
- 论文删除先实现 Workspace 解绑和 generation 失效；是否物理删除 raw/parsed/canonical 必须另行审批。
- 在此之前，保持 Web Job 重启丢失和删除不支持的现有外部行为。

## 明确不进入阶段二默认范围

本节是阶段一生成时的初始边界；用户随后明确授权阶段二接入 LightRAG，因此“LightRAG 接入”已由 [`app/knowledge_engine/lightrag_engine.py`](../../app/knowledge_engine/lightrag_engine.py) 完成。删除旧 GraphRAG、前端重写、现有数据库表 ALTER、全仓重命名仍未授权且未执行。

阶段二实际文件与初始建议略有收口：统一服务、LightRAG、generation 和 index lifecycle 统一位于 `app/knowledge_engine/`，Evidence 位于 `app/evidence/`，避免同时创建 `knowledge_service/` 和 `index_lifecycle/` 两套顶层命名。实现结果见 [`stage2_implementation.md`](stage2_implementation.md)，阶段三列表见 [`stage3_file_tasks.md`](stage3_file_tasks.md)。
