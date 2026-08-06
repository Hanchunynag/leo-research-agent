# 阶段三废弃与保留模块

本阶段没有物理删除旧 GraphRAG、Dense RAG 或旧 Agent；下列模块已从生产 Web/CLI composition root 移除，但保留用于行为基线、Adapter、影子评估和独立回滚。

| 模块 | 状态 | 替代入口 | 处置 |
|---|---|---|---|
| [`app/agentic/harness.py`](../../app/agentic/harness.py) | legacy/deprecated | [`app/research/harness.py`](../../app/research/harness.py) | 保留测试，禁止新增后端相关顶层状态 |
| [`app/agentic/service.py`](../../app/agentic/service.py) | legacy/deprecated | [`app/research/runtime.py`](../../app/research/runtime.py) | 保留阶段一回归和回滚，不再由 Web/CLI 默认构造 |
| [`app/runtime/retrieval.py`](../../app/runtime/retrieval.py) | migration adapter backend | Unified Knowledge Service | 只能在 composition 层注入 Unified Service |
| [`app/runtime/graphrag.py`](../../app/runtime/graphrag.py) | migration/shadow backend | Unified Knowledge Service | 不删除，不允许 Agent 直接导入 |
| `app/graphrag/` | migration/shadow implementation | Knowledge Service 内部 | 不删除，关系基线仍依赖 |
| `app/indexing/` Dense/BM25 | lexical/dense signal | Knowledge Service 内部 | 保留为内部召回与回归，不作为第二 Agent RAG |

[`app/contracts/adapters.py`](../../app/contracts/adapters.py) 的阶段一 Adapter 继续保留。删除这些模块必须等价替代测试、阶段二 shadow 验收、active generation 切换与回滚均通过后另行提交；本阶段不做破坏性清理。
