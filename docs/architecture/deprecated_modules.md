# 已退役模块

以下历史索引和运行实现已从当前生产组合根移除，不保留兼容入口、配置别名或 shadow path：

| 领域 | 当前入口 |
|---|---|
| Paper/Content indexing | `app/indexing/paper.py`, `app/indexing/paper_dense.py`, `app/indexing/dense.py` |
| Hierarchical retrieval | `app/retrieval/paper.py`, `app/retrieval/hierarchical.py` |
| Evidence governance | `app/evidence/service.py` |
| Agent orchestration | `app/orchestration/` |

运行时只能通过 Unified Knowledge Service、KnowledgeIndexService 和 CrewAI capability adapter 访问这些能力。
