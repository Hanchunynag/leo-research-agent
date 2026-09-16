# Research orchestration boundary

当前 Scholar 业务的推荐上层编排是 CrewAI Flow/Crew。Research、Writer、Reviewer 通过结构化 contract 交换结果，RAG 检索统一进入 hierarchical retrieval，Evidence 统一进入 verification/selection。

该边界不向 Agent 暴露 embedding provider、BM25 index、Qdrant collection 或 Canonical 文件路径。任何需要读取文献的行为都必须经过 Research capability；任何需要修改稿件的行为都必须先产生 DraftPatch，再等待人工 Approval。
