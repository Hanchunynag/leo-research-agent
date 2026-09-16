# 当前实现与后续计划

## 已落地

- MinerU -> Canonical 是唯一 PDF 事实链；
- Paper metadata 与 Content chunks 已分为 Level-1/Level-2；
- Paper BM25、Content BM25、Paper Dense、Content Dense 和 hierarchical retrieval 已有独立入口；
- Content Dense manifest 已记录 model/revision/text policy/chunk policy/tokenizer 与 per-paper digest；Paper Dense 在 Paper metadata corpus 变化时全量重算；
- 新增、更新、删除对 Level-2 Content Dense 以 `paper_id` 执行局部 Qdrant upsert/delete；Level-1 Paper Dense 对变化后的完整 Paper metadata 集合重新 Embedding；
- CrewAI Flow/Crew 已作为上层 Scholar 编排入口，Agent 不接触索引实现；
- Evidence verification、citation validation 和 human approval 保持在领域服务边界内。

## 运行规则

正常 PDF 导入不使用 `--force`。发现 Canonical 变化时，只重新构建对应 Paper 的 Level-2 Content；如果导入导致 Paper metadata corpus 变化，则 Level-1 Paper Dense 对全部 Paper metadata 重新 Embedding。全局 BM25 可以更新 IDF，但不能因此重新调用 Level-2 Dense provider。

`--force` 是管理员 rebuild 入口，用于 policy/model/revision/text-policy 变化、旧 manifest 恢复或明确的全库维护窗口。

## 后续工作

1. 将 document-level IndexReport 接入更细粒度的后台 Job 进度和可恢复 replay；
2. 为远程 Qdrant 增加同等 payload filter、批量 upsert 和删除审计；
3. 用真实论文集持续测量 Paper-stage recall、Content-stage recall、RRF 和 Cross-Encoder latency；
4. 在不扩大 Agent 权限的前提下完成 CrewAI production parity gate。
