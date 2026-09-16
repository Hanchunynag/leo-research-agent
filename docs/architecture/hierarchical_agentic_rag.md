# Hierarchical RAG 与 CrewAI 编排

## 检索主链

```text
User Query
  -> Query planning / translation
  -> Paper BM25 + Paper BGE-M3
  -> Paper RRF
  -> Top-K paper_id
  -> Content BM25 + Content BGE-M3（paper_id filter）
  -> Content RRF
  -> BGE Cross-Encoder
  -> Evidence verification
  -> Claim/Citation validation
```

Paper metadata 只生成一个 Level-1 vector；Content chunks 属于单篇 Paper，统一物理 collection 不改变其独立生命周期。

## Agent 边界

CrewAI 的 Supervisor、Research、Writer、Reviewer 是认知角色，不是存储或检索实现：

- Supervisor：选择 bounded route，不能读取 RAG backend；
- Research：通过 capability adapter 获取 verified EvidencePack；
- Writer：只使用 request-scoped manuscript state 和 verified evidence，提出 DraftPatch；
- Reviewer：审查 Patch、Evidence 和约束，不能直接修改稿件；
- Human：接受或拒绝 Patch，Safe Apply 由 domain service 执行。

## 运行安全

Flow state 只保存 run、session、project、route、review round 和 contract reference；不启用跨任务长期 Memory。所有 evidence/citation 仍由既有 stores 和 verification service 持久化。
