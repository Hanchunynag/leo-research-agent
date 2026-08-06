# 领域模型

代码定义位于 [`app/contracts/domain.py`](../../app/contracts/domain.py)，均为 frozen/slots dataclass，不触发现有数据库 Schema 变化。

## Corpus 身份

| 类型 | 身份 | 关键关系 | 当前实现映射 |
|---|---|---|---|
| `Work` | `work_id` | 聚合一个或多个 Document | `build_identity`、`WorkCatalogRecord` |
| `Document` | `document_id` | 一个具体 PDF；属于 Work；拥有 Section/Chunk/Block | `paper.json.identity`、`PaperCatalogRecord` |
| Section | `section_id` | Document 的结构节点 | 当前仍使用 `build_structure` 生成的 dict；阶段一未复制定义 |
| Chunk | `chunk_id`/稳定 `chunk_key` | 检索最小文本单元，引用回 Block | 当前仍使用 `build_chunks` 生成的 dict |
| Block | `block_id` | 页级 canonical 原子内容 | `paper.json.blocks` |

`Document.source_sha256` 和 `canonical_path` 明确把索引投影锚定到 canonical 文档。`work_id` 允许为空，以真实表达尚未通过身份核验的 canonical PDF；进入 VerifiedEvidence 前则必须有稳定 Work/Document locator。Work 与 Document 分开，避免不同 PDF 版本错误共享 `document_id`。

## Workspace

- `ResearchWorkspace(workspace_id, scope_version, name)`：课题隔离根。
- `WorkspaceDocument`：Document 成员关系，记录加入和可选移除 scope version。
- `ResearchDirection`：Workspace 内的目标与约束。
- `EvidenceRequest`：每次检索固定 request/workspace/scope/query/top_k，可选 work/document scope。

规则：`scope_version` 从 1 开始且由调用者显式传入；任何缺失或 0 值在契约构造阶段失败。阶段二由 [`WorkspaceService.ensure_default`](../../app/workspaces/service.py) 为现有数据创建兼容 `default` workspace，但过滤仍校验请求 workspace/scope，不能把 default 当作绕过授权的全局通道。

## Evidence 状态机

```text
backend result dict
    -> CandidateEvidence
    -> VerifiedEvidenceBundle
       -> VerifiedEvidence
       -> rejected candidate + issue
    -> SelectedEvidence
    -> ContextPack
```

`CandidateEvidence` 保存 score/retrieval_source 和可能不完整的 locator。`VerifiedEvidence` 强制要求 work/document/chunk、有效页范围、至少一个 block_id、content_hash 和 verification_method。`SelectedEvidence` 增加选择顺序、理由和 token 数。`ContextPack` 冻结最终证据、渲染文本和预算。

现有 `app.graph.models.EvidenceCandidate` 是 GraphRAG 内部候选，不等同于新 `CandidateEvidence`；现有 `app.context.models.EvidenceItem` 接近 SelectedEvidence 的渲染形态，但没有独立 Verified 阶段。

## Runtime

- `AgentRun`：后端无关的 run/state/answer/usage/latency 投影；usage 允许 `None`，因为旧实现没有统一记录。
- `IndexGeneration`：corpus/scope 与后端投影版本的统一代次；新生命周期为 pending/building/validating/active/retired/failed，`superseded` 仅保留阶段一兼容。
- `IndexProfile`：索引和查询共同引用的 engine、mode、Top-K、token limit、embedding/LLM model 配置快照。

Evidence 状态由 `retrieved -> filtered -> ranked -> backfilled -> verified -> selected` 单向推进；失败候选进入 `rejected`。直接性为 direct/indirect/inferred，等级为 primary/candidate/analogy/graph_inference。

## 不变量

1. Workspace 请求必须带 `workspace_id + scope_version`。
2. VerifiedEvidence 必须能定位到 Document、Chunk、页范围和 Block。
3. Context token_count 不得超过 token_budget。
4. Index/Agent 计数不得为负。
5. Adapter 必须显式记录 verification method；字段齐全不等于语义蕴含成立。
