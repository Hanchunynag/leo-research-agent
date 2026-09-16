# 领域模型

## 文献身份

| 类型 | 标识 | 关系 |
|---|---|---|
| Work | `work_id` | 可聚合多个具体 PDF 版本 |
| Paper | `paper_id` | 索引与 metadata 生命周期单位 |
| Document | `document_id` | 一个具体 PDF |
| Section | `section_id` | Document 的结构节点 |
| Chunk | `chunk_id` | Paper-owned 检索单元 |
| Block | `block_id` | Canonical 页级原子内容 |

`paper_id` 划定增量边界；`document_id` 用于具体文件的来源和证据回查；`work_id` 只用于实体归并和展示，不跨 Paper 合并 Chunk 或 vector。

## 两级 Chunk

```text
Paper
  ├── Level-1: <paper_id>_metadata（1 个 metadata/abstract chunk）
  └── Level-2: C001, C002, ...（只属于当前 paper）
```

`chunk_id` 在 Paper 内稳定；物理 vector ID 使用 `paper_id:chunk_id` 派生，因此不同论文可以安全地各自拥有 `C001`。

## Evidence 状态

```text
CandidateEvidence -> VerifiedEvidence -> SelectedEvidence -> ContextPack
```

Candidate 可不完整；Verified 必须可回查 Canonical locator、页面、blocks 和 content hash；Selected 才能进入生成上下文。

## Workspace 与运行

`ResearchWorkspace(workspace_id, scope_version)` 隔离可见文献；`EvidenceRequest` 固定 query、scope 和 top-k；`AgentRun` 保存 route、usage、trace、answer 和终止原因；`IndexReport` 保存 paper digest、embedding revision、变更数量和失败信息。

## 不变量

1. 任何 vector 必须绑定 `paper_id/chunk_id/level/section_path/content_hash/embedding_revision`。
2. Level-1 每篇 Paper 恰好一个 vector。
3. Level-2 vector 的构建、更新、删除只能以所属 `paper_id` 为范围。
4. VerifiedEvidence 必须有 document、chunk、page 和 block locator。
5. Context token count 不得超过 token budget。
