# 服务契约

## Knowledge and indexing

```python
index_paper(paper_id, canonical_path) -> IndexReport
update_paper(paper_id, canonical_path) -> IndexReport
delete_paper(paper_id) -> DeleteReport
retrieve(request: EvidenceRequest) -> Sequence[CandidateEvidence]
```

索引服务接受论文级 identity，内部负责 Canonical、structure、Chunk、BM25、Paper Dense 和 Content Dense 的顺序更新。Paper metadata corpus 变化时 Paper Dense 全量重算；Content Dense 始终按 `paper_id` 增量更新。Agent 不获得 Qdrant、BM25 文件或 model client。

## Retrieval contract

```text
Paper stage:  BM25(metadata) + Dense(metadata) -> RRF -> paper_id set
Content stage: BM25(content, paper_id set) + Dense(content, paper_id set)
              -> RRF -> Cross-Encoder -> CandidateEvidence
```

Content retrieval 必须携带 `paper_id` scope；没有 Paper candidate 时不得退回全库细粒度 Dense。

## Evidence contract

```text
CandidateEvidence -> VerifiedEvidence -> SelectedEvidence -> ContextPack
```

VerifiedEvidence 需要 `paper_id`、`document_id`、`chunk_id`、页范围、block IDs、content hash 和 verification method。无法回查 Canonical 的候选进入 rejected，不可直接进入生成上下文。

## Orchestration contract

CrewAI Flow 只传递 Pydantic structured contracts：研究请求、EvidencePack、DraftPatch、ReviewReport 和 ApprovalRequest。Supervisor 决定 bounded route，Research 读取证据能力，Writer 只能提出 Patch，Reviewer 只能返回审查结果；批准和 Safe Apply 仍由人工边界与领域服务负责。

## Failure rules

- provider、revision、text policy 或 manifest 不兼容时拒绝查询/增量写入；
- 缺少按论文 digest 的旧 Dense manifest 时要求显式管理员 rebuild；
- 任何引用验证失败都 fail closed；
- 索引操作失败只保留失败记录，不删除 Canonical；
- BM25 全局 IDF 更新不得触发 Dense provider 调用。
