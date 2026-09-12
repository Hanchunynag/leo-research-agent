# Citation Lifecycle（Phase 2E）

## Authority and identity

`references.bib` is the current LaTeX Project bibliography source of truth.
`ScholarProjectStore` 的 Citation Registry 只保存审计投影：文献身份、观察到的
BibKey、metadata hash、Evidence 关系和状态。`CitationIdentity` 与项目内
BibKey 是两个不同的概念；同一 DOI 可以在不同 Project 使用不同 BibKey。

身份解析保持保守顺序：exact DOI、versionless arXiv ID、已有 canonical ID，
最后才是完全一致的 title + first author + year。歧义或不同 DOI 不会因为标题相似
而自动合并。

## Resolution and proposal

Writing Runtime 把 Verified Evidence 交给确定性的 `CitationResolutionService`，
服务按需读取并 hash `references.bib`，通过 `BibliographySynchronizer` 复用用户已有
BibKey。没有现有 entry 时，只有 title/authors/year 等 Verified Metadata 足够，才
由 deterministic `BibKeyGenerator` 生成 `BibEntryCandidate` 和 `BibliographyChange`。
未知的 volume、pages、publisher 等字段保持缺失；元数据冲突 fail closed。

```text
Verified Evidence
  → CitationIdentity
  → current references.bib sync/hash
  → existing BibKey
       or BibEntryCandidate + CitationRequirement
  → CitationBinding
```

Search Result、Web snippet 和未经验证的 Evidence 不能生成 Citation。External
Evidence 仍保持 request-scoped；一旦被 Binding 引用，Project 只保存最小可复核的
canonical locator、provider、publication/retrieval time、content hash 和 validation
projection，不会导入 Workspace Corpus 或创建 Qdrant point。

## Draft and approval

Introduction 的 `DraftPatch` 扩展保存 `bibliography_base_hash`、CitationBinding、
CitationRequirement 和 `BibliographyChange`。Preview 因而同时展示 manuscript diff
与新增 Bib entry。Conclusion/Abstract 仍然 Citation OFF。

Resolver 和 Skill 都不能写 `.bib`。Human-facing client ACCEPT 后，现有
`PatchApprovalService` 才会按以下顺序处理：

```text
re-read/hash .bib
  → reuse a newly-added user BibKey when identity matches
  → validate bindings and additions
  → atomic ADD to .bib
  → base-hash/original-content guarded .tex apply
  → re-read both files and reconcile Registry
```

用户在审批前修改 `.bib` 时不会被覆盖。若同一 identity 已由用户用自定义 key 加入，
正文中的 proposal key 会在 Apply 时重写为用户 key；若是无关修改、key collision、
identity conflict 或 stale hash，则 Patch 不应用。若 Bib 已写入但 `.tex` 随后无法
应用，结果明确报告 `PARTIAL_APPLY`，不伪装成事务回滚。
