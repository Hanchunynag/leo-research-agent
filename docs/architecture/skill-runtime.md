# Phase 2C Skill Runtime

## Skill、Agent 与 Tool

Skill 是任务 SOP 和完成策略；`SkillRegistry` 只登记名称、任务类型、所需 Context、
Capability Profile 和 `SKILL.md` 路径。Skill 不直接执行 Research、不写数据库，也不拥有
Manuscript 写权限。未来 Agent 可以选择并调用 Skill Runtime，但不能改变这些边界。

Tool/Research Capability 是可组合的能力边界；`support-claim` 可以调用 local Research，
Conclusion/Abstract Profile 则没有 `research.local`、`evidence.read` 或 `citation.resolve`。

## Task Routing

`TaskRouter` 支持显式 `WRITE_INTRODUCTION`、`SUPPORT_CLAIM`、`WRITE_CONCLUSION` 和
`WRITE_ABSTRACT`。自然语言只做低风险关键词选择；多义或低置信度输入返回
`needs_user_choice`，不会创建 Contribution、Research Direction 或执行 Tool。

## Capability Profiles

Runtime 使用代码中的 `CapabilityProfile` 进行约束。通用别名包括：

```text
READ_MANUSCRIPT / READ_FACTS / READ_CONTRIBUTIONS
LOCAL_RESEARCH / READ_EVIDENCE / RESOLVE_CITATION
REVIEW / CREATE_DRAFT_PATCH
WEB_RESEARCH / WRITE_MANUSCRIPT / WRITE_FACTS / WRITE_CONTRIBUTIONS
```

Introduction 拥有 Literature Writing 所需的 Research/Evidence/Citation 能力；Support Claim
拥有 local Research 和 Evidence Read 但没有 Patch；Conclusion/Abstract 只读取最新 Manuscript、
Facts、confirmed Contributions，使用 Review 和 DraftPatch，不接触 RAG。

## Result Types

`SkillResult` 是通用外层结果。Writing Skills 的 payload 是现有 `WritingResult`，其中仍然
包含共享 `SectionDraft`/Project-owned `DraftPatch`；Research Skill 的 payload 是
`ClaimSupportResult`，不会伪造 SectionDraft 或 DraftPatch。

`ClaimSupportResult` 保留 original/normalized Claim、sub-claim、support/counter/qualifying
Evidence、coverage、unresolved 和 citation requirements。没有支持证据是
`INSUFFICIENT_EVIDENCE`，不是 `CONTRADICTED`。

## Shared Synthesis Runtime

Conclusion 与 Abstract 共享 `SynthesisWritingService`、`SharedManuscriptReviewer`、
`SectionDraft`、`DraftPatch` 和 Human Approval。两者都关闭 Local/Web Research 和 Citation。
Conclusion 需要当前 Method/Results；Abstract 还需要 Introduction/Conclusion，并先构造
`AbstractFactSet`。Results 缺失、上游 deterministic section stale、数值冲突或新贡献会阻止
生成 FINAL-ready Patch。

Method/Experiment/Results/Conclusion 的 deterministic dependency graph 由
`ManuscriptSynchronizer` 维护：上游变化只会标记下游 stale，不使用 LLM 推断依赖。
