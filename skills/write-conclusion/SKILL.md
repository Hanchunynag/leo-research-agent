---
name: write-conclusion
description: Synthesize a Conclusion from the latest manuscript, confirmed contributions, and supported results.
metadata:
  task_type: WRITE_CONCLUSION
  capability_source: SkillRegistry
---

# write-conclusion

## When to Use

Use only when the user explicitly asks to write or revise the LaTeX Conclusion.

## Required Context

Read the latest problem/context, Method, Experiment/Results, confirmed Contributions, Manuscript
Facts, and existing Conclusion. The current Conclusion hash is the concurrency token.

## Research Requirements

Local and Web Research are disabled. Conclusion claims must come from the user's manuscript and
confirmed Project Registry, not from external literature.

## Workflow

```text
read latest manuscript snapshot
  -> verify deterministic upstream sections are not stale
  -> synthesize problem/method/findings/contributions
  -> review fact, result and contribution consistency
  -> produce an unapplied DraftPatch
```

## Hard Constraints

Do not call Research Capability, add citations, invent numerical results, create Contributions,
modify Facts, write files, or turn an unverified Contribution into an experimentally proven claim.
Future Work must come from explicit user/manuscript text and remains a candidate when not confirmed.

## Completion Criteria

Every strong conclusion is traceable to current manuscript content or a confirmed Contribution;
BLOCKER/HIGH issues remain visible in ReviewReport; output is a shared `DraftPatch` awaiting Human
Approval.
