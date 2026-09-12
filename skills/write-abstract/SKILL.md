---
name: write-abstract
description: Synthesize a concise Abstract from the latest full manuscript state and verified manuscript facts.
metadata:
  task_type: WRITE_ABSTRACT
  capability_source: SkillRegistry
---

# write-abstract

## When to Use

Use only when the user explicitly asks to write or revise the LaTeX Abstract.

## Required Context

Read the latest problem, Method, Experiment/Results, Conclusion, confirmed Contributions and
Manuscript Facts. Extract an `AbstractFactSet` before drafting. Missing Results is an insufficient
manuscript state, not an invitation to use prior knowledge.

## Research Requirements

Local and Web Research are disabled. Abstracts contain no generated citation and no external
literature claim.

## Workflow

```text
read latest full-manuscript snapshot
  -> build AbstractFactSet
  -> synthesize problem/method/setting/result/meaning
  -> review numeric, fact, method and contribution consistency
  -> produce an unapplied DraftPatch
```

## Hard Constraints

Do not create citations, Contributions, Facts, Methods, experimental settings or Results. Every
number must come from Manuscript Facts or current Results/Table text. Do not write the manuscript.

## Completion Criteria

The abstract is concise, contains no unsupported new information, passes the shared Abstract review
policy, and is returned as a Project-owned DraftPatch requiring Human Approval.
