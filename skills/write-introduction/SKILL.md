---
name: write-introduction
description: Draft a literature-grounded Introduction using verified evidence and the shared approval workflow.
metadata:
  task_type: WRITE_INTRODUCTION
  capability_source: SkillRegistry
---

# write-introduction

## When to Use

Use only when the user explicitly asks to write or revise the LaTeX Introduction. The current
target section is `introduction`; the vertical supports `WRITE` and `REVISE` modes.

## Required Context

The runtime supplies the latest Introduction content and hash, the user instruction, relevant
confirmed Manuscript Facts, user-confirmed Contributions, a rhetorical ClaimPlan, and verified
local EvidencePacks. Unrelated conversation history, the whole project database, and the full
knowledge corpus are not part of the skill context.

## Research Requirements

Break the request into bounded needs for technical problem, existing approaches, and limitations
when relevant. Delegate each need through `ResearchRequest` and the local
`ResearchCapabilityService`. External literature claims require verified evidence and a stable
citation key or an explicit citation requirement.

An Introduction must cite at least 5 distinct papers. Distinctness is counted by stable paper
identity (`paper_id`, or the canonical external identity when there is no local paper id), not by
Chunk count, Evidence count, or duplicate BibKeys. The research budget must first recall a broad
Paper-Level shortlist and then obtain verified content evidence within that shortlist.

## Workflow

```text
read latest Introduction
  -> identify relevant rhetorical moves
  -> build ResearchNeed values
  -> delegate local ResearchRequest values
  -> build ClaimPlan
  -> evidence-first draft
  -> deterministic and optional semantic review
  -> produce DraftPatch only
```

## Hard Constraints

Do not create or confirm Contributions, modify Manuscript Facts, invent evidence locators or
citation keys, use Web Research, access retrieval internals, or write any manuscript file.
Preserve unrelated Introduction text in REVISE mode. Contributions must come only from the
user-confirmed registry, and external academic claims must remain traceable to verified evidence.
The runtime must not create or register a DraftPatch when fewer than 5 distinct paper identities
are citation-bound in the final draft.

## Completion Criteria

The result contains a current base hash, structured ClaimPlan, EvidencePack provenance, SectionDraft,
ReviewReport, and an un-applied DraftPatch. BLOCKER/HIGH issues produce `NEEDS_USER_REVIEW`; the
runtime never calls `workspace.apply_patch()`.
