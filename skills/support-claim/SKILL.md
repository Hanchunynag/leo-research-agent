---
name: support-claim
description: Assess whether a user academic claim is supported by verified local or permitted external evidence.
metadata:
  task_type: SUPPORT_CLAIM
  capability_source: SkillRegistry
---

# support-claim

## When to Use

Use when the user asks whether a specific academic Claim is supported by local literature. This is
a research-only task; it does not write or revise any manuscript section.

## Required Context

Use only the user Claim, its conservative normalization, bounded sub-claims, and the verified local
EvidencePack returned by ResearchCapabilityService.

## Research Requirements

Search local hierarchical knowledge through the Research Capability boundary. Every supporting,
counter, or qualifying source must retain its verified paper/section/chunk provenance. No search
preview is sufficient evidence.

## Workflow

```text
normalize claim
  -> split only explicit independent clauses
  -> bounded local ResearchRequest
  -> EvidencePack validation
  -> ClaimSupportResult
```

## Hard Constraints

Do not write LaTeX, create DraftPatch, create or confirm Contributions, modify Manuscript Facts, use
Web Research, or treat missing support as proof that the Claim is false. `INSUFFICIENT_EVIDENCE` is
distinct from `CONTRADICTED`.

## Completion Criteria

The result preserves claim-to-evidence relations, distinguishes support/counter/qualifying evidence,
lists unresolved sub-claims and citation requirements, and contains no manuscript mutation.
