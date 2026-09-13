# ScholarHarness V1 Final Release

## Final boundary

```text
User
  → ScholarHarnessService
  → Scholar Deep Agent
  → Skill / CapabilityProfile
  → Research or Reviewer Subagent
  → Scholar Domain Runtime
  → Evidence / Citation / DraftPatch
  → Human Approval
  → LaTeX Project
```

The Deep Agent is an in-process Harness, not a replacement for the Research
Engine or the Project Runtime. The Domain stores remain authoritative:

| State | Authority |
| --- | --- |
| Session/run lifecycle | Session Runtime |
| Working checkpoint | LangGraph SQLite Saver |
| Manuscript/Facts/Contributions | Scholar Project Store and files |
| Evidence/Citation | Existing governed services and Project projections |
| Proposed changes | Immutable DraftPatch |
| File mutation | Human-approved PatchApprovalService |

## Release validation

The final evaluation uses a fixed LaTeX fixture and real Production Composition.
It checks routing, capability visibility, Research invocation, context
isolation, Domain Result type, Patch approval, checkpoint resume and
termination reasons. It does not turn prose quality into an unsupported numeric
score.

`SUPPORT_CLAIM` is Research-only and never creates a Patch. `WRITE_INTRODUCTION`
may use governed Local/Web Evidence and Citation Lifecycle. `WRITE_CONCLUSION`
and `WRITE_ABSTRACT` are manuscript synthesis Skills with Research and Citation
disabled. The release fixture is executed against the configured Production LLM;
provider errors remain failed evaluation records and are never replaced by a fake
result.

## Reproducibility

Production requires configured LLM settings and a persistent
`LEO_AGENTIC_SCHOLAR_CHECKPOINT_PATH`. Test/local-fast modes may explicitly use
`InMemorySaver`. Web/CLI requests use the same `ScholarRuntimeFactory` and
`ScholarHarnessService`.

## Known limitations and future work

- A LaTeX compiler is not bundled; the Bridge stores a build request and accepts
  diagnostics from VS Code LaTeX Workshop.
- Context overflow currently fails closed. The Harness does use a deterministic,
  bounded Reviewer evidence projection; broader context compression remains future
  work.
- Web PDFs are not automatically imported into the Canonical Corpus.
- Cross-file `.tex`/`.bib` Apply is ordered rather than transactional; partial
  application is explicitly reported.
- Multi-user cloud deployment, additional Skills, advanced citation
  recommendation and complex bibliography conflict UX remain future work.

## Current validation status

The repository regression suite is green. A real Production E2E startup was also
verified with `SqliteSaver`, local retrieval configuration and managed shutdown,
but the configured DeepSeek endpoint currently returns HTTP 402
`Insufficient Balance`. Consequently the four Domain flows and Human Approval
release gate remain pending an available external LLM account.

**Release Status:** `NOT RELEASED — EXTERNAL_PROVIDER_UNAVAILABLE`
