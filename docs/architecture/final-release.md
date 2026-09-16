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

The CrewAI Flow is executed by an independent Worker; the API only creates a
durable Run/Job and returns `202`. The Deep Agent/LangGraph path remains a
fallback, not a replacement for the Research Engine or the Project Runtime.
The Domain stores remain authoritative:

| State | Authority |
| --- | --- |
| Session/run lifecycle | Session Runtime |
| Working checkpoint | LangGraph SQLite Saver |
| Manuscript/Facts/Contributions | Scholar Project Store and files |
| Evidence/Citation | Existing governed services and Project projections |
| Proposed changes | Immutable DraftPatch |
| File mutation | Human-approved PatchApprovalService |
| Async queue | Persistent Job Repository |
| Product trace | RunEventStore (persist before live publish) |

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

Production requires configured LLM settings, a persistent
`LEO_AGENTIC_SCHOLAR_CHECKPOINT_PATH`, and the external Worker process.
Test/local-fast modes may explicitly use `InMemorySaver`. The async Web API and
Worker use the same `ScholarRuntimeFactory` and `ScholarRunManager`.

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

The repository regression suite, static checks, lockfile check and frontend build
are green. The final closure also verified the real Provider path through
Support Claim, Introduction, Conclusion and Abstract; Introduction completed
Reviewer PASS, Human Approval and Safe Apply, while a deliberately unavailable
Provider returned `FAILED` without a fabricated result. The current release
result is `CREWAI_PRODUCTION_READY`.
