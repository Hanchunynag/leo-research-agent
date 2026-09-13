# ScholarHarness Progress

## Current phase

Phase 4: End-to-End Validation, Release Hardening & Project Finalization
(release hardening complete; external-provider E2E gate pending).

## Completed

- Created branch `phase-1a-state-ownership-review` from the clean `main` baseline.
- Audited the existing Research Engine, legacy Session Store, Harness, RAG contracts and LangGraph integration.
- Added the first Session Runtime foundation: catalog, Session-per-DB, Conversation messages, Run metadata, result projection, Session lock and orphan-run marker.
- Added an Application Facade contract through `ResearchApplicationFacade.research_topic(...)` without changing the existing Research Engine.
- Added Scholar domain contracts, read-on-request LaTeX hash synchronization, base-hash Patch Guard, Manuscript Facts and user-confirmed Contribution storage.
- Migrated CLI Agentic answer and Web `/api/answers` through `ResearchApplicationFacade`.
- New Web/CLI requests no longer inject `AgenticSessionStore` into the existing Agent Service; legacy session APIs remain readable.
- Added `LegacySessionAdapter` for read-only migrate-on-open Conversation compatibility; legacy Evidence remains readable from the compatibility Store until a stable Evidence projection is defined.
- Added explicit `run_id`, `trace_id`, `thread_id`, `job_id`, and `project_id` correlation propagation.
- Added and verified the official LangGraph SQLite Checkpointer Factory; its production lifecycle is injected explicitly at Graph/Harness construction.
- Full regression after boundary and correlation changes: `298 passed, 1 skipped, 6 warnings`.

## In progress

- Explicit legacy Session management commands still read `AgenticSessionStore`; historical data migration is not performed.
- SQLite Checkpointer lifecycle is injected by the caller; deployment wiring must keep the lifecycle-scoped saver open for the Graph/Harness lifetime.

## Completed in Phase 1C

- Added framework-agnostic `ResearchRequest` and bounded `ResearchBudget`.
- Added `ResearchCapabilityService.search_papers()`, `search_sections()` and `read_evidence()` as adapters over the existing UnifiedKnowledgeService and CanonicalCorpusService.
- Reused `CandidateEvidence`, `VerifiedEvidence` and `EvidenceIntelligencePipeline`; no second RAG or evidence engine was added.
- Added deterministic candidate locator/source validation and structured local-only `EvidencePack` claim/evidence projection.
- Added capability contract tests covering paper/section filters, locator ownership, fabricated source rejection, claim mapping and deterministic coverage.
- Added `docs/architecture/research-capability-contract.md` and updated the ScholarHarness boundary document.

## Completed in Phase 2A

- Added `WritingRequest`, `ResearchNeed`, `ClaimPlan`, `SectionDraft`, `WritingResult` and runtime `CapabilitySet` contracts.
- Added `skills/write-introduction/SKILL.md` and deterministic rhetorical ResearchNeed decomposition.
- Added `ScholarWritingService` / `ManuscriptSupervisor` and thin `ResearchDelegate` over the existing Research Capability Layer.
- Added evidence-first Introduction drafting with bounded revision loop, citation requirements and no fabricated BibKeys.
- Added deterministic `IntroductionReviewer` and optional chat-completion semantic judge adapter.
- Extended DraftPatch and ReviewReport with project, citation, contribution, original-content and report correlation without changing Patch Guard write semantics.
- Added the real project fixture integration test: latest manuscript hash → Research delegation → EvidencePack → ClaimPlan → Draft → Review → unapplied DraftPatch.

## Completed in Phase 2B

- Added Project-owned immutable DraftPatch persistence and Patch lifecycle/audit tables in the existing `.scholar/project.db`.
- Added `PatchApprovalService` with human-only Accept/Reject, Review Gate, base-hash/original-content revalidation, atomic Apply, conflict preservation and idempotent replay.
- Persisted post-Apply manuscript hash/version projection and old/new hash audit; Build state is separate from Apply state.
- Added `LatexBridgeService`, normalized `LatexDiagnostic`/`BuildResult`, FastAPI Patch/Build endpoints and shared-service CLI fallback (`scholar patch ...`).
- Added a thin `vscode-extension/` using native `vscode.diff` and the external `latex-workshop.build` command; it has no Research or arbitrary file-write authority.
- Added Phase 2B lifecycle, conflict, API and Build contract tests.

## Completed in Phase 2C

- Added shared `SkillRegistry`, explicit task routing, `CapabilityProfile`, `SkillExecutionContext` and tagged `SkillResult` without introducing Agent middleware.
- Added `support-claim` as a local-only Research Skill returning verified-provenance `ClaimSupportResult`, with unresolved/counter/qualifying evidence semantics and no DraftPatch path.
- Added shared Conclusion/Abstract Synthesis Runtime, `AbstractFactSet`, deterministic Fact/Result/Contribution review rules, and shared Project-owned DraftPatch persistence.
- Added `write-conclusion`, `write-abstract` and `support-claim` SOP documents; added deterministic Method/Experiment/Results/Conclusion dependency invalidation.
- Added cross-Skill capability and synthesis integration tests while preserving the Introduction vertical.

## Current test status

- Phase 2B focused tests: 10 passed including API, Apply, conflict, idempotency, Build and VS Code bridge contracts.
- Phase 2C focused tests: 5 passed covering Registry, routing, Support Claim, synthesis, capability separation and stale dependencies.
- Previous Phase 2A baseline: 309 passed, 1 skipped, 6 warnings.
- Phase 2D focused tests: 11 passed; Phase 2E focused Citation tests: 7 passed;
  full regression after Citation Lifecycle: 344 passed, 1 skipped, 6 warnings;
  Ruff and `git diff --check` passed.

## Completed in Phase 2E

- Added Project-scoped `CitationIdentity`, `CitationBinding`, `BibEntryCandidate`,
  `BibliographyChange`, `CitationResolutionResult` and stable Citation statuses.
- Added `BibliographySynchronizer` using the declared `bibtexparser` parser; `references.bib`
  is read-on-demand, hash-bound and user-managed, with deterministic DOI/arXiv/metadata identity
  matching and BibKey collision handling.
- Extended Introduction `WritingContext`/`DraftPatch` with binding IDs, bibliography hash,
  bibliography changes and citation requirements. Support Claim can return Citation Bindings
  without creating a Patch; Conclusion/Abstract remain citation-disabled.
- Added Project Citation Registry and request-scoped External Evidence audit projections. No
  Web source is imported into the Canonical Corpus or Qdrant.
- Extended Human-approved Apply to revalidate `.bib`, reuse a user-added custom BibKey, apply
  bibliography additions before `.tex`, report `BIB_*` conflicts and `PARTIAL_APPLY`, then
  re-sync both files.
- Added Citation Lifecycle and wrong-identity reviewer checks, plus deterministic lifecycle and
  user-managed bibliography integration tests.

## Next step

- Phase 2E implementation and final validation are complete; Citation Registry remains an
  index/audit projection, and bibliography changes remain Human-approved proposals.

## Completed in Phase 3A

- Created branch `phase-3a-langchain1-deepagents` and captured the resolver snapshot for the direct in-process migration.
- Upgraded the production dependency line to LangChain `1.4.0`, LangChain Core `1.6.3`, LangGraph `1.2.11`, Checkpoint `4.2.0`, SQLite Checkpoint `3.1.1` and Deep Agents `0.7.13`; no 0.3 production compatibility branch remains.
- Added the narrow LangChain 1.x Provider/Message/Tool/Async/Stream adapter while keeping existing Provider and Domain contracts framework-agnostic.
- Migrated and regression-tested the existing Research Graph on LangGraph 1.x without changing Research Engine semantics.
- Added persistent Deep Agents checkpoint injection and `ScholarHarnessService.resume(...)`; a new service/runtime can resume the same `thread_id` after SQLite close/reopen.
- Added the in-process `ScholarHarnessService` with explicit TaskRouter use, progressive Skill metadata/content loading, isolated Research and Reviewer Subagents, high-level Domain Result output and scoped read-only Skill filesystem.
- Introduction Research results use a request-scoped EvidencePack handoff, preventing duplicate ResearchCapability calls; formal ClaimPlan/Evidence/Review/Patch ownership remains in existing Writing Runtime.
- Deep Agents cannot write manuscript/bibliography/Facts/Contributions, invoke unrestricted filesystem/Shell/HTTP/Qdrant, or approve/apply DraftPatch. Conclusion and Abstract retain no Research tool visibility.
- Added LangChain adapter, Harness isolation, Skill routing and process-restart resume tests; existing Citation, Freshness, Writing, Approval, VS Code and Research regression tests remain green.

## Phase 2D implementation

- Added deterministic `FreshnessPolicy` and extended `ResearchRequest` with freshness mode,
  date window, explicit-latest and domain-sensitivity fields.
- Added `WebLiteratureAdapter` and request-scoped discovery/metadata cache over the existing
  ToolGateway literature contract; no Web Agent or second search loop was introduced.
- Added normalized Literature Search contracts, DOI/arXiv/conservative bibliographic identity,
  metadata conflict provenance and local/Web deduplication.
- Extended Candidate/Verified Evidence and Citation projection for real external locators;
  abstract/full-text candidates are validated by the existing Evidence Intelligence Pipeline
  and External Source Resolver, while snippets are rejected.
- Added structured provider/freshness failure classification and Harness provider/tool details;
  provider failure never becomes an unsupported claim, and FRESH_REQUIRED fails closed.
- Introduction and Support Claim Profiles can request Web only through Research Capability;
  Conclusion and Abstract remain Research-disabled. No manuscript, Facts, Contributions,
  Workspace Scope, DraftPatch or Approval lifecycle is mutated by Web Research.
- Added deterministic Web policy tests for local skip, freshness-required fallback, default
  local-only behavior, snippets, provider timeout, DOI reuse and metadata conflict preservation.

## Known compatibility issues

- The legacy `AgenticSessionStore` remains a read-compatible source for old session commands, not a writer for new Web/CLI requests.
- Bibliography edits are intentionally ordered, not cross-file transactional. If `.bib` is
  applied and `.tex` fails, the Patch is recorded as `PARTIAL_APPLY`; automatic rollback is
  not attempted.

## Completed in Phase 3B

- Added `ScholarRuntimeFactory`/`ScholarRuntimeBundle` as the single complete Scholar Runtime
  composition root for production, test and local-fast modes.
- Production now requires a configured official SQLite Checkpointer and fails fast instead of
  silently using `InMemorySaver`; bundle shutdown owns checkpoint/provider resource cleanup.
- Web, CLI and test entrypoints use the same Harness contract for Scholar request/resume/status;
  the existing human Patch commands remain outside Agent permissions.
- Added startup orphan detection, persistent process-restart resume coverage, correlation checks
  for Session/Project/Thread/Run, and stable runtime failure codes/termination metadata.
- Added deterministic `ScholarHarnessEvaluationSuite` metrics for routing, research decisions,
  forbidden tools, result validity, context isolation, resume and capability violations.
- Added bounded Supervisor/Research/Reviewer context accounting and redacted Harness trace
  correlation/usage metadata without copying Domain state into Deep Agents memory.
- Added production runtime architecture documentation and preserved the existing Research,
  Evidence, Citation, Writing, DraftPatch and Human Approval contracts.

## Phase 4 finalization

- Committed the confirmed Phase 1A–3B implementation as the stable release baseline
  (`16c80b5`) and kept local `.scholar/` Project Runtime databases out of Git.
- Added the versioned `examples/scholar-demo/` LaTeX Project and
  `scripts/run_scholar_final_e2e.py`; both use the existing Production Factory and
  ScholarHarnessService rather than a second Demo pipeline.
- Verified real Production composition with the configured LLM settings, real SQLite
  `SqliteSaver`, existing corpus/index configuration, and managed resource shutdown.
- Ran the fixed four-skill Production E2E against the configured DeepSeek endpoint.
  Runtime startup, SQLite composition, routing, capability visibility and failure
  classification were exercised; the provider returned HTTP 402
  `Insufficient Balance` before Domain Results could be produced. This is recorded
  as an external-provider release blocker, not converted into a passing fixture.
- Added final release architecture and README guidance for Production startup,
  Scholar request/resume/status, DraftPatch Human Approval, and evaluation.
- Made Patch and Writing Runtime review identifiers deterministic per request so a
  resumed Domain execution cannot create a duplicate immutable proposal.

## Phase 4R Console finalization

- Added the read-only Scholar Web Console at `/scholar`, reusing the existing
  ScholarHarnessService, Session/Project stores, Harness diagnostics and
  Evaluation suite.
- Added Run snapshot and deterministic Run Event projection plus SSE replay with an
  `after` cursor; no second workflow state or unrestricted file/data access was added.
- Added Evidence/Citation, Manuscript State, DraftPatch, Runtime and Evaluation
  panels, with a clearly labelled fixture-only `/scholar?demo=1` mode.
- Added Console contract tests, React production build coverage and browser checks;
  the Console remains usable while the real external Provider is unavailable.

## V1 release status

The implementation scope is frozen for V1 validation. V1 is not yet released:
the real external LLM balance must be restored and the four end-to-end Domain
flows, human-approved Introduction Apply/build and restart/failure validation
must be rerun successfully. New Skills, Agents, Tools, Corpus import, cloud
deployment, advanced citation recommendation, and complex Context Compression
remain Future Work. Known limitations remain: the LaTeX compiler is external to
the Bridge, cross-file Apply is ordered and can report `PARTIAL_APPLY`, and real
external Web availability depends on provider/network configuration.
