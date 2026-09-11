# ScholarHarness Progress

## Current phase

Phase 1B: Application boundary convergence and Checkpointer compatibility.

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
- Added and verified the official `langgraph-checkpoint-sqlite==2.0.0` Factory without switching the production Graph yet.
- Full regression after boundary and correlation changes: `298 passed, 1 skipped, 6 warnings`.

## In progress

- Explicit legacy Session management commands still read `AgenticSessionStore`; historical data migration is not performed.
- LangGraph production Graph has not yet been switched to the SQLite Factory; the official Factory and isolated restart Spike pass.
- No Deep Agents dependency, Supervisor, Research Agent, Reviewer Agent or Writing Skill has been added.

## Next step

- Add the Graph Runtime checkpointer lifecycle switch behind the new Factory.
- Add the legacy read/migrate-on-open Adapter and keep one-way ownership.
- Proceed to Research Tool Adapter and EvidencePack only after the boundary regression remains green.

## Known compatibility issues

- `langgraph-checkpoint-sqlite==2.0.0` is compatible with the current `langgraph==0.3.34` / `langgraph-checkpoint==2.1.2` dependency graph; production integration still needs a Graph Runtime lifecycle change.
- The legacy `AgenticSessionStore` remains a read-compatible source for old session commands, not a writer for new Web/CLI requests.
