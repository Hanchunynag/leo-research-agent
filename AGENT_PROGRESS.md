# ScholarHarness Progress

## Current phase

Phase 1: Session/Application foundation and Scholar domain contracts.

## Completed

- Created branch `phase-1a-state-ownership-review` from the clean `main` baseline.
- Audited the existing Research Engine, legacy Session Store, Harness, RAG contracts and LangGraph integration.
- Added the first Session Runtime foundation: catalog, Session-per-DB, Conversation messages, Run metadata, result projection, Session lock and orphan-run marker.
- Added an Application Facade contract through `ResearchApplicationFacade.research_topic(...)` without changing the existing Research Engine.
- Added Scholar domain contracts, read-on-request LaTeX hash synchronization, base-hash Patch Guard, Manuscript Facts and user-confirmed Contribution storage.

## In progress

- Existing Web/CLI construction still uses the legacy `AgenticSessionStore` path; migration and wiring must be completed through an Adapter.
- Persistent LangGraph Checkpointer compatibility has not been solved. Production still uses `InMemorySaver`.
- No Deep Agents dependency, Supervisor, Research Agent, Reviewer Agent or Writing Skill has been added.

## Next step

- Add focused tests for Session isolation, Facade normalization, manuscript hash refresh, patch conflicts and contribution/fact authority.
- Run the existing regression suite.
- Perform the official LangGraph SQLite Checkpointer compatibility spike before implementing restart resume.

## Known compatibility issues

- Current `langgraph==0.3.x` / `langgraph-checkpoint==2.x` environment has no verified compatible official SQLite Checkpointer package.
- Legacy `HarnessAgentService` still persists through `AgenticSessionStore`; do not enable the new Facade in production until the Adapter migration path is tested.
