# Scholar Runtime Progress

## Current state

The Scholar production path is unified on `ScholarOrchestrationService` with a
CrewAI Manager and deterministic Flow. Research, Writer, Reviewer, Session
Runtime, Project Store, Citation, Patch Approval and Worker lifecycle remain
owned by their existing domain services.

The bounded Manager loop persists `RunGoal`, `RunState`, recovery actions and
domain-result references in Session Runtime `run_checkpoints`. Human approval
continues to be the only path that applies a DraftPatch, with review and
base-hash guards enforced by the domain services.

Paper parsing and retrieval remain reusable corpus capabilities, but every
production research, writing, review, and approval request now enters the
CrewAI Scholar Run path. There is no alternate agent runtime.

## Verification

Run the offline regression suite with isolated test configuration:

```bash
LEO_MYSQL_ENABLED=false \
LEO_LLM_API_KEY=file-test-key \
LEO_LLM_TIMEOUT_SECONDS=45 \
LEO_LLM_MAX_TOKENS=700 \
.venv/bin/pytest -q
```

External-provider and production-corpus E2E must still be executed explicitly
in the target deployment environment; offline tests do not claim that coverage.
