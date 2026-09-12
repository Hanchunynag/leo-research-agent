# Phase 3B Production Scholar Runtime

## Composition root

`ScholarRuntimeFactory` is the only complete Scholar Runtime composition
boundary. It creates the runtime-scoped `SessionManager`, Project Store,
Research Capability, Web/metadata adapters, Citation and Approval services,
Writing/Reviewer adapters, Skill Runtime, Deep Agents Harness and checkpointer.
The factory only assembles dependencies; task routing, research policy and
writing decisions remain in their existing domain services.

Web and CLI Scholar requests use the same `ScholarHarnessService` contract:

```text
ScholarRuntimeFactory
  → ScholarHarnessService.scholar_request / resume / status
  → existing Skill Runtime and Domain Result
```

Patch show/accept/reject remains a human-facing Approval command. It is not a
Harness tool and is not exposed to the Deep Agent.

## Runtime modes and resources

`AgenticRAGConfig.runtime_mode` is one of `production`, `test` or `local-fast`.
Production requires `LEO_AGENTIC_SCHOLAR_CHECKPOINT_PATH` (or an explicit
`scholar_checkpoint_path`) and opens the official LangGraph `SqliteSaver`.
Missing or unusable persistent storage raises `CHECKPOINT_UNAVAILABLE`; it
never silently falls back to `InMemorySaver`. Memory checkpoints are allowed
only when the mode explicitly says `test`/`local-fast`, or in a unit-test
dependency override.

The returned `ScholarRuntimeBundle` owns the checkpointer, provider clients and
other runtime resources. `close()`/the context manager closes them. Startup
also marks stale Session `RUNNING` records as `INTERRUPTED`; a later resume is
allowed only when the corresponding persistent checkpoint still exists.

The stores retain separate authority:

```text
Session DB       → session/run lifecycle and result projection
LangGraph saver  → resumable Deep Agent working state
Project DB       → manuscript, facts, contributions, evidence, citation, patch
```

The saver is not a copy of Project state and is not the source of truth for
Facts, Contributions, Evidence, Citation Bindings or DraftPatch.

## Correlation and resume

`session_id`, `run_id`, `thread_id`, `trace_id`, `job_id` and `project_id` are
independent correlation fields. A new request cannot reuse a thread already
bound to a Session. Resume requires a matching Session/Project/Thread Run in
`INTERRUPTED`, `RUNNING` or `WAITING_USER` state and a checkpoint in the
configured saver; otherwise it returns `RESUME_UNAVAILABLE` or
`RUN_CORRELATION_INVALID`.

Resume reuses the existing Run ID and Session lifecycle. Domain side effects
are still owned by their Project services, and a resumed graph does not create
a second Patch, Contribution or Citation Binding merely because the Harness
object was rebuilt.

## Harness evaluation and budgets

`ScholarHarnessEvaluationSuite` consumes existing Harness metadata and trace;
it does not create a second Agent or evaluation runtime. Fixed cases cover all
four current Skills and report Task Routing Accuracy, Forbidden Tool Call
Count, Unexpected Research Rate, Required Research Miss Rate, Domain Result
Validity, Context Isolation Violation Count, Resume Success Rate and
Capability Violation Count.

The Harness records bounded, redacted context estimates separately for the
Supervisor, Research Subagent and Reviewer Subagent. The code-level
`ScholarContextBudget` is authoritative; Skill Markdown cannot enlarge it.
Trace records correlation, selected Skill, capability/tool visibility,
subagent calls, provider usage, context usage, result type and termination
reason without persisting full manuscript text, credentials or large tool
payloads.

## Failure and termination

Runtime failures keep stable boundaries: configuration/checkpoint failure,
provider failure, capability violation, resume unavailability and domain
conflict are not converted into a generic successful result. Research/Web
failure remains distinct from unsupported evidence. Runs expose a termination
reason such as `COMPLETED`, `NEEDS_USER_REVIEW`, `INSUFFICIENT_EVIDENCE`,
`DOMAIN_CONFLICT`, `PROVIDER_FAILED`, `BUDGET_EXHAUSTED` or `INTERRUPTED`.

The Deep Agent remains the only user-level intelligent router. It can select a
registered Skill and delegate to the existing Research/Reviewer adapters, but
cannot access unrestricted filesystem, shell, HTTP, Qdrant, databases, direct
manuscript/bibliography writes or Patch Approval. Conclusion and Abstract do
not receive a Research Subagent. Human Approval remains outside the Agent and
is still the only path that applies a DraftPatch.
