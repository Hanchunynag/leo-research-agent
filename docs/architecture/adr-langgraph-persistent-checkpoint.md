# ADR: LangGraph Persistent Checkpointer Runtime

## Status

Accepted and implemented in Phase 3A.

## Decision

Use the official LangGraph 1.x SQLite checkpointer with the current dependency line:

```text
langchain==1.4.0
langchain-core==1.6.3
langgraph==1.2.11
langgraph-checkpoint==4.2.0
langgraph-checkpoint-sqlite==3.1.1
```

`app/langchain_agent/checkpoint_factory.py` exposes a lifecycle-scoped Factory. `memory` yields the existing `InMemorySaver`; `sqlite` lazily opens the official `SqliteSaver`, calls `setup()`, and closes the connection when the context exits. No pickle, JSON Graph State dump, or custom checkpoint serializer is used.

## Evidence

The resolver selected `langgraph-checkpoint-sqlite==3.1.1` with the LangGraph 1.x dependency set. Tests verify Graph compilation, `thread_id` checkpoint creation, database close/reopen, loading the same thread, interrupt-before-node behavior, resume from persisted state, multiple checkpoint records, and Deep Agent service/runtime reconstruction.

## Consequence

`app/langchain_agent/checkpoint_factory.py` owns only checkpointer construction and lifecycle. `ScholarHarnessService` accepts an injected saver and exposes `resume(thread_id, resume_value, ...)`; a caller keeps the lifecycle-scoped saver open and may rebuild the service after process restart. The saver contains only LangGraph/Deep Agents working state, while Session/Project stores remain the business source of truth and Patch Approval is not represented as an agent interrupt.
