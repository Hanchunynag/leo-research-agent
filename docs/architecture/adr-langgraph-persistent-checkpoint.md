# ADR: LangGraph Persistent Checkpointer Compatibility

## Status

Accepted compatibility candidate; production Graph switch deferred to the next runtime-lifecycle change.

## Decision

Use the official `langgraph-checkpoint-sqlite==2.0.0` package with the current dependency line:

```text
langchain==0.3.30
langchain-core==0.3.86
langgraph==0.3.34
langgraph-checkpoint==2.1.2
langgraph-checkpoint-sqlite==2.0.0
```

`app/langchain_agent/checkpoint_factory.py` exposes a lifecycle-scoped Factory. `memory` yields the existing `InMemorySaver`; `sqlite` lazily opens the official `SqliteSaver`, calls `setup()`, and closes the connection when the context exits. No pickle, JSON Graph State dump, or custom checkpoint serializer is used.

## Evidence

The isolated resolver selected `langgraph-checkpoint-sqlite==2.0.0` with `aiosqlite==0.20.0` and the existing checkpoint dependency. The Spike verified import, Graph compilation, `thread_id` checkpoint creation, database close/reopen, loading the same thread, interrupt-before-node behavior, resume from the persisted checkpoint, and multiple checkpoint records.

## Consequence

The dependency and Factory are now available, but `LangGraphResearchRuntime` still defaults to its existing `InMemorySaver` construction. A later change must inject the Factory at Runtime construction, place `checkpoint.db` in the Session-private directory, and test connection lifecycle and Session isolation before enabling SQLite persistence in production.
