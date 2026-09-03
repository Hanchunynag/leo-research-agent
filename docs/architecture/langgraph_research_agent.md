# LangChain + LangGraph Research Agent

The production entry point is `build_langchain_agent_service`.  The existing legacy RAG,
Selected Evidence boundary, Claim-Evidence validation, and Research Harness remain the
source of truth for retrieval and answer safety.  LangGraph adds state and orchestration:

```text
START
  -> TranslationSkill
  -> ReferenceResolver
  -> TaskClassifier
  -> ResearchPlanner
  -> ScopeReadSkill
  -> PaperResolver
  -> MetadataFetch
  -> Agent (structured action)
       ├─ tool -> observation -> Agent
       ├─ clarify -> interrupt -> resume -> Agent
       └─ final -> END
```

The development checkpointer is `InMemorySaver`.  A stable `thread_id` is used for an
interrupt/resume pair; the Web API exposes `POST /api/answers/resume` for this boundary.
The state includes the original query, bilingual retrieval query, task type, deterministic
research plan, current skill, scope, resolved papers, publication metadata, sorted timeline,
paper-to-evidence index, paper reference map, evidence, iteration limit, trace, tool calls,
and final answer.

`TaskClassifier` distinguishes direct QA, paper summary, multi-paper summary, author
analysis, timeline, comparison, and literature search. `ResearchPlanner` is deliberately
deterministic: it only declares metadata needs, evidence focus, ordering, and answer shape;
it does not select a retrieval backend or alter the user's retrieval query.

For a timeline task, `PaperResolver` lists the current immutable Workspace scope only when
no explicit paper reference is available, `MetadataFetch` resolves each paper's publication
date, and the Agent stores an ascending timeline before invoking the existing bilingual RAG.
For a comparison task, the plan requires method, data/experiment, contribution, and
limitation dimensions; returned Selected Evidence is indexed by document so synthesis can
avoid silently using one paper as evidence for another.

Chinese characters in the original user query set `output_language=zh`; this value is
included in the answer-generation contract, so English retrieval terms do not change the
answer language.  LLM usage is retained per call in the translation diagnostics, Agent
trace, and Research Harness provider usage, and is rendered in the Web Diagnostics panel.

The atomic document tools are available through `build_research_tools`:

```text
workspace.list_documents
document.get_outline
document.read
language.translate
```

The existing `knowledge.retrieve`, literature, parsing, and job tools are not removed and
the Agent never receives backend-specific Qdrant/BM25/Neo4j/LightRAG objects.
