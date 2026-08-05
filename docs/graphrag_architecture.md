# Incremental Agentic Scientific GraphRAG

## Planes

The Index Plane converts canonical structure-aware chunks into one active, application-level
epoch across three independently transactional stores. `index_registry.sqlite3` is the commit
authority; `lexical.sqlite3` contains FTS5 versions; Qdrant contains named `dense` vectors; and
Neo4j contains both provenance and semantic graphs. A failed pending epoch never changes query
visibility.

The Query Plane routes and rewrites a question, creates at most five retrieval queries, rejects
drifted variants, dispatches lexical/dense/graph/community routes, applies route-weighted and
query-weighted RRF, reranks with the original question, checks graph/source coverage, and
generates only from graph paths, claims and original chunks. Legacy hybrid retrieval remains
available only as an ablation.

## Registry schema

`index_epochs` stores model/policy/ontology versions and pending/active/failed/superseded state.
`documents`, `chunk_versions`, `embedding_entries`, `graph_extractions`, and
`community_versions` store version validity. `index_operations` is the idempotent outbox.
Entity aliases, resolution events and reversible merge events are also persisted here; the
Agent Session database is never reused.

Visibility is always:

```sql
valid_from_epoch <= :active_epoch
AND (valid_to_epoch IS NULL OR :active_epoch < valid_to_epoch)
```

## Stable identities and incremental synchronization

`chunk_key` hashes `document_id`, `section_id`, ordered block IDs, and chunk policy version.
Display `chunk_id` remains unchanged. Independent content, dense-text and graph-text hashes
classify added, dense-changed, graph-changed, unchanged and deleted chunks. Unchanged dense
texts never invoke the embedding provider. Point IDs are UUID5 over chunk key, dense hash,
model and revision. A model/revision change builds a separate versioned collection before the
alias can switch.

Every outbox operation hashes epoch, store, operation type, object ID and target hash. Backend
writes are idempotent. Only after store/source validation succeeds are old versions closed and
the pending epoch atomically activated.

## Neo4j model

Constraints are created for `Work.work_id`, `Document.document_id`, `Section.section_id`,
`Chunk.chunk_key`, `Entity.entity_id`, `RelationClaim.claim_id`, and
`Community.community_id`. Indexes cover normalized entity name/type, claim predicate, and
chunk document/work IDs. `entity_names` is a full-text index over canonical name, normalized
name, and aliases.

Provenance edges are `HAS_DOCUMENT`, `HAS_SECTION`, and `HAS_CHUNK`. Semantic evidence uses
`MENTIONS`, `SUPPORTS`, `SUBJECT`, and `OBJECT`. A `RelationClaim` retains polarity,
qualifiers, quote, confidence, chunk/block/page provenance and extractor versions. `RELATED`
is only a support/oppose/neutral aggregate; conflicting claims are never overwritten.

## Qdrant collections

The stable aliases are `leo_chunks_dense`, `leo_entities_dense`, and
`leo_community_reports_dense`; concrete collections append a model/revision fingerprint. All
collections use the named vector `dense` with cosine distance. Chunk payloads carry the full
source identity and epoch validity.

## Harness

```text
INITIALIZED → ROUTING → PLANNING → QUERY_EXPANDING → QUERY_VALIDATING
→ RETRIEVAL_DISPATCHING → {LEXICAL,DENSE,GRAPH,COMMUNITY}_RETRIEVING
→ QUERY_FUSING → RERANKING → COVERAGE_CHECKING → EVIDENCE_SELECTING
→ CONTEXT_BUILDING → [COMPACTING] → GENERATING → STRUCTURAL_VALIDATING
→ SEMANTIC_VALIDATING → [REPAIRING | focused RETRIEVAL_DISPATCHING]
→ PERSISTING → COMPLETED | REFUSED | FAILED
```

The second retrieval round permits at most two coverage-focused queries. Graph traversal is
bounded to two hops and twenty paths.

## Query expansion, drift and fusion

RQ0 is always the original question. Complexity controls zero-to-four additional queries.
The drift guard checks target category, named constraints, unexpected categories, semantic
similarity and duplicates; one optional adjudication is allowed. Rejected expansions are
dropped and RQ0 remains.

Within and across queries, the configured score is
`Σ query_weight × route_weight / (rrf_k + rank)`. Repeated matches accumulate and retain all
query IDs, routes, per-query ranks and per-route contributions. The original question—not an
expansion—is used for cross-encoder reranking.

## Operations

```bash
docker compose -f docker-compose.graph.yml up -d
uv run python main.py knowledge migrate-to-graphrag
uv run python main.py knowledge sync
uv run python main.py knowledge status
uv run python main.py graph validate
uv run python main.py answer "伪距率与速度状态有什么关系？" --retrieval-mode graphrag
```

Migration reads the existing `data/knowledge/chunks.jsonl`; no PDF upload or identity rewrite
is performed. Real Neo4j integration tests are optional in CI and can be run locally after the
Compose service is healthy.
