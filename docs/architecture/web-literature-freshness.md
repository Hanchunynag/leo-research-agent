# Phase 2D Web Literature & Freshness Policy

Web Literature is a Research Capability adapter, not a Skill, Agent, Planner or
second evidence system. `ResearchCapabilityService.research()` remains the
only routing boundary:

```text
Skill
  → ResearchRequest
  → FreshnessPolicy
  → Local Research
  → Evidence Coverage
  → bounded WebLiteratureAdapter (optional)
  → metadata normalization / canonical deduplication
  → External Source Resolver
  → EvidenceIntelligencePipeline
  → Verified EvidencePack
  → Citation / Reviewer / Skill Result
```

## Freshness

`ResearchRequest.freshness_mode` is one of `LOCAL_ONLY`, `LOCAL_FIRST` or
`FRESH_REQUIRED`. The pure `FreshnessPolicy` also consumes explicit date
windows, `explicit_latest`, domain sensitivity, local coverage, unresolved
claims, conflicts and publication dates. `retrieved_at` never substitutes for
`publication_date`.

- `LOCAL_ONLY` never calls Web.
- `LOCAL_FIRST` uses verified local evidence when coverage and provenance are
  sufficient, and otherwise performs bounded Web fallback.
- `FRESH_REQUIRED` always performs an external freshness check. Missing or
  unverifiable external publication/content evidence fails closed, including
  when the provider is unavailable.

The decision is serialized in the EvidencePack metadata and in the
`FRESHNESS_DECIDE` RunEvent trace step.

## External Evidence

Search results are discovery candidates only. An external candidate must carry
`canonical_id`, a real `source_locator`, `locator_type` (`ABSTRACT` or
`FULLTEXT_SPAN`), provider, retrieval time and validation state. The adapter
can create a candidate from a verified provider abstract or full-text span;
snippets and title similarity alone are rejected. The existing evidence
pipeline then asks the injected External Source Resolver to verify content and
content hash before creating `VerifiedEvidence`.

Local evidence keeps its `document_id`/`chunk_id`/page/block contract. External
evidence does not fabricate those fields and is rendered with its canonical
locator. Citation resolution reuses an existing project BibKey or emits a
Human-approved BibEntryCandidate; a DOI without enough metadata remains a
citation requirement.

## Identity, cache and ownership

Deduplication uses DOI, versionless arXiv ID, trusted external IDs and finally
the conservative title/first-author/year identity. A Web hit matching a local
work reuses the local parsed source; it does not create a duplicate Document
or write the workspace corpus. New external sources are request-scoped.

`LiteratureDiscoveryCache` stores only normalized discovery/metadata with a
fingerprint, retrieval time and TTL. Cache hits still pass the current
FreshnessPolicy, canonical deduplication and Evidence validation; the cache is
not an Evidence Store.

Provider failures remain structured (`timeout`, `rate limit`, unavailable,
metadata/identity failures) and are recorded in pack metadata and the RunEvent
tool/provider trace. They never become `UNSUPPORTED` claims. Local
fallback may remain usable; `FRESH_REQUIRED` returns unresolved/fail-closed
when freshness cannot be verified.

Conclusion and Abstract profiles retain no local or Web Research capability.
Web Literature therefore cannot bypass Skill capability enforcement, and it
does not alter DraftPatch or Human Approval lifecycle.

When Web Verified Evidence is later used by Introduction, Citation Lifecycle
resolves its DOI/arXiv identity against the current project `references.bib`.
Web discovery cache remains insufficient for Citation audit; the minimal
External Evidence projection is persisted only when a Citation Binding is
created.

## Orchestration boundary

The Manager does not call a Web provider. Research specialists receive only
the high-level `ResearchCapabilityService` adapter; the adapter
constructs `ResearchRequest`, applies this policy and records the existing
Research trace. Production composition supplies the same Web adapter
to Web and CLI runtimes, while Conclusion and Abstract remain without any
Research capability.
