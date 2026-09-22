# CrewAI Release Gate

Run:

```bash
uv run python main.py scholar release-check
```

The command emits a structured JSON report with exactly one release status:

- `CREWAI_PRODUCTION_READY`
- `CREWAI_PRODUCTION_EXTERNAL_BLOCKED`
- `CREWAI_PRODUCTION_NOT_READY`

It checks the CrewAI dependency and Flow path, the Manager plus three
specialist capability boundary,
local async/event-store contracts, CrewAI production-default configuration,
and the persisted Knowledge Index readiness projection. The Knowledge Index
check covers Paper BM25/Dense, Content BM25/Dense, source digests, coverage,
manifest provenance and embedding revision; it never runs a rebuild. Missing
external provider configuration is reported as external blocked only after all
local engineering checks pass; code, contract and Knowledge Index failures are
not hidden.

The current code-level closure verifies the deterministic provider/fixture path,
including Introduction Reviewer PASS and Human Approval, and verifies that an
unavailable provider fails without a fabricated answer. A real external
Provider, production paper corpus and deployed Worker still require an explicit
environment-level E2E; this repository state must not claim those external
conditions as verified from fixture tests alone.

The operational split is explicit:

| Projection | Meaning |
| --- | --- |
| Infrastructure / Runtime | API, Worker, stores and CrewAI runtime can start |
| Knowledge Index | Current Paper and Content projections are initialized and consistent |
| External Provider | Requires explicit real-provider E2E in the target environment |

`uv run python main.py knowledge status` is the administrator-facing source for
the current index status. `docker compose up` does not implicitly rebuild it.
