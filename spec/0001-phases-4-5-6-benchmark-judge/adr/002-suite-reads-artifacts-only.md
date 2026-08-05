# ADR 002: Suite reads published artifacts only

Status: accepted · Date: 2026-08-05  
Deciders: evalanche architects

## Context

Phase 5 needs a detailed multi-run benchmark view. The per-run report is already
a trusted, versioned JSON contract (2.1). Allowing the suite builder to query
PostgreSQL or call providers would couple visualization to live state, risk
rescoring side effects, and tempt changes to single-run reports.

Dominating constraint: **keep single-run report 2.1 unchanged and suite
reproducible from files alone**.

## Decision

`evalharness.suite` may read only:

- `suite.yaml`
- member run report JSON (`schema_version` 2.1)
- optional compare JSON (`schema_version` 1.0)
- optional `calibration.json` paths (source of truth for judge gate badges)
- optional judgment/RAG JSON artifacts by path (summaries only; gate bit not
  inferred from judgment when `calibration_digest` is null)

It must not import the store, open `DATABASE_URL`, invoke providers, or rewrite
member artifacts. HTML is a pure view over `suite.json`. Gallery text stays
within Phase 3 truncation bounds.

## Consequences

Easy: offline sharing, golden tests, clear trust boundary. Hard: operators must
export reports before suite build; stale files possible (mitigate with digests
in suite identity). Give up: live DB explorer (Phase 5 non-goal).

## Alternatives considered

- **Suite queries Postgres:** Faster for local ops; couples to schema and breaks
  offline HTML story. Rejected.
- **Extend RunReport 2.1 with suite sections:** Bloated single-run narrative;
  violates phase goal. Rejected.
