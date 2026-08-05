# Reports

## Purpose

This document describes the **artifacts a run produces**, the **audiences they serve**,
and the honesty rules baked into them. Reporting is the read‑only end of the
[data plane](dataplane.md#reporting); the numbers it presents come from the metric
catalog ([metrics.md](metrics.md)) and statistics package.

## Artifacts

Every run writes three files under the `--output` directory, named by run id:

| File | Format | For |
|------|--------|-----|
| `{run_id}.json` | Versioned JSON (`schema_version`) | Machine consumption, diffing, archival evidence |
| `{run_id}.html` | Self‑contained HTML (inline Plotly) | Humans — one file, no external assets |
| `{run_id}.xml` | JUnit XML | CI systems — coverage and pass‑rate as test cases |

Reporting is **read‑only**: `evalctl run` computes idempotent scores and aggregates
first (via `ScoringEngine`), then the reporter reads stored rows. It never mutates
evaluation state, and it never re‑calls a provider.

## What every report contains

The `RunReport` (`reporting/report.py`) carries the run's identity and verdict:

- **Identity** — `config_sha256`, `model_digest`, `dataset_sha256`, `run_status`.
- **Coverage** — `planned_generations`, `written_generations`, `coverage`,
  `coverage_floor`, and the `publishable` verdict.
- **Quality** — `pass_rate` with a Wilson 95% `pass_rate_ci`, and the per‑metric
  `metric_aggregates` (value, CI, method, config hash).
- **Operational** — `outcome_histogram`, `latency` percentiles (p50/p90/p95/p99 + mean +
  max), `finish_reasons`, and a sample of `trace_ids`.

## Audience views

The HTML report coordinates three views over the same underlying data, each scoped to
what its audience needs — and what it should *not* see:

| View | Contains | Deliberately omits |
|------|----------|--------------------|
| **Leadership** | Headline coverage and a cost‑per‑correct summary | Prompts and raw provider responses |
| **Research** | Confidence method (Wilson 95%) and metric provenance; flaky cases excluded from claims | — |
| **Engineering** | Retries, cache hits/rate, latency, finish reasons, failures, trace samples | — |

Each visualization has a neighboring **accessible data table**, so the report is
readable without relying on the chart alone.

## Honesty rules

These are the same invariants enforced elsewhere, made visible in the report:

- **Coverage uses planned `(case, repeat)` cardinality as its denominator.** A partial
  run cannot become publishable simply because missing rows are absent — see
  [dataplane.md](dataplane.md#coverage-and-publishability).
- **No point estimate without an interval.** The pass rate always ships with its Wilson
  CI ([principle #4](principles.md)).
- **Harness failures are excluded** from the model‑quality denominator and shown
  separately in the outcome histogram.
- **Publishability is gated**, not assumed: `completed` status + full planned coverage +
  coverage ≥ floor. The JUnit artifact fails the `coverage` test case when the gate
  fails, so CI can enforce it.

## Determinism

The HTML embeds Plotly with a fixed chart `div_id` (rather than Plotly's default random
UUID) so byte‑identical runs produce byte‑identical reports — which is what makes the
committed `fixtures/poc/` golden comparison meaningful.

## Related

- [dataplane.md](dataplane.md) — coverage, publishability, and the read‑only reporting stage
- [metrics.md](metrics.md) — the metrics and statistics behind the numbers
- [operations.md](operations.md#proof-of-concept) — the committed PoC report and golden test
- [guide.md §6.7](guide.md#67-statistics-package) — the confidence methods reported
