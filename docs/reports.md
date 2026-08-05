# Reports

## Purpose

This document describes the **artifacts a run produces**, the **dashboard they render**,
and the honesty rules baked into them. Reporting is the read‑only end of the
[data plane](dataplane.md#reporting); the numbers it presents come from the metric
catalog ([metrics.md](metrics.md)) and statistics package.

## Artifacts

Every run writes three files under the `--output` directory, named by run id:

| File | Format | For |
|------|--------|-----|
| `{run_id}.json` | Versioned JSON (`schema_version`) | Machine consumption, diffing, archival evidence |
| `{run_id}.html` | Self‑contained HTML (inline Vega‑Lite) | Humans — one file, no external assets |
| `{run_id}.xml` | JUnit XML | CI systems — coverage and pass‑rate as test cases |

**The JSON is the contract; the HTML is a view over it.** Every number, chart, and table
in the dashboard is a projection of the JSON artifact, so a future explorer UI or a
notebook reads the same fields the report renders.

Schema `2.0` replaced the former per‑audience `views` object with the flat fields below.
Consumers pinned to `1.0` must migrate: `views.leadership.cost_per_correct` is now
`cost_per_correct`, `views.engineering.{retries,cache_hits,cache_rate}` are top level,
and `views.research.confidence_method` is `confidence_method`.

Schema `2.1` is additive: it adds structured **evaluation context** so a reader can
understand the dashboard without querying the database — `model`, `dataset`,
`prompt_template`, `decode_params`, and a bounded `case_examples` sample (failures
first; text truncated; no `raw_response`).

Reporting is **read‑only**: `evalctl run` computes idempotent scores and aggregates
first (via `ScoringEngine`), then the reporter reads stored rows. It never mutates
evaluation state, and it never re‑calls a provider.

## What every report contains

The `RunReport` (`reporting/report.py`) carries the run's identity and verdict:

- **Identity** — `config_sha256`, `model_digest`, `dataset_sha256`, `run_status`.
- **Context** — `model` (provider, name, resolved version, optional size/context),
  `dataset` (name/version/split/case count/slice dimensions), `prompt_template`
  (name/version/truncated body), `decode_params`, and `case_examples` (bounded
  input / reference / output rows with primary-metric pass/fail).
- **Coverage** — `planned_generations`, `written_generations`, `coverage`,
  `coverage_floor`, and the `publishable` verdict.
- **Quality** — `pass_rate` with its denominator `pass_rate_n` and a Wilson 95%
  `pass_rate_ci`, the `primary_metric` that pass rate is computed from, and the
  `metric_aggregates` (value, CI, method, config hash) for the overall slice **and each
  slice rollup**.
- **Operational** — `outcome_histogram`, `harness_failures`, `latency` percentiles
  (p50/p90/p95/p99 + mean + max), `finish_reasons`, `retries`, `cache_hits`,
  `cache_rate`, and a sample of `trace_ids`.
- **Cost** — `cost_usd_total` and `cost_per_correct` (`null` when nothing passed).

## One dashboard, read top to bottom

The HTML is a **single narrative** that any reader can follow — there are no per‑audience
tabs. Each section is titled as the question it answers, and depth is available through
disclosure rather than separation:

| Section | Question it answers | Contains |
|---------|---------------------|----------|
| **Verdict** | Can I trust this run? | Publishable badge plus each gate itemized with its actual value |
| **Headline** | What are the numbers? | Pass rate with CI and n, coverage against floor, cost per correct, latency p95 |
| **Context** | What was evaluated? | Model, dataset, decode params, truncated prompt, sampled inputs/outputs |
| **Quality** | How good is the model? | Metric scores with CIs; pass rate by slice, worst first |
| **Reliability** | What broke, whose fault? | Outcomes split into model outcomes and harness failures; finish reasons, retries, cache |
| **Cost and speed** | What did it cost? | Latency percentiles beside the table that includes the mean |
| **Provenance** | How do I reproduce? | Config, dataset, and model digests; trace sample |

Two rules keep it honest. Each visualization has a neighboring **accessible data table**,
so the report is readable without relying on the chart alone. And **sections without data
are omitted rather than rendered empty** — a run with no slice rollups shows no slice
chart instead of an empty frame, and a chart never labels a "weakest" slice when every
slice scored the same.

## Honesty rules

These are the same invariants enforced elsewhere, made visible in the report:

- **Coverage uses planned `(case, repeat)` cardinality as its denominator.** A partial
  run cannot become publishable simply because missing rows are absent — see
  [dataplane.md](dataplane.md#coverage-and-publishability).
- **No point estimate without an interval.** The pass rate always ships with its Wilson
  CI ([principle #4](principles.md)).
- **Harness failures are excluded** from the model‑quality denominator and shown
  separately in the outcome histogram — the chart colors and labels the two categories
  differently so the distinction survives a glance.
- **Aggregates carry slice breakdowns.** `ScoringEngine` rolls up every metric per
  `dimension=value` alongside `__overall__`, and the dashboard sorts slices worst‑first
  so the weakest cohort leads rather than hiding behind a healthy average.
- **Publishability is gated**, not assumed: `completed` status + full planned coverage +
  coverage ≥ floor. The JUnit artifact fails the `coverage` test case when the gate
  fails, so CI can enforce it.

## Determinism

Charts are Vega‑Lite specifications rendered through Altair. Two details keep the
artifact byte‑reproducible, which is what makes the committed `fixtures/poc/` golden
comparison meaningful:

- **Fixed div ids.** Each chart is embedded with an explicit id rather than Altair's
  default random `altair-viz-<uuid>`.
- **One inlined runtime.** `vl_convert.javascript_bundle()` emits Vega, Vega‑Lite, and
  Vega‑Embed once into the document, so the report renders offline with no CDN request
  and carries a single copy of the runtime no matter how many charts it has.

Chart specs are JSON, so the same definitions can be re‑rendered elsewhere — a static
PNG for CI, or a future explorer UI — without reimplementing them.

## Related

- [dataplane.md](dataplane.md) — coverage, publishability, and the read‑only reporting stage
- [metrics.md](metrics.md) — the metrics and statistics behind the numbers
- [operations.md](operations.md#proof-of-concept) — the committed PoC report and golden test
- [guide.md §6.7](guide.md#67-statistics-package) — the confidence methods reported
