# Contract: benchmark suite (Phase 5)

Status: accepted · Version: **0.1** · Consumers: `evalctl suite *`, suite HTML

## Purpose

Multi-run benchmark input and derived view over **published artifacts only**.

## Input: `suite.yaml`

```yaml
schema_version: "0.1"
name: nightly-quality
description: "Optional human description"
member_runs:
  - path: artifacts/runs/example.json   # RunReport 2.1 JSON
    role: candidate                     # baseline | candidate | reference
    label: "llama3.2:3b / squad-dev"
    # optional overrides for display only:
    model: "llama3.2:3b"
    prompt: "qa-v1"
    dataset: "squad-dev"
    domain: "general"
    task: "qa_short"
compares:
  - path: artifacts/compares/example.json  # runs compare schema 1.0
primary_metrics:
  - dataset: squad-dev
    metric: squad_f1
# Phase 6 optional:
# calibrations:
#   - path: artifacts/judges/calibration.json
# judge_artifacts / rag_artifacts: paths only; gate bit from calibration paths
```

Validation rules:

1. `schema_version` must be in the suite allow list (initially only `"0.1"`).
2. Each `member_runs[].path` must load JSON with `schema_version == "2.1"`.
   Unknown or older run schemas → reject member with `UNSUPPORTED_SCHEMA`.
3. Each compare path must have `schema_version == "1.0"` (current CLI).
4. Roles are from the enum above; default unique paths.
5. `primary_metrics` entries must reference dataset names present in members'
   `dataset.name` (or declared label dataset override).
6. Failure gallery text inherits report truncation (280 chars); no
   `raw_response`.

## Output: `suite.json`

Derived, deterministic view model (`schema_version: "0.1"`). Required top-level:

| Field | Meaning |
|-------|---------|
| `name`, `description` | From manifest |
| `suite_digest` | SHA-256 over canonical JSON of inputs (sorted member digests + compare digests + primary_metrics) |
| `members[]` | Per-run identity: run_id, paths, role, label, publishable, coverage, model_digest, dataset_sha256, primary_metric headline |
| `exclusions[]` | `{run_id, reason}` for leaderboard exclusions |
| `coverage_matrix` | members × publishable/coverage |
| `quality_tables` | per domain/task with metric, n, CI |
| `leaderboards[]` | declared members only; skip non-publishable by default |
| `slices[]` | overall + weakest-first slices from member aggregates |
| `comparisons[]` | paired deltas, significance, effect size, flaky exclusions from compare JSON |
| `failure_gallery[]` | bounded from `case_examples` (≤280 chars; no `raw_response`) |
| `ops` | latency/cost/retries/cache/harness_failures rollups |
| `calibrations[]` | optional summaries from calibration.json (source of `gating_allowed`) |
| `judge_artifacts` / `rag_artifacts` | optional paths+summaries (Phase 6; omit if absent) |

Serialization: UTF-8, stable key order for golden diffs (canonical dumps).

## Output: `suite.html`

Self-contained HTML view over `suite.json`. Rules from `docs/reports.md`:

- Fixed chart div ids; one inlined Vega runtime; no CDN.
- Accessible table beside every chart.
- Omit empty panels.
- Show exclusion reasons; non-publishable must not silently enter leaderboards.
- Judge gate badges read calibration artifacts, not raw judgment `gating_allowed`
  unless `calibration_digest` is attached.

## Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `UNSUPPORTED_SCHEMA` | Member/compare/suite version unknown | No |
| `MISSING_ARTIFACT` | Path not found | No |
| `NON_PUBLISHABLE_MEMBER` | Informational exclusion, not hard fail of build | N/A |
| `PRIMARY_METRIC_UNKNOWN` | Declared metric missing in member aggregates | No |

## Hard non-goals (enforce in code)

- Import `evalharness.store` or open `DATABASE_URL`
- Call providers or rescore
- Mutate or rewrite member run JSON/HTML
- Fetch network resources at render time

## Compatibility

- **Within `0.1`:** additive optional fields only (expand-then-contract). Readers
  ignore unknown optional fields; writers must not remove required `0.1` fields.
- **Incompatible changes:** bump to a new `schema_version` (e.g. `0.2`) and add
  it to the validator allow list. Old builders/readers that do not list the new
  version reject with `UNSUPPORTED_SCHEMA`.
- Do not treat "0.x" as free-for-breakage without a version bump.
- Run report stays **2.1** unchanged.
