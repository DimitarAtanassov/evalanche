# Principles

## Purpose

These are the **non‑negotiables** for evalanche. They are not style preferences — they
are the invariants that make results reproducible and honest, and they are enforced by
code and schema, not by convention. Violating one in a PR is grounds for rejection.
Each principle below states the rule, *why* it exists, and where it lives in the code.

## The non‑negotiables

1. **Immutability.** Generation rows are written once with provider‑time outcomes.
   Scores are separate rows; corrections are new rows, not silent mutations — scoring
   never updates `generations.outcome`.
   *Why:* a frozen output is the only thing you can honestly re‑score months later.
   *Where:* `Executor` writes `generations`; `ScoringEngine` only inserts `scores` /
   `metric_aggregates`. See [dataplane.md](dataplane.md#the-two-stages).

2. **Content addressing.** Datasets, templates, and configs are hashed. Different
   hashes ⇒ different runs. Upserting an existing `(name, version)` with a different
   hash is a hard error; resume verifies FKs and `config_sha256`.
   *Why:* prevents "we improved 3%" when the dataset silently changed underneath you.
   *Where:* `hashing.py`, `datasets/loader.py`, resume checks in `cli.py` / `executor.py`.

3. **Harness errors ≠ model errors.** Always report them separately. Exclude harness
   failures from coverage denominators; report coverage loss against planned
   `(case, repeat)` cardinality.
   *Why:* a flaky network must never make a model look worse.
   *Where:* `domain/enums.FailureOutcome`; coverage math in `reporting/report.py`. See
   [dataplane.md](dataplane.md#coverage-and-publishability).

4. **No point estimate without an interval.** Rates ship with a 95% CI (Wilson for
   binomials today). Continuous metrics and comparisons add bootstrap / paired tests
   and multiplicity control.
   *Why:* a bare "92%" hides whether *n=25* or *n=25,000*.
   *Where:* `statistics/` (`wilson_interval`, `bca_bootstrap`, `compare_binary`);
   surfaced in reports and the CLI summary. See [metrics.md](metrics.md#statistics-honest-comparison).

5. **Generation ≠ scoring.** Module and schema boundaries enforce it. Rescore must not
   call providers.
   *Why:* scoring cheaply and repeatedly is the whole point of the system.
   *Where:* `ScoringEngine.rescore_run` (no provider argument); `evalctl runs rescore`.

6. **Everything versioned.** Dataset, template, model digest, decode params, metric
   implementation, normalizer / metric config.
   *Why:* reproducibility and honest diffs.
   *Where:* `runs.config_sha256`; `scores.(metric_version, metric_config_sha256)`.

7. **Deterministic where possible, honest where not.** Record seed / temperature /
   top_p / top_k and whether the provider honors seeding.
   *Why:* local models rarely guarantee bit‑exact repeats — say so instead of pretending.
   *Where:* `runs.decode_params`; `Capabilities.supports_seed`; flaky‑case detection in
   `statistics/`.

8. **Resume over restart.** Checkpoint after each generation; resume skips completed
   `(case_id, repeat_idx)` with idempotent inserts.
   *Why:* a long run that dies at 90% must not restart from zero.
   *Where:* `UNIQUE (run_id, case_id, repeat_idx)` + `ON CONFLICT DO NOTHING`.

9. **One provider file to extend.** New backends implement `Provider` + one
   entry‑point line. No runner/scorer/store/API edits for adapters.
   *Why:* keeps the blast radius of a new backend tiny.
   *Where:* `domain/provider.Provider`, `providers/registry.py`, `pyproject.toml` entry
   points. See [providers.md](providers.md#adding-a-provider).

10. **Docs match code.** If behavior changes, update `docs/` in the same PR.
    *Why:* stale docs are worse than none; this doc set is part of the contract.

## Related

- [architecture.md](architecture.md) — the seams these principles protect
- [dataplane.md](dataplane.md) — immutability, coverage, and resume in action
- [guide.md §1.3](guide.md#13-core-invariants-the-nonnegotiables) — the same invariants with a code‑location table
