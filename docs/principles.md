# Principles

Non-negotiable rules for **evalanche**. Violating them in a PR is grounds for rejection.

1. **Immutability.** Generation rows are written once with provider-time outcomes. Scores are separate rows. Corrections are new rows, not silent mutations — scoring never updates `generations.outcome`.

2. **Content addressing.** Datasets, templates, and configs are hashed. Different hashes ⇒ different runs. Upserting an existing `(name, version)` with a different hash is a hard error. Resume verifies FKs and `config_sha256`.

3. **Harness errors ≠ model errors.** Always report separately. Exclude harness failures from coverage denominators; report coverage loss against planned `(case, repeat)` cardinality.

4. **No point estimate without an interval.** Rates ship with 95% CI (Wilson for binomials today). Continuous metrics and comparisons must add bootstrap / paired tests and multiplicity control as they land.

5. **Generation ≠ scoring.** Module and schema boundaries enforce it. Rescore must not call providers.

6. **Everything versioned.** Dataset, template, model digest, decode params, metric implementation, normalizer ruleset.

7. **Deterministic where possible, honest where not.** Record seed/temperature/top_p/top_k and whether the provider honors seeding.

8. **Resume over restart.** Checkpoints after each generation; resume skips completed `(case_id, repeat_idx)` with idempotent inserts.

9. **One provider file to extend.** New backends: implement `Provider` + entry-point line. No runner/scorer/store/API edits for adapters.

10. **Docs match code.** If behavior changes, update `docs/` in the same PR.
