# Principles

Non-negotiable rules for **evalanche**. Violating them in a PR is grounds for rejection.

1. **Immutability.** Generation rows are written once. Scores are separate. Corrections are new rows, not silent mutations. (There is a narrow interim outcome update to `failed_score` after scoring; prefer eliminating that in favor of score-only interpretation.)

2. **Content addressing.** Datasets, templates, and configs are hashed. Different hashes ⇒ different runs. No silent cross-hash comparison.

3. **Harness errors ≠ model errors.** Always report separately. Exclude harness failures from model-quality denominators; report coverage loss.

4. **No point estimate without an interval.** Rates ship with 95% CI (Wilson for binomials today). Continuous metrics and comparisons must add bootstrap / paired tests and multiplicity control as they land.

5. **Generation ≠ scoring.** Module and schema boundaries enforce it. Rescore must not call providers.

6. **Everything versioned.** Dataset, template, model digest, decode params, metric implementation, normalizer ruleset.

7. **Deterministic where possible, honest where not.** Record seed/temperature/top_p/top_k and whether the provider honors seeding.

8. **Resume over restart.** Checkpoints after each generation; resume skips completed `(case_id, repeat_idx)`.

9. **One provider file to extend.** New backends: implement `Provider` + entry-point line. No runner/scorer/store/API edits for adapters.

10. **Docs match code.** If behavior changes, update `docs/` in the same PR.
