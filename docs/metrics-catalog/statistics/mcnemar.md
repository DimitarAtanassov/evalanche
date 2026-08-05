# McNemar & paired comparison

Covers `exact_mcnemar`, `compare_binary`, `apply_multiplicity`, plus the two diagnostic
helpers `find_flaky_cases` and `between_repeat_variance`.

## TL;DR

To compare two models on the **same** cases with binary outcomes (pass/fail), you only care
about the cases where they **disagree**. McNemar's exact test asks whether the disagreements
are lopsided enough to be real. `compare_binary` wraps that with an effect size and a CI into
one result.

## What it measures & why you'd use it

When baseline and candidate are scored on identical cases, an unpaired test throws away the
pairing and loses power. **McNemar's test** looks only at discordant pairs — cases the
baseline got right and the candidate got wrong (\(b\)), and vice‑versa (\(c\)) — and tests
whether \(b\) and \(c\) are balanced. This is the correct significance test for paired binary
A/B comparisons (`evalctl runs compare`).

## Intuition (tiny worked example)

`baseline = [T, T, F]`, `candidate = [F, T, T]`. Case 1: baseline‑only correct → \(b=1\).
Case 3: candidate‑only correct → \(c=1\). Case 2: both correct (concordant, ignored). With
\(b=c=1\), the disagreements are perfectly balanced → **p = 1.0**. That's the assertion in
[`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
(`exact_mcnemar(...) == (1, 1, 1.0)`).

## Formal definition

\(b\) = baseline‑correct & candidate‑wrong, \(c\) = baseline‑wrong & candidate‑correct. The
exact test is a two‑sided binomial test of \(\min(b, c)\) successes in \(b+c\) trials against
\(p = 0.5\):

```62:68:src/evalharness/statistics/core.py
def exact_mcnemar(baseline: list[bool], candidate: list[bool]) -> tuple[int, int, float]:
    if len(baseline) != len(candidate):
        raise ValueError("Paired samples must have identical length")
    b = sum(old and not new for old, new in zip(baseline, candidate, strict=True))
    c = sum(not old and new for old, new in zip(baseline, candidate, strict=True))
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return b, c, p
```

`compare_binary` bundles this with a [paired bootstrap](bootstrap.md) delta and
[Cohen's h](effect-sizes.md) into a `ComparisonResult`:

```45:63:src/evalharness/statistics/comparison.py
    delta, interval = paired_bootstrap(
        [float(value) for value in baseline],
        [float(value) for value in candidate],
        seed=seed,
    )
    _, _, p_value = exact_mcnemar(baseline, candidate)
    effects = effect_sizes(old_rate, new_rate)
    return ComparisonResult(
        metric=metric,
        n=len(baseline),
        baseline=old_rate,
        candidate=new_rate,
        absolute_delta=delta,
        relative_delta=effects["relative_delta"],
        cohens_h=float(effects["cohens_h"] or 0.0),
        ci_low=interval[0],
        ci_high=interval[1],
        p_value=p_value,
    )
```

## Inputs & requirements

- **`exact_mcnemar(baseline, candidate)`** — equal‑length boolean lists (raises otherwise).
- **`compare_binary(metric, baseline, candidate, *, seed=0)`** — aligned booleans; empty
  input raises ("No aligned, non‑flaky cases to compare").
- **`apply_multiplicity(results, q=0.05)`** — a list of `ComparisonResult` → same list with
  `significant_bh` set by [Benjamini–Hochberg](benjamini-hochberg.md).

## Output

- `exact_mcnemar` → `(b, c, p_value)`.
- `compare_binary` → `ComparisonResult(metric, n, baseline, candidate, absolute_delta,
  relative_delta, cohens_h, ci_low, ci_high, p_value, significant_bh=False)`.

## The diagnostic helpers

- **`find_flaky_cases(outcomes)`** → cases whose repeated boolean outcomes are **not all
  equal** (`len(set(values)) > 1`). These are **excluded from compare claims** — a case that
  flips across repeats can't cleanly credit a win/loss — but they are **never removed from
  storage** ([principle #1/#3](../../principles.md)). At `temperature = 0`, flakiness means
  the backend isn't bit‑deterministic.
- **`between_repeat_variance(values_by_case)`** → the mean of per‑case sample variances
  (ddof=1), a stability diagnostic: high variance across repeats means noisy generation.

## Pitfalls & gotchas

- **Exclude flaky cases first.** `runs compare` drops them before `compare_binary`; comparing
  on flaky cases inflates or masks differences.
- **Concordant pairs carry no information.** McNemar ignores cases both models get right/wrong
  — a huge n with few disagreements still yields a large p‑value. That's correct, not a bug.
- **p‑value ≠ effect size.** A significant McNemar with a tiny [Cohen's h](effect-sizes.md)
  is a real but trivial difference. Report both.
- **Always multiplicity‑correct.** Comparing many metrics/slices without
  [BH](benjamini-hochberg.md) manufactures false positives.

## How it composes

- The significance engine of `evalctl runs compare`: McNemar (p) + paired bootstrap (delta
  CI) + Cohen's h (effect) → `ComparisonResult`, then BH across the list.
- Operates on any metric's per‑case booleans — most commonly
  [`exact_match`](../lexical-structured/exact-match.md)`.passed` (SQL query 11 in
  [guide §5.5](../../guide.md#55-query-library)).

## References & code

- Code: [`core.py`](../../../src/evalharness/statistics/core.py),
  [`comparison.py`](../../../src/evalharness/statistics/comparison.py).
- Test: `test_exact_mcnemar_and_bh`.
- Guide: [§6.7](../../guide.md#67-statistics-package), [§4.5](../../guide.md#45-evalctl-runs-compare).
- Lineage: McNemar (1947).
