# Benjamini–Hochberg (FDR control)

## TL;DR

When you test many things at once (many metrics × many slices), some will look "significant"
by pure chance. Benjamini–Hochberg decides which p‑values to trust while controlling the
**false discovery rate** — the expected fraction of your "wins" that are actually noise.

## What it measures & why you'd use it

Test 20 slices at p < 0.05 and you expect ~1 false positive even if nothing changed. A
dashboard full of metrics and slices is a multiple‑comparisons minefield. **BH** is the
modern, less‑conservative alternative to Bonferroni: instead of controlling the chance of
*any* false positive, it controls the *proportion* of false positives among your rejections,
which keeps power while still protecting you. The harness applies it across every
`ComparisonResult` in `evalctl runs compare`.

## Intuition (tiny worked example)

p‑values `[0.001, 0.02, 0.5]` at q = 0.05. Sorted, compare each to \(q \cdot \text{rank}/m\):

- rank 1: 0.001 ≤ 0.05·1/3 = 0.0167 ✓
- rank 2: 0.02 ≤ 0.05·2/3 = 0.0333 ✓
- rank 3: 0.5 ≤ 0.05·3/3 = 0.05 ✗

Largest passing rank is 2 → reject the first two. Result `[True, True, False]`, exactly the
assertion in [`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
(`test_exact_mcnemar_and_bh`).

## Formal definition

Sort p‑values ascending. Find the largest rank \(k\) such that \(p_{(k)} \le \frac{k}{m} q\);
reject all hypotheses with rank ≤ \(k\).

```71:81:src/evalharness/statistics/core.py
def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    cutoff = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / count:
            cutoff = rank
    rejected = [False] * count
    for rank, index in enumerate(order, start=1):
        rejected[index] = rank <= cutoff
    return rejected
```

The scan keeps the **largest** passing rank (a step‑up procedure), so a later small p‑value
"rescues" earlier ones — the returned list is in the **original input order**.

## Inputs & requirements

- **Signature:** `benjamini_hochberg(p_values: list[float], q: float = 0.05)` →
  `list[bool]`.
- Wrapped by `apply_multiplicity(results, q=0.05)` in
  [`comparison.py`](../../../src/evalharness/statistics/comparison.py), which sets
  `significant_bh` on each `ComparisonResult`.

## Output

- A boolean list aligned to the input p‑values: `True` = reject the null (a real difference).

## Pitfalls & gotchas

- **Controls FDR, not FWER.** BH allows a controlled *fraction* of false discoveries — it's
  the right trade for exploratory eval dashboards, but if a single false positive is
  catastrophic, use a family‑wise method.
- **Apply across the whole family of tests**, not per‑metric in isolation — the point is the
  joint decision. `apply_multiplicity` does this over the full result list.
- **A single comparison still gets `significant_bh`.** With one test, BH reduces to the raw
  threshold; the field is always populated so downstream code is uniform.
- **q is not α.** `q` is the target false‑discovery *rate*; don't conflate it with a
  per‑test significance level.

## How it composes

- The final stage of the [comparison pipeline](mcnemar.md): McNemar p‑values → BH →
  `significant_bh`.
- Pair with [Cohen's h](effect-sizes.md): BH says *which* differences are real; the effect
  size says which are *worth caring about*.

## References & code

- Code: [`benjamini_hochberg`](../../../src/evalharness/statistics/core.py),
  [`apply_multiplicity`](../../../src/evalharness/statistics/comparison.py).
- Test: `test_exact_mcnemar_and_bh`.
- Guide: [§6.7](../../guide.md#67-statistics-package).
- Lineage: Benjamini & Hochberg (1995).
