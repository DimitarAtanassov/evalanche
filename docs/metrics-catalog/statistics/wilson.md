# Wilson score interval

## TL;DR

The right confidence interval for a **pass rate** (a proportion). Unlike the naive
"± 1.96·std" interval, it never runs off the ends of `[0, 1]` and stays sensible when the
rate is near 0%, near 100%, or the sample is small.

## What it measures & why you'd use it

Every Bernoulli aggregate in the harness — `exact_match` pass rate, `assertions`,
thresholded metrics — ships with a Wilson 95% interval. It answers "given `k` successes out
of `n`, what's the plausible range for the true success probability?" The Wilson interval is
the default because the textbook normal‑approximation ("Wald") interval is badly wrong for
small `n` or extreme rates (it can give negative lower bounds or a zero‑width interval at
0/n).

## Intuition (tiny worked example)

50 successes out of 100 → point estimate 0.5, Wilson 95% ≈ **[0.404, 0.596]**. That's the
golden value asserted in
[`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
(`wilson_interval(50, 100)` → `(0.4038…, 0.5962…)`). Note the interval is symmetric here
because 0.5 is centered; at `9/10` it would be asymmetric, hugging the ceiling.

## Formal definition

For `successes` \(k\), trials \(n\), \(\hat p = k/n\), and \(z\) the normal quantile for the
confidence level:

\[
\text{center} = \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad
\text{margin} = \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.
\]

```13:21:src/evalharness/statistics/core.py
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
```

## Inputs & requirements

- **Signature:** `wilson_interval(successes: int, n: int, confidence: float = 0.95)` in
  [`statistics/core.py`](../../../src/evalharness/statistics/core.py) (z from the Normal
  quantile).
- [`scoring/stats.py`](../../../src/evalharness/scoring/stats.py) re‑exports this same
  function for the scoring path; it is a compatibility shim, not a second implementation.
  Import from `evalharness.statistics` in new code.
- `n = 0` → `(0.0, 0.0)`.

## Output

- A `(low, high)` tuple clamped to `[0, 1]`. `high − low` shrinks as `n` grows.

## Pitfalls & gotchas

- **Read the interval, not just the point.** A 100% rate over `n=5` is `[0.566, 1.0]` — not
  a victory. This is the whole reason the CI ships.
- **Wilson is for proportions only.** For the mean of a *continuous* metric (ROUGE, cosine),
  use a [bootstrap](bootstrap.md), not Wilson.
- **Thresholded aggregates count *passes*, not the mean.** Several `ScalarMetric`s Wilson the
  count of `value ≥ threshold`, so the interval is about the pass rate at that threshold, not
  the mean score — know which you're reporting.
- **One implementation, two import paths.** `scoring/stats.py` re‑exports the
  `statistics/core.py` function, so both paths give identical numbers.

## How it composes

- The default interval for every rate in the catalog — `exact_match`, `assertions`,
  `classification` accuracy, and thresholded overlap metrics.
- For A/B *differences* in rates, move to [McNemar + paired bootstrap](mcnemar.md); Wilson is
  for a single rate.

## References & code

- Code: [`wilson_interval`](../../../src/evalharness/statistics/core.py).
- Test: `test_wilson_and_pass_at_k_golden`.
- Guide: [§6.7](../../guide.md#67-statistics-package).
- Lineage: Wilson (1927).
