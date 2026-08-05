# Bootstrap CIs: `bca_bootstrap` & `paired_bootstrap`

## TL;DR

For the **mean of a continuous metric** (ROUGE, cosine, latency) — where Wilson doesn't
apply — resample the data thousands of times to get a confidence interval. **BCa** is the
bias‑corrected, accelerated variant that adjusts for skew. `paired_bootstrap` does the same
for a **run‑vs‑run delta**.

## What it measures & why you'd use it

Wilson intervals only cover proportions. Continuous metrics need a different tool, and the
**bootstrap** is the general‑purpose answer: it estimates the sampling distribution of *any*
statistic by resampling the observed data with replacement. **BCa** improves on the naive
percentile bootstrap by correcting for bias and skew in that distribution, giving accurate
CIs even when the metric's distribution is lopsided (as ROUGE and cosine often are).
`paired_bootstrap` applies BCa to the per‑case differences \(c_i - b_i\), the correct CI for
"how much did candidate beat baseline?"

## Intuition (tiny worked example)

You have 200 per‑case ROUGE‑L scores; their mean is 0.42. Is the true mean plausibly 0.40,
or could it be 0.30? BCa resamples those 200 values 10,000 times, recomputes the mean each
time, and reports the 2.5th/97.5th percentiles (bias/skew‑adjusted) — say `[0.39, 0.45]`.
For a paired comparison, you instead bootstrap the 200 *deltas* between baseline and
candidate on the same cases, which cancels per‑case difficulty and tightens the interval.

## Formal definition

BCa adjusts the percentile bootstrap by a bias‑correction \(z_0\) and acceleration \(a\)
(estimated by jackknife), implemented via `scipy.stats.bootstrap(method="BCa")`. Paired:
\(\delta_i = c_i - b_i\); report \(\bar\delta\) and \(\text{BCa}(\delta)\).

```24:44:src/evalharness/statistics/core.py
def bca_bootstrap(
    values: list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        value = float(statistic(data)) if data.size else 0.0
        return value, value
    result = stats.bootstrap(
        (data,),
        statistic,
        method="BCa",
        n_resamples=resamples,
        confidence_level=confidence,
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)
```

```47:59:src/evalharness/statistics/core.py
def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    left = np.asarray(baseline, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired samples must have identical shape")
    delta = right - left
    return float(np.mean(delta)), bca_bootstrap(delta.tolist(), resamples=resamples, seed=seed)
```

## Inputs & requirements

- **`bca_bootstrap(values, statistic=np.mean, *, resamples=10_000, seed=0,
  confidence=0.95)`** — any list of floats and any statistic.
- **`paired_bootstrap(baseline, candidate, *, resamples=10_000, seed=0)`** — the two lists
  must have **identical shape** (aligned cases) or it raises.
- **Fewer than 2 values** → the CI collapses to the point value (no resampling possible).

## Output

- `bca_bootstrap` → `(low, high)`.
- `paired_bootstrap` → `(mean_delta, (low, high))`.

## Pitfalls & gotchas

- **Always seed when publishing.** Default `seed=0`; a bootstrap is stochastic, so an
  unseeded CI isn't reproducible.
- **Paired needs alignment.** The `strict` shape check enforces that baseline and candidate
  cover the same cases in the same order — mismatched inputs raise rather than silently
  mis‑pair.
- **BCa needs enough data.** With `n < 2` you get a degenerate interval; with very small `n`
  BCa's acceleration estimate is unstable — widen the dataset.
- **Use paired for comparisons.** Bootstrapping baseline and candidate *independently* and
  differencing the CIs is wrong and over‑wide; bootstrap the **deltas**.

## How it composes

- The continuous‑metric complement to [Wilson](wilson.md): Wilson for rates, BCa for means.
- `paired_bootstrap` is a component of [`compare_binary`](mcnemar.md) — it supplies the
  delta CI while McNemar supplies the p‑value.
- Use it to put a CI on mean [ROUGE](../text-overlap/rouge.md), [cosine](../semantic-similarity/README.md),
  and [NDCG](../retrieval-ranking/ndcg.md), whose built‑in aggregates only give a thresholded
  Wilson.

## References & code

- Code: [`bca_bootstrap`, `paired_bootstrap`](../../../src/evalharness/statistics/core.py);
  `scipy.stats.bootstrap`.
- Guide: [§6.7](../../guide.md#67-statistics-package).
- Lineage: Efron (1987), BCa; Efron & Tibshirani (1993).
