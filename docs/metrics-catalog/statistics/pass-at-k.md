# pass@k

## TL;DR

If you let the model try \(k\) times and count it a success when **any** attempt works, what
is the expected success rate? `pass_at_k` is the **unbiased** HumanEval‑style estimator —
not the naive "best of k, score once."

## What it measures & why you'd use it

Code‑generation and agent tasks often allow multiple attempts: generate \(k\) samples, accept
if any passes the unit tests. Naively sampling \(k\) and reporting whether one passed is a
noisy, biased estimate of the true pass@k. The unbiased estimator uses \(n \ge k\) generated
samples, of which \(c\) passed, and computes the probability that a random size‑\(k\) subset
contains at least one success — averaging over all subsets analytically.

## Intuition (tiny worked example)

`pass_at_k(n=10, c=2, k=3)`: from 10 samples, 2 passed. The chance a random 3‑sample draw
misses **both** successes is \(\binom{8}{3}/\binom{10}{3}\); pass@3 is one minus that ≈
**0.5333**. That's the golden value in
[`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
(`pass_at_k(10, 2, 3) ≈ 0.5333`). Contrast the naive estimate (did the first 3 happen to
include a pass?), which is far noisier.

## Formal definition

\[
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1}\frac{n-c-i}{n-i}.
\]

Computed in log‑gamma space for numerical stability (and returns `1.0` when \(n-c < k\), i.e.
too few failures to avoid a success):

```96:107:src/evalharness/statistics/core.py
def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or k < 1:
        raise ValueError("Require 0 <= c <= n and k >= 1")
    if n - c < k:
        return 1.0
    log_failure = (
        math.lgamma(n - c + 1)
        - math.lgamma(n - c - k + 1)
        - math.lgamma(n + 1)
        + math.lgamma(n - k + 1)
    )
    return -math.expm1(log_failure)
```

## Inputs & requirements

- **Signature:** `pass_at_k(n: int, c: int, k: int)` — total samples \(n\), correct \(c\),
  budget \(k\).
- Requires `0 ≤ c ≤ n` and `k ≥ 1` (else `ValueError`).
- \(n - c < k\) → `1.0` (with so few failures, every size‑\(k\) draw contains a success).

## Output

- A probability in `[0, 1]` — the unbiased expected pass rate under a \(k\)‑attempt budget.

## Pitfalls & gotchas

- **Record \(n, c, k\).** pass@k is a *function* of the sampling budget; a number without its
  \(k\) (and the \(n\) it was estimated from) is uninterpretable. This is the single most
  common misuse.
- **Not "best of k then score once."** That collapses the estimator to a single noisy
  Bernoulli draw and throws away the variance reduction. Generate \(n \ge k\) and use the
  formula.
- **Needs \(n \ge k\).** You must actually sample at least \(k\); with \(n = k\) the estimate
  is valid but high‑variance — sample more where you can.
- **Per‑task, then average.** Compute pass@k per problem, then average across problems (the
  HumanEval convention); don't pool \(c\) and \(n\) across heterogeneous tasks.

## How it composes

- The right success metric for multi‑attempt code/agent tasks; combine per‑task pass@k with a
  [bootstrap CI](bootstrap.md) over tasks for uncertainty.
- Orthogonal to the pass‑rate [Wilson interval](wilson.md), which is for single‑attempt
  Bernoulli outcomes.

## References & code

- Code: [`pass_at_k`](../../../src/evalharness/statistics/core.py).
- Test: `test_wilson_and_pass_at_k_golden`.
- Guide: [§6.7](../../guide.md#67-statistics-package).
- Lineage: Chen et al. (2021), "Evaluating Large Language Models Trained on Code" (HumanEval).
