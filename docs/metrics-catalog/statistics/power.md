# Power & required sample size

## TL;DR

**How many cases do I need before I run?** Given a baseline rate and the smallest
improvement you care about (the MDE), `required_sample_size` tells you the per‑arm \(n\) that
gives you an 80% chance of detecting it at α = 0.05. Size the study *before* spending
inference.

## What it measures & why you'd use it

A common failure mode: run a small eval, see a "significant" 2% win, celebrate — then a
larger follow‑up fails to replicate. The root cause is usually underpowered design. **Power
analysis** flips the question: given the effect size you care about, *how big must the
dataset be?* `evalctl power` is the CLI front‑end; use it before collecting data, not after.

## Intuition (tiny worked example)

Baseline rate 50%, minimum detectable effect (MDE) +10 pp → candidate ≈ 60%. At α = 0.05
and power 0.8, `required_sample_size(0.5, 0.1)` returns a number **> 100** (asserted in
[`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)). That is
*per arm* — you need that many cases on baseline *and* on candidate. Smaller MDEs demand
much larger \(n\); a 1 pp effect at the same power needs thousands.

## Formal definition

Uses Cohen's h between \(p_0\) and \(p_0 + \text{MDE}\), then the two‑sample arcsine sample‑size
formula:

\[
h = \big|2\big(\arcsin\sqrt{p_1} - \arcsin\sqrt{p_0}\big)\big|, \qquad
n = \Big\lceil 2\Big(\frac{z_{1-\alpha/2} + z_{\text{power}}}{h}\Big)^2 \Big\rceil.
\]

```110:123:src/evalharness/statistics/core.py
def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    candidate = min(1 - 1e-9, max(1e-9, baseline_rate + minimum_detectable_effect))
    h = abs(2 * (math.asin(math.sqrt(candidate)) - math.asin(math.sqrt(baseline_rate))))
    if h == 0:
        raise ValueError("minimum_detectable_effect must be non-zero")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return int(math.ceil(float(2 * ((z_alpha + z_power) / h) ** 2)))
```

Rates are clipped away from `{0, 1}` so the arcsine is defined.

## Inputs & requirements

- **Signature:** `required_sample_size(baseline_rate, minimum_detectable_effect, *,
  alpha=0.05, power=0.8)`.
- **CLI:** `evalctl power --baseline-rate <float> --mde <float> [--power 0.8] [--alpha
  0.05]` → `{"sample_size_per_arm": …, "power": …}`.
- `minimum_detectable_effect` must be non‑zero (else `ValueError`).

## Output

- An integer — the **per‑arm** sample size. Double it for a two‑arm A/B.

## Pitfalls & gotchas

- **Per arm, not total.** The number is for each group; a two‑arm experiment needs \(2n\).
- **MDE is a *product* decision, not a statistic.** Pick the smallest improvement that would
  change your shipping decision — a tiny MDE forces an enormous \(n\), which is honest.
- **Assumes a simple two‑sided rate comparison.** It doesn't account for multiple testing
  (use a tighter α / apply [BH](benjamini-hochberg.md) later) or for continuous metrics
  (bootstrap power is a different calculation).
- **Don't use it to rationalize a small run after the fact.** Power analysis is a *design*
  tool; post‑hoc power ("we had 40% power") is nearly uninformative.

## How it composes

- The design‑time counterpart of [Cohen's h](effect-sizes.md) and the [comparison
  pipeline](mcnemar.md): decide the MDE with domain judgment, size with this, then run and
  compare with McNemar/BCa/BH.
- Complements [Wilson](wilson.md): Wilson tells you the CI after the fact; power tells you
  how wide that CI will be *before* you spend.

## References & code

- Code: [`required_sample_size`](../../../src/evalharness/statistics/core.py); CLI `evalctl
  power`.
- Guide: [§6.7](../../guide.md#67-statistics-package), [§4.6](../../guide.md#46-evalctl-power).
- Lineage: Cohen (1988) arcsine sample‑size formula.
