# Effect sizes: Cohen's h & deltas

## TL;DR

A p‑value tells you a difference is *real*; an **effect size** tells you if it's *big enough
to care about*. `effect_sizes` reports the absolute delta, the relative delta, and **Cohen's
h** — a variance‑stabilized effect size for comparing two proportions.

## What it measures & why you'd use it

With enough data, a 0.1% improvement can be "statistically significant" and completely
irrelevant. Effect sizes restore proportion. For two rates \(p_0\) (baseline) and \(p_1\)
(candidate), the raw delta \(p_1 - p_0\) is intuitive but its *meaning* changes with the base
rate (going 0.50→0.55 is easy; 0.94→0.99 is hard). **Cohen's h** applies an arcsine
transform so a given h means the same "amount" of change regardless of where you are on the
`[0, 1]` scale — which is exactly why the harness also uses it for
[power/sample size](power.md).

## Intuition (tiny worked example)

Baseline 0.50 → candidate 0.55. Absolute delta = **+0.05**, relative delta = **+10%**.
Cohen's h = \(2(\arcsin\sqrt{0.55} - \arcsin\sqrt{0.50}) \approx 0.10\) — a "small" effect by
Cohen's rough guide (0.2 small, 0.5 medium, 0.8 large). The same +0.05 from 0.94→0.99 yields
a *larger* h, because gains near the ceiling are harder and count for more.

## Formal definition

\[
\text{absolute} = p_1 - p_0, \quad
\text{relative} = \frac{p_1 - p_0}{p_0}, \quad
h = 2\big(\arcsin\sqrt{p_1} - \arcsin\sqrt{p_0}\big).
\]

```84:88:src/evalharness/statistics/core.py
def effect_sizes(baseline: float, candidate: float) -> dict[str, float | None]:
    absolute = candidate - baseline
    relative = absolute / baseline if baseline else None
    h = 2 * (math.asin(math.sqrt(candidate)) - math.asin(math.sqrt(baseline)))
    return {"absolute_delta": absolute, "relative_delta": relative, "cohens_h": h}
```

## Inputs & requirements

- **Signature:** `effect_sizes(baseline: float, candidate: float)` — two rates in `[0, 1]`.
- `relative_delta` is **`None`** when `baseline == 0` (division guarded).
- Consumed by [`compare_binary`](mcnemar.md), which surfaces `absolute_delta`,
  `relative_delta`, and `cohens_h` on every `ComparisonResult`.

## Output

- `absolute_delta` — signed rate difference.
- `relative_delta` — fractional change (or `None`).
- `cohens_h` — arcsine effect size; sign follows candidate − baseline.

## Pitfalls & gotchas

- **Report an effect size beside every p‑value.** "Significant" without magnitude invites
  over‑claiming small wins — the whole reason this is bundled into `ComparisonResult`.
- **Relative delta explodes near zero baseline.** A jump 0.001→0.002 is "+100%" but trivial;
  prefer absolute delta / Cohen's h when the base rate is tiny, and note `None` when baseline
  is 0.
- **Cohen's thresholds are heuristics.** 0.2/0.5/0.8 are rules of thumb, not physics — pair
  with domain judgment.
- **h uses rates, not per‑case values.** It's defined on the two proportions; the paired‑case
  uncertainty comes from the [bootstrap CI](bootstrap.md), not from h.

## How it composes

- The "so what?" beside [McNemar](mcnemar.md)'s p‑value and [BH](benjamini-hochberg.md)'s
  significance flag.
- The **same arcsine machinery** powers [required sample size](power.md) — effect size and
  power are two views of one transform.

## References & code

- Code: [`effect_sizes`](../../../src/evalharness/statistics/core.py).
- Guide: [§6.7](../../guide.md#67-statistics-package).
- Lineage: Cohen (1988), *Statistical Power Analysis*.
