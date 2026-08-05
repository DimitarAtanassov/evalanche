# Reliability, ECE, Brier & NLL

Part of [`calibration_metrics`](../../../src/evalharness/scoring/calibration.py). These four
outputs answer: **how well do the model's confidences match reality?**

## TL;DR

- **Reliability table** — group predictions by confidence and compare each group's average
  confidence to its actual accuracy.
- **Adaptive ECE** — one number summarizing the gap in that table (0 = perfectly
  calibrated).
- **Brier** — mean squared error between confidence and correctness.
- **NLL** — how surprised the model's confidence is by the truth (log loss).

## What they measure & why

A model that is 90% confident should be right 90% of the time. The reliability table makes
this visible bin by bin; **ECE** collapses it to a scalar; **Brier** and **NLL** are proper
scoring rules that reward honest probabilities and punish both over‑ and under‑confidence,
with NLL punishing confident‑and‑wrong far more harshly than Brier.

## Intuition (tiny worked example)

`correct = [True, False, True]`, `confidence = [0.9, 0.2, 0.7]`, `bins = 3`. Sorted by
confidence, each bin has one item:

- bin @0.2: accuracy 0, |0−0.2| = 0.2
- bin @0.7: accuracy 1, |1−0.7| = 0.3
- bin @0.9: accuracy 1, |1−0.9| = 0.1

Equal weights (1/3 each) → ECE = (0.2 + 0.3 + 0.1)/3 = **0.2**. Brier = mean of
\((0.9-1)^2, (0.2-0)^2, (0.7-1)^2\) = (0.01 + 0.04 + 0.09)/3 ≈ **0.047**. The test only
asserts \(0 \le \text{ECE} \le 1\), which this satisfies.

## Formal definitions

Probabilities are clipped to \([10^{-12}, 1-10^{-12}]\). Bins are **equal‑mass**: sort by
\(p\), split indices into `min(bins, n)` groups.

- **Adaptive ECE:** \(\displaystyle \sum_{b} \frac{|b|}{n}\,\big|\text{acc}(b) - \overline{\text{conf}}(b)\big|.\)
- **Brier:** \(\displaystyle \frac{1}{n}\sum_i (p_i - y_i)^2.\)
- **NLL** (binary cross‑entropy): \(\displaystyle -\frac{1}{n}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big].\)

```22:46:src/evalharness/scoring/calibration.py
    order = np.argsort(p)
    groups = np.array_split(order, min(bins, len(order)))
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    for group in groups:
        accuracy = float(np.mean(y[group]))
        mean_confidence = float(np.mean(p[group]))
        weight = len(group) / len(y)
        ece += weight * abs(accuracy - mean_confidence)
        reliability.append(
            {"n": len(group), "accuracy": accuracy, "confidence": mean_confidence}
        )
    ...
    return {
        "adaptive_ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "reliability": reliability,
        ...
```

## Inputs & requirements

- **Signature:** `calibration_metrics(correct: list[bool], confidence: list[float], *,
  bins: int = 15)`.
- Lists must be **non‑empty and aligned** — otherwise `ValueError`.
- Confidence must be a real signal (logprobs / elicited). Not a `Metric`; not registered.

## Output

- `adaptive_ece` ∈ `[0, 1]` — **lower is better**; 0 = confidences match accuracy exactly.
- `brier` ∈ `[0, 1]` — lower is better.
- `nll` ≥ 0 — lower is better; unbounded above (a confident wrong answer explodes it).
- `reliability` — list of `{n, accuracy, confidence}` per bin; plot accuracy vs confidence
  against the diagonal.

## Pitfalls & gotchas

- **Bin count changes ECE.** Equal‑mass bins reduce (not eliminate) the sensitivity of
  fixed‑width ECE, but the number still moves with `bins`. Report it; compare like with
  like. There is **no fixed‑width ECE** in this code.
- **ECE hides direction.** A model can be over‑confident on one bin and under‑confident on
  another and still show a small ECE; read the reliability table, not just the scalar.
- **Brier/NLL mix calibration and refinement.** They reward both good calibration *and*
  good discrimination; a low Brier doesn't isolate calibration the way the reliability table
  does.
- **Clipping.** Probabilities exactly at 0/1 are clipped to avoid infinite NLL — extreme
  confidences are slightly tempered.

## How it composes

- Read alongside [risk–coverage / AURC](risk-coverage-aurc.md) from the *same* call: ECE
  tells you if confidences are honest; risk–coverage tells you how to *use* them.
- Calibrate first, then set an operating [threshold](threshold-calibration.md).

## References & code

- Code: [`calibration_metrics`](../../../src/evalharness/scoring/calibration.py).
- Guide: [§6.3](../../guide.md#63-calibration-helpers-not-registry-metrics).
- Lineage: Brier (1950); Naeini et al. (2015) for ECE; Guo et al. (2017) on modern NN
  calibration.
