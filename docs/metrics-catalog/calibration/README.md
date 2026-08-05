# Calibration & selective prediction

**Does the model know when it doesn't know?** Accuracy asks "is the answer right?"
Calibration asks a different, complementary question: "when the model says it's 90% sure, is
it actually right about 90% of the time?" A well‑calibrated model lets you *act* on its
confidence — abstain when unsure, route hard cases to a human, trade coverage for accuracy.

These are **helper functions, not registry metrics.** They live in
[`scoring/calibration.py`](../../../src/evalharness/scoring/calibration.py) and are invoked
by `evalctl calibrate` and from research code — you will not find them in
`MetricRegistry.defaults()`, and `runs rescore --metrics` cannot call them. They take two
aligned lists — `correct: list[bool]` and `confidence: list[float]` — not `Case`/`Generation`
objects.

> **Requirement: real confidence.** Every metric here needs a genuine confidence signal
> (token logprobs or elicited probabilities). If you don't have one, **skip this family** —
> do not invent confidences. See [`Requirement.LOGPROBS`](../../../src/evalharness/core/enums.py).

## The two entry points

| Function | Produces | Doc |
|----------|----------|-----|
| `calibration_metrics(correct, confidence, *, bins=15)` | Adaptive ECE, Brier, NLL, reliability table | [Reliability, ECE, Brier & NLL](reliability-ece-brier-nll.md) |
| ↳ (same call) | Risk–coverage curve, AURC, accuracy@80% coverage, ROC‑AUC, PR‑AUC | [Selective prediction & risk–coverage](risk-coverage-aurc.md) |
| `calibrate_threshold(labels, scores)` | Operating threshold, ROC‑AUC, PR‑AUC, dev F1 | [Threshold calibration](threshold-calibration.md) |

`calibration_metrics` returns **everything in one dict** — the split into two docs is
pedagogical (goodness‑of‑confidence vs. acting‑on‑confidence), not two code paths.

## Where these surface

- **`evalctl calibrate <inputs.jsonl>`** reads JSONL rows with `label` and `score`, runs
  `calibrate_threshold`, and prints ROC‑AUC / PR‑AUC / threshold / dev F1.
- **Research / release code** calls `calibration_metrics` directly on stored correctness +
  confidence.
- **ROC‑AUC / PR‑AUC** appear here (and in [semantic similarity](../semantic-similarity/README.md))
  because they need a *score*, which is why they're not in the hard‑label
  [classification](../classification/README.md) metric.

## Pitfalls that apply to the whole family

- **Bin sensitivity.** ECE depends on the binning scheme; this code uses **equal‑mass
  (adaptive) bins**, default 15. There is **no fixed‑width / quantile ECE variant** in the
  code — don't cite one. Report the bin count.
- **Fit thresholds on *dev*, never holdout.** `calibrate_threshold` is a *fitting* routine;
  applying its threshold to the same data it was tuned on inflates everything.
- **AUCs need both classes.** ROC‑AUC returns `None` when only one class is present; PR‑AUC
  needs at least one positive.

## Related

- Guide: [§6.3](../../guide.md#63-calibration-helpers-not-registry-metrics).
- Narrative: [`metrics.md`](../../metrics.md#calibration).
- Test: [`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
  (`test_power_and_calibration`).
