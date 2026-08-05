# Threshold calibration (`calibrate_threshold`)

## TL;DR

Given labeled examples with continuous scores, find the score cutoff that **maximizes F1 on
your development data**, and report how separable the two classes are (ROC‑AUC, PR‑AUC). This
is how you turn a similarity/confidence score into a yes/no decision — honestly.

## What it measures & why you'd use it

Any continuous score — a cosine similarity, a Levenshtein similarity, a model confidence —
needs a **threshold** to become a decision ("accept if ≥ τ"). Picking τ by eyeballing, or
reusing a magic `0.8`, is how teams accidentally overfit. `calibrate_threshold` sweeps the
precision–recall curve, picks the τ that maximizes F1 **on dev**, and hands back the AUCs so
you report separability alongside the operating point. It backs `evalctl calibrate`.

## Intuition (tiny worked example)

`labels = [True, True, False, False]`, `scores = [0.9, 0.8, 0.3, 0.1]`. The positives score
higher than the negatives, so a threshold anywhere in `(0.3, 0.8]` perfectly separates them:
dev F1 = **1.0**, ROC‑AUC = 1.0, PR‑AUC = 1.0. That's exactly the assertion in
[`tests/test_statistics_catalog.py`](../../../tests/test_statistics_catalog.py)
(`test_power_and_calibration`: `result["dev_f1"] == 1.0`).

## Formal definition

Compute the PR curve, form \(F_1 = 2PR/(P+R)\) at each threshold, and take the argmax over
all but the last point (the last PR point has no threshold):

\[
\tau^* = \arg\max_{t} F_1(t), \qquad \text{report } \tau^*,\ F_1(\tau^*),\ \text{ROC‑AUC},\ \text{PR‑AUC}.
\]

```58:69:src/evalharness/scoring/calibration.py
def calibrate_threshold(labels: list[bool], scores: list[float]) -> dict[str, float]:
    if len(labels) != len(scores) or len(set(labels)) < 2:
        raise ValueError("Calibration requires aligned positive and negative examples")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1[:-1]))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "threshold": float(thresholds[index]),
        "dev_f1": float(f1[index]),
    }
```

## Inputs & requirements

- **Signature:** `calibrate_threshold(labels: list[bool], scores: list[float])`.
- Requires **both classes present** and aligned lengths — otherwise `ValueError` ("requires
  aligned positive and negative examples").
- **CLI:** `evalctl calibrate <inputs.jsonl>` where each JSONL row carries a `label` and a
  `score`.

## Output

- `threshold` — the operating cutoff \(\tau^*\).
- `dev_f1` — F1 achieved at \(\tau^*\) **on the fitting data** (optimistic by construction).
- `roc_auc`, `pr_auc` — class separability, independent of the chosen threshold.

## Pitfalls & gotchas

- **Dev only — never fit on holdout.** `dev_f1` is an *in‑sample* optimum; reporting it as
  test performance is a classic leak. Fit τ on dev, then apply the *frozen* τ to holdout and
  report that number.
- **F1‑optimal ≠ your cost trade‑off.** Argmax‑F1 weights precision and recall equally. If
  false positives and false negatives cost differently, choose τ from the risk/coverage or
  cost curve instead.
- **Threshold indexing.** `precision_recall_curve` returns one fewer threshold than
  precision/recall points; the code correctly drops the last F1 entry (`f1[:-1]`) so the
  chosen index maps to a real threshold.
- **AUCs still need both classes** (guaranteed by the input check here).

## How it composes

- The decision‑making cap on the calibration family: after
  [ECE/Brier/NLL](reliability-ece-brier-nll.md) and
  [risk–coverage](risk-coverage-aurc.md), this picks the cutoff you'll ship.
- The **canonical way to threshold** [semantic cosine](../semantic-similarity/README.md) and
  [`normalized_levenshtein`](../lexical-structured/normalized-levenshtein.md) — report
  ROC‑AUC and the operating point, not a magic constant.

## References & code

- Code: [`calibrate_threshold`](../../../src/evalharness/scoring/calibration.py); CLI
  `evalctl calibrate`.
- Guide: [§6.3](../../guide.md#63-calibration-helpers-not-registry-metrics),
  [§4.7](../../guide.md#47-evalctl-calibrate).
