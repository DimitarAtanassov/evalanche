# Selective prediction: risk–coverage, AURC & accuracy@coverage

Also part of [`calibration_metrics`](../../../src/evalharness/scoring/calibration.py). These
outputs answer: **if the model abstains when unsure, how much better does it get?**

## TL;DR

Rank answers from most‑ to least‑confident. As you *cover* more of them (answer instead of
abstain), your error rate (*risk*) rises. The risk–coverage curve draws that trade‑off;
**AURC** summarizes it (lower is better); **accuracy@80% coverage** reads off one operating
point.

## What they measure & why

In production you rarely have to answer everything. If the model can rank its own answers by
confidence, you can answer the confident 80% and route the rest to a human — buying accuracy
by giving up coverage. Selective‑prediction metrics quantify exactly how favorable that
trade is, and let you compare two models on *usable* accuracy rather than blanket accuracy.

## Intuition (tiny worked example)

`correct = [True, False, True]`, `confidence = [0.9, 0.2, 0.7]`. Ranked by confidence:
`0.9(✓), 0.7(✓), 0.2(✗)`. Cumulative error and coverage:

| covered | included | errors so far | coverage | risk |
|---------|----------|---------------|----------|------|
| 1 | 0.9 ✓ | 0 | 1/3 | 0.00 |
| 2 | +0.7 ✓ | 0 | 2/3 | 0.00 |
| 3 | +0.2 ✗ | 1 | 3/3 | 0.33 |

Risk stays 0 until the low‑confidence wrong answer is included — exactly the behavior you
want from a usable confidence signal. AURC is the area under this risk‑vs‑coverage curve
(small here); accuracy@80% coverage reads the risk at the 80th‑percentile coverage index.

## Formal definitions

Rank by descending confidence. With cumulative errors \(E_k\) over the top \(k\):

\[
\text{coverage}_k = \frac{k}{n}, \qquad \text{risk}_k = \frac{E_k}{k}, \qquad
\text{AURC} = \int \text{risk}\,\, d\,\text{coverage}\ (\text{trapezoid}).
\]

Accuracy@80% coverage \(= 1 - \text{risk}_{k^*}\) where \(k^* = \lceil 0.8n \rceil - 1\).

```38:52:src/evalharness/scoring/calibration.py
    ranked = np.argsort(-p)
    cumulative_errors = np.cumsum(1 - y[ranked])
    coverage = np.arange(1, len(y) + 1) / len(y)
    risk = cumulative_errors / np.arange(1, len(y) + 1)
    idx_80 = max(0, math.ceil(0.8 * len(y)) - 1)
    return {
        ...
        "risk_coverage": [
            {"coverage": float(c), "risk": float(r)} for c, r in zip(coverage, risk, strict=True)
        ],
        "aurc": float(np.trapezoid(risk, coverage)),
        "accuracy_at_80_coverage": float(1 - risk[idx_80]),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(correct)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if any(correct) else None,
    }
```

## Inputs & requirements

Same call and preconditions as [reliability/ECE](reliability-ece-brier-nll.md): aligned
non‑empty `correct` + `confidence`. Confidence must be a real ranking signal — the whole
method assumes the model can order its own answers by trustworthiness.

## Output

- `risk_coverage` — list of `{coverage, risk}` points; plot to see the trade‑off.
- `aurc` — **lower is better**; a perfect confidence ranking that surfaces all correct
  answers first has near‑zero AURC.
- `accuracy_at_80_coverage` ∈ `[0, 1]` — accuracy if you answer the most‑confident 80%.
- `roc_auc` / `pr_auc` — discrimination of confidence vs correctness (`None` if a class is
  absent / no positives). Higher is better.

## Pitfalls & gotchas

- **AURC rewards *ranking*, not calibration.** A model can have great AURC (good ordering)
  yet be badly miscalibrated (wrong absolute probabilities) — read
  [ECE](reliability-ece-brier-nll.md) too.
- **Small n makes the curve jagged.** With few points the 80%‑coverage index snaps to a
  single sample; interpret operating points with the sample size in mind.
- **ROC‑AUC is not accuracy.** It measures separability of correct vs incorrect by
  confidence; a high AUC with low accuracy means "the model knows which answers are wrong,
  it just gets many wrong."
- **AUCs need both classes.** All‑correct or all‑wrong → `roc_auc = None`.

## How it composes

- The **action layer** on top of [ECE/Brier/NLL](reliability-ece-brier-nll.md): first check
  confidences are honest, then decide the coverage you'll ship.
- Set the actual cutoff with [`calibrate_threshold`](threshold-calibration.md) on dev data.

## References & code

- Code: [`calibration_metrics`](../../../src/evalharness/scoring/calibration.py).
- Guide: [§6.3](../../guide.md#63-calibration-helpers-not-registry-metrics).
- Lineage: El‑Yaniv & Wiener (2010) selective prediction; Geifman & El‑Yaniv (2017) for
  risk–coverage / AURC.
