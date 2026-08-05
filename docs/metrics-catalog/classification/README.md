# Classification metrics

**One registered metric, a whole report.** `classification` scores a single label per case
with plain equality, but its real value is in the **aggregate**, which turns those per‑case
booleans into the standard classification battery: accuracy, balanced accuracy,
macro/micro/weighted precision/recall/F1, Matthews correlation coefficient (MCC), and
Cohen's κ — all in one pass, using scikit‑learn.

This family has a single metric, so this README *is* the in‑depth metric doc.

- **Code:** [`ClassificationMetric`](../../../src/evalharness/scoring/catalog.py)
- **Related:** [ROC‑AUC / PR‑AUC](../calibration/README.md) live in the calibration family
  (they need scores/confidences, not hard labels).

---

## `classification`

### 1. TL;DR

Per case: did the predicted label equal the expected label? Across the run: the full
imbalance‑aware scorecard, headlined by accuracy with a Wilson interval, with MCC and
balanced accuracy in the detail so a majority‑class guesser can't hide.

### 2. What it measures & why you'd use it

Accuracy alone lies under class imbalance: if 95% of cases are "negative", a model that
always says "negative" scores 95% and learns nothing. Practitioners therefore report
**balanced accuracy** (average recall across classes), **MCC** (a correlation coefficient
that is only high when the model does well on *all* classes), and **macro F1** (unweighted
mean over classes). `classification` computes all of them so the headline number can't be
gamed.

### 3. Intuition (tiny worked example)

Four cases, labels/predictions:

| case | expected | predicted | correct? |
|------|----------|-----------|----------|
| 1 | `spam` | `spam` | ✓ |
| 2 | `ham` | `ham` | ✓ |
| 3 | `ham` | `spam` | ✗ |
| 4 | `ham` | `ham` | ✓ |

Accuracy = 3/4 = **0.75**. But the model missed 1 of 3 `ham` and got the only `spam`, so
balanced accuracy = mean(recall_spam=1.0, recall_ham=2/3) ≈ **0.83**, and MCC is well below
1 because the confusion is asymmetric. Reporting only accuracy would hide that the model
over‑predicts `spam`.

### 4. Formal definitions

Per case: \(\text{value} = \mathbb{1}[\text{pred} = \text{expected}]\) (after `.strip()`).

Aggregate detail (all via scikit‑learn over the collected `(expected, predicted)` pairs):

- **Accuracy** = `accuracy_score` = fraction correct → the `AggregateValue.value`.
- **Balanced accuracy** = `balanced_accuracy_score` = mean per‑class recall.
- **Macro F1** = `f1_score(average="macro")` = unweighted mean of per‑class F1.
- **Micro F1** = `f1_score(average="micro")` = global TP/FP/FN pooled (equals accuracy for
  single‑label).
- **Weighted P/R/F1** = `precision_recall_fscore_support(average="weighted",
  zero_division=0)` = support‑weighted means.
- **MCC** = `matthews_corrcoef` — the \(\phi\) coefficient generalized to multiclass;
  \([-1, 1]\), 0 = chance.
- **Cohen's κ** = `cohen_kappa_score` — agreement corrected for chance.

```242:261:src/evalharness/scoring/catalog.py
        accuracy = float(accuracy_score(expected, predicted)) if expected else 0.0
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                expected, predicted, average="weighted", zero_division=0
            )
            ...
        detail = {
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)) ...
            "macro_f1": float(f1_score(expected, predicted, average="macro")) ...
            "micro_f1": float(f1_score(expected, predicted, average="micro")) ...
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "mcc": float(matthews_corrcoef(expected, predicted)) ...
            "cohen_kappa": float(cohen_kappa_score(expected, predicted)) ...
        }
```

### 5. Inputs & requirements

- **Task types:** `classification` only.
- **Requires:** nothing declared (`requires = frozenset()`), but a case with no
  `expected_label` (or a missing output) scores `NULL` and is dropped from the aggregate.
- **Reads:** `gen.output` (stripped) and `case.expected_label`
  ([`core/models.py`](../../../src/evalharness/core/models.py)).

### 6. Output & aggregation

- **Per‑case value:** `1.0` / `0.0` / `NULL`; `detail` has the `predicted` and `expected`
  strings (these are what the aggregate re‑reads).
- **Aggregate value:** **accuracy**, with a **Wilson 95% CI** over the count of exact
  matches; `AggregateValue.method` is a **JSON string** encoding the full detail dict above
  (balanced accuracy, macro/micro/weighted P/R/F1, MCC, κ). So the rich scorecard travels in
  `method`, not in separate columns.
- **High vs low:** higher is better across every sub‑metric; MCC and κ can go negative
  (worse than chance).

### 7. Registered name / version & config

- **Name / version:** `classification` / `1.0.0`.
- **Config:** `{"threshold": 1.0}` (exact label equality to "pass").

### 8. Pitfalls & gotchas

- **Free‑form chatter fails exact match.** `"The label is: spam"` ≠ `spam`. Constrain
  decoding (enum / classification head) or post‑parse the label before scoring, or accuracy
  understates the model.
- **Report MCC / balanced accuracy, not just accuracy** under imbalance — that's the whole
  point of the metric. A high accuracy with a low MCC is a majority‑class guesser.
- **No confusion matrix object is stored** — the aggregate captures the summary stats, not
  the full matrix. If you need the matrix, recompute from the per‑case `detail` rows (SQL
  query 8 in [guide §5.5](../../guide.md#55-query-library) pulls them) or from
  `predicted`/`expected`.
- **Micro F1 == accuracy** for single‑label multiclass — don't report both as if
  independent.
- **AUCs are not here.** ROC‑AUC / PR‑AUC need a *score*, not a hard label; they live in the
  [calibration family](../calibration/README.md) and `calibrate_threshold`.

### 9. How it composes

- On label tasks it is both the **gate** and the **quality metric**; layer
  [calibration](../calibration/README.md) on top when the model emits confidences and you
  care whether they're trustworthy.
- For A/B comparison of two classifiers, feed per‑case correctness booleans into
  [`compare_binary`](../statistics/mcnemar.md) (paired McNemar + BCa) and control FDR with
  [Benjamini–Hochberg](../statistics/benjamini-hochberg.md) across classes/slices.

### 10. References & code

- Code: [`ClassificationMetric`](../../../src/evalharness/scoring/catalog.py); scikit‑learn
  `sklearn.metrics`.
- Guide: [§6.2](../../guide.md#62-classification). Narrative:
  [`metrics.md`](../../metrics.md#classification).
- Lineage: Matthews (1975) for MCC; Cohen (1960) for κ.
