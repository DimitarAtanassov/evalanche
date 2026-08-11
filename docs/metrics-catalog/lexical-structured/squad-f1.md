# `squad_f1`

## TL;DR

How much do the **words** of the answer and the reference overlap, balancing "did you
include the right words" (recall) against "did you avoid padding with wrong ones"
(precision)? A number in `[0, 1]`. It is exact match's forgiving cousin.

## What it measures & why you'd use it

`exact_match` is all‑or‑nothing. But `"the blue whale"` vs `"blue whale"` is *almost*
right, and a good QA metric should say so. `squad_f1` treats each answer as a **bag of
tokens** and computes the harmonic mean of token precision and recall. Reach for it on
extractive / short‑answer QA where the exact surface string varies but the content words
should overlap.

## Intuition (tiny worked example)

Reference `"the blue whale"` → tokens `{the, blue, whale}`. Output `"blue whale"` → tokens
`{blue, whale}`.

- Overlap (multiset intersection) = 2 tokens (`blue`, `whale`).
- Precision = overlap / |predicted| = 2/2 = **1.0** (nothing extra).
- Recall = overlap / |expected| = 2/3 ≈ **0.667** (missed `the`).
- \(F_1 = 2PR/(P+R) = 2(1)(0.667)/(1.667) = 0.8.\)

So the score is **0.8** — exactly the value asserted in
[`tests/test_metric_catalog.py`](../../../tests/test_metric_catalog.py) (`test_squad_metric_token_overlap`).

## Formal definition

Tokenize with `re.findall(r"\w+", text.casefold())` (case‑folded word characters). With
predicted multiset \(P\) and expected multiset \(E\), overlap \(o = \sum \min(P, E)\)
(multiset intersection):

\[
\text{precision} = \frac{o}{|P|}, \quad
\text{recall} = \frac{o}{|E|}, \quad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.
\]

Empty‑bag edge cases are handled explicitly: if `predicted` is empty, precision is
`float(not expected)`; symmetrically for recall — so two empty answers score `1.0`, and one
empty vs one non‑empty scores `0.0`.

```21:26:src/evalharness/scoring/metrics/lexical/squad_f1.py
        predicted, expected = tokens(gen.output), tokens(reference)
        common = Counter(predicted) & Counter(expected)
        overlap = sum(common.values())
        precision = overlap / len(predicted) if predicted else float(not expected)
        recall = overlap / len(expected) if expected else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

## Inputs & requirements

- **Task types:** `generation`, `qa_short`, `summarization`, `rag` (the `ALL_TEXT_TASKS`
  set).
- **Requires:** `REFERENCE` (via `_reference(case)` → `reference_answer` or
  `references[0]`).
- Missing output/reference → `value=None`, `detail={"reason": "missing"}`.

## Output & aggregation

- **Per‑case value:** \(F_1 \in [0, 1]\); `detail` carries `precision`, `recall`, `f1`.
- **Aggregate:** inherited `ScalarMetric` behavior — **mean + Wilson** over thresholded
  successes, `method="mean+wilson"`. Default pass threshold **`0.5`**: a case "passes" when
  \(F_1 \ge 0.5\), and the Wilson interval is over those pass counts, *not* over the mean
  \(F_1\) itself.
- **High vs low:** higher is better; `1.0` is identical token bags.

## Registered name / version & config

- **Name / version:** `squad_f1` / `1.0.0`.
- **Config:** none beyond the inherited `threshold` (default `0.5`) used for `passed` and
  the aggregate.

## Pitfalls & gotchas

- **Still lexical.** Synonyms with no shared tokens score `0` — `"car"` vs `"automobile"`
  gets no credit. For meaning, use [semantic similarity](../semantic-similarity/README.md).
- **Token bag, not order.** `"dog bites man"` and `"man bites dog"` score identically. If
  order matters, add [`rouge_l`](../text-overlap/rouge.md) (LCS‑based) or an exact gate.
- **Threshold vs mean.** The aggregate Wilson interval is over *passes at 0.5*, not over
  the continuous mean; if you want a CI on the mean \(F_1\), bootstrap it
  ([BCa](../statistics/bootstrap.md)).
- **Stopwords count.** Because articles aren't stripped here (unlike `exact_match`), short
  references are dominated by function words; interpret low‑token cases cautiously.

## How it composes

- The **partial‑credit layer** behind an `exact_match` gate on QA.
- Complements [`normalized_levenshtein`](normalized-levenshtein.md) (character‑level) —
  token‑F1 rewards word overlap, edit distance rewards near‑copies.
- For a mean with an interval, wrap the per‑case \(F_1\) list in
  [`bca_bootstrap`](../statistics/bootstrap.md).

## References & code

- Code: [`SquadMetric`](../../../src/evalharness/scoring/metrics/lexical/squad_f1.py).
- Test: `test_squad_metric_token_overlap` in
  [`tests/test_metric_catalog.py`](../../../tests/test_metric_catalog.py) (asserts `0.8`).
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
- Lineage: token‑level F1 from the SQuAD reading‑comprehension benchmark (Rajpurkar et al.,
  2016).
