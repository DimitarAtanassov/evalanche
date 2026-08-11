# `normalized_levenshtein`

## TL;DR

How few single‑character edits turn the output into the reference, expressed as a
similarity in `[0, 1]` where `1.0` is identical? Great for near‑copies and OCR‑ish noise.

## What it measures & why you'd use it

Where [`squad_f1`](squad-f1.md) works at the token level, Levenshtein works at the
**character** level. It counts the minimum number of insertions, deletions, and
substitutions to convert one string into another (the *edit distance*), then normalizes by
length so the score is comparable across answer sizes. Reach for it when answers are
*almost* the reference: a dropped letter, a transposed digit, a stray character from OCR or
a flaky decode.

## Intuition (tiny worked example)

Output `"colour"`, reference `"color"`. One deletion (`u`) turns the first into the second,
so edit distance = 1 over a max length of 6 → similarity ≈ `1 − 1/6 = 0.833`. Above the
default `0.8` threshold, so `passed = True`. Drop to `"colouur"` (distance 2) → `1 − 2/7 ≈
0.714` → below threshold, `passed = False`.

## Formal definition

Let \(d(o, r)\) be the Levenshtein edit distance and \(L = \max(|o|, |r|)\). The metric
uses RapidFuzz's `normalized_similarity`:

\[
\text{sim}(o, r) = 1 - \frac{d(o, r)}{L} \in [0, 1].
\]

```20:27:src/evalharness/scoring/metrics/lexical/levenshtein.py
    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        value = float(Levenshtein.normalized_similarity(gen.output, reference))
        return value, {"threshold": self.threshold}
```

Note the comparison is on the **raw** strings (RapidFuzz), *not* the normalized ones —
casing and punctuation count as edits here.

## Inputs & requirements

- **Task types:** `generation`, `qa_short`, `summarization`, `rag`.
- **Requires:** `REFERENCE`.
- Missing output/reference → `value=None`, `detail={"reason": "missing"}`.

## Output & aggregation

- **Per‑case value:** similarity \(\in [0, 1]\); `detail` records the `threshold`.
- **Aggregate:** inherited **mean + Wilson** over thresholded successes; the pass threshold
  defaults to **`0.8`** (constructor `LevenshteinMetric(threshold=0.8)`), and that value is
  folded into `config` and thus the config hash.
- **High vs low:** higher is better; `1.0` is identical strings.

## Registered name / version & config

- **Name / version:** `normalized_levenshtein` / `1.0.0`.
- **Config:** `threshold` (default `0.8`) — the only knob; it sets `passed` and the
  aggregate pass count and enters `metric_config_sha256`.

## Pitfalls & gotchas

- **Calibrate the threshold on *dev*.** `0.8` is a default, not a law. Fit it with
  [`evalctl calibrate`](../calibration/threshold-calibration.md) and report the ROC‑AUC /
  operating point rather than shipping a magic number to holdout.
- **Length‑sensitive.** A single wrong character costs far more on a short answer than a
  long one; two answers with the same *number* of errors get different scores by length.
- **Case & punctuation are edits.** Unlike `exact_match`, there is no normalizer here — if
  you don't care about casing, either lowercase upstream or prefer `exact_match`.
- **Not semantic.** Character similarity says nothing about meaning; a plausible paraphrase
  can score low.

## How it composes

- Sits between `exact_match` (too strict) and `squad_f1` (token‑level) for **near‑copy**
  answers — e.g. transcription, code identifiers, IDs.
- Its threshold is a natural candidate for the same dev‑set calibration workflow as
  [semantic cosine](../semantic-similarity/README.md); reuse
  [`calibrate_threshold`](../calibration/threshold-calibration.md).

## References & code

- Code: [`LevenshteinMetric`](../../../src/evalharness/scoring/metrics/lexical/levenshtein.py); RapidFuzz
  `Levenshtein.normalized_similarity`.
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
- Lineage: Levenshtein (1966), edit distance.
