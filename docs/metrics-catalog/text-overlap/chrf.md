# `chrf_pp`

## TL;DR

BLEU's character‑level cousin: measures overlap of **character** n‑grams (plus some word
n‑grams), which makes it robust to morphology, typos, and word‑order wiggle. Value in
`[0, 1]`.

## What it measures & why you'd use it

Word‑level metrics like BLEU are brittle for morphologically rich languages (German, Finnish,
Turkish) where one root inflects many ways — a correct translation with a different suffix
scores as a total miss. **chrF** works on character n‑grams, so it gives partial credit for
shared stems; the **++** variant (`word_order=2`) adds word unigram+bigram matching on top.
Reach for it whenever BLEU feels unfairly harsh on inflection or minor spelling variation.

## Intuition (tiny worked example)

Reference `"running"`, output `"runs"`. BLEU sees two different words → ~0. chrF sees shared
character 3‑grams (`run`) → a positive score. That partial credit for shared morphology is
the whole point.

## Formal definition

chrF is an F‑score over character n‑gram precision and recall (default up to 6‑grams),
β‑weighted toward recall; the ++ form adds word n‑grams via `word_order=2`. The metric uses
SacreBLEU's `sentence_chrf(..., word_order=2)` and divides the 0–100 score by 100:

```391:398:src/evalharness/scoring/catalog.py
    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        score = sacrebleu.sentence_chrf(gen.output, [reference], word_order=2)
        return score.score / 100, {"signature": "chrF2++", "raw_score": score.score}
```

## Inputs & requirements

- **Task types:** `ALL_TEXT_TASKS`. **Requires:** `REFERENCE`.
- Missing output/reference → `NULL`.

## Output & aggregation

- **Per‑case value:** chrF2++ / 100 ∈ `[0, 1]`; `detail` records `signature = "chrF2++"` and
  the raw 0–100 `raw_score`.
- **Aggregate:** inherited **mean + Wilson** over the `0.5` threshold. Unlike BLEU, chrF's
  aggregate here is the mean of per‑sentence chrF (not a special corpus recomputation), so a
  [BCa CI](../statistics/bootstrap.md) on the mean is appropriate for publishing.
- **High vs low:** higher is better.

## Registered name / version & config

- **Name / version:** `chrf_pp` / `1.0.0`.
- **Config:** none beyond inherited `threshold`; `word_order=2` is fixed (the "++"), giving
  the `chrF2++` signature.

## Pitfalls & gotchas

- **Sentence‑level here, unlike BLEU's corpus aggregate.** The mean of per‑sentence chrF is a
  reasonable summary, but state that it's a mean; don't imply a corpus chrF recomputation.
- **Character overlap can flatter gibberish** that happens to share substrings; still an
  overlap metric, still not a quality verdict.
- **Comparable only with matching signature.** Report `chrF2++`; a different `word_order` or
  n‑gram order isn't comparable.

## How it composes

- Best paired with [SacreBLEU](sacrebleu.md) on translation — chrF for morphology, BLEU for
  the familiar baseline — and with meaning metrics for adequacy.
- A solid default when the target language is morphologically rich.

## References & code

- Code: [`ChrfMetric`](../../../src/evalharness/scoring/catalog.py); `sacrebleu.sentence_chrf`.
- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap).
- Lineage: Popović (2015) chrF; (2017) chrF++.
