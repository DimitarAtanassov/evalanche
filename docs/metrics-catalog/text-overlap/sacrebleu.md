# `sacrebleu`

## TL;DR

BLEU — the classic translation metric — measuring n‑gram **precision** with a brevity
penalty. Stored **per sentence** for inspection, but the number that matters is the **corpus
BLEU** computed on aggregate, with its SacreBLEU signature for reproducibility.

## What it measures & why you'd use it

BLEU counts how many of the output's n‑grams (up to 4‑grams) appear in the reference,
penalizing outputs that are too short. It is the long‑standing machine‑translation tripwire.
The critical design point here: BLEU is a **corpus‑level** statistic. The metric stores a
sentence BLEU per case (handy for spotting bad examples) but its `aggregate()` recomputes a
proper **corpus BLEU** — never the mean of sentence scores.

## Intuition (tiny worked example)

Two sentences, each with a 4‑gram or two matching the reference. Sentence BLEU on a single
short sentence is noisy (missing higher‑order n‑grams tank it). Corpus BLEU pools n‑gram
counts across *all* sentences before dividing, so a rare 4‑gram match in one sentence still
contributes — which is why the corpus number is both higher and more stable than the average
of the per‑sentence numbers. This is the whole reason not to average sentence BLEUs.

## Formal definition

BLEU = brevity penalty × geometric mean of modified n‑gram precisions \(p_1..p_4\):

\[
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\!\Big(\sum_{n=1}^{4} \tfrac{1}{4}\log p_n\Big), \qquad
\mathrm{BP} = \min\!\big(1, e^{1 - r/c}\big),
\]

with candidate length \(c\), reference length \(r\). The metric divides SacreBLEU's 0–100
score by 100 to land in `[0, 1]`.

```417:440:src/evalharness/scoring/catalog.py
    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [
            value for value in values if value.value is not None and "hypothesis" in value.detail
        ]
        metric = BLEU()
        score = (
            metric.corpus_score(
                [str(value.detail["hypothesis"]) for value in valid],
                [[str(value.detail["reference"]) for value in valid]],
            )
            if valid
            else None
        )
        return AggregateValue(
            ...
            f"corpus BLEU; {metric.get_signature()}" if score else "corpus BLEU",
        )
```

## Inputs & requirements

- **Task types:** `ALL_TEXT_TASKS`. **Requires:** `REFERENCE`.
- Per‑case `detail` stores `hypothesis` and `reference` strings **specifically so the
  aggregate can recompute corpus BLEU** — don't strip them.
- Missing output/reference → `NULL`.

## Output & aggregation

- **Per‑case value:** sentence BLEU / 100 ∈ `[0, 1]` (with `detail.sentence_score`).
- **Aggregate value:** **corpus BLEU / 100**, and `AggregateValue.method` = `"corpus BLEU;
  <SacreBLEU signature>"`. **No CI** (`ci_low/high = None`) — BLEU's corpus score isn't a
  mean of i.i.d. per‑case values, so a Wilson/mean CI wouldn't be meaningful.
- **High vs low:** higher is better.

## Registered name / version & config

- **Name / version:** `sacrebleu` / `1.0.0`.
- **Config:** none beyond inherited `threshold`. The **SacreBLEU signature** (tokenizer,
  smoothing, version) is recorded in the aggregate `method` string — quote it in reports so
  numbers are comparable across teams.

## Pitfalls & gotchas

- **Never average sentence BLEUs for the headline.** The mean of sentence BLEUs ≠ corpus
  BLEU and is systematically wrong. Use the aggregate.
- **Report the signature.** BLEU is only comparable under the same tokenization/smoothing —
  that's exactly the problem SacreBLEU's signature solves. A bare "BLEU 34" is not
  reproducible.
- **Short/segment‑level BLEU is noisy.** Sentence BLEU on short outputs is dominated by
  missing 4‑grams; use it to *find* bad cases, not to grade them.
- **Overlap ≠ adequacy.** BLEU misses meaning and can reward fluent mistranslations that
  echo reference n‑grams.

## How it composes

- Translation tripwire; pair with [chrF++](chrf.md) (better for morphologically rich
  languages) and meaning metrics for adequacy.
- Because its aggregate lacks a CI, if you need uncertainty on a BLEU delta, use a **paired
  bootstrap over sentence‑level scores** ([bootstrap](../statistics/bootstrap.md)) rather
  than the corpus point estimate alone.

## References & code

- Code: [`BleuMetric`](../../../src/evalharness/scoring/catalog.py); `sacrebleu`.
- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap).
- Lineage: Papineni et al. (2002) for BLEU; Post (2018), "A Call for Clarity in Reporting
  BLEU Scores" (SacreBLEU).
