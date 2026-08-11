# `rouge_l`

## TL;DR

How much of the reference's wording does the summary recover — measured by the **longest
common subsequence** of words (ROUGE‑L) plus n‑gram overlap (ROUGE‑1/2)? The registered value
is the ROUGE‑L F‑measure in `[0, 1]`.

## What it measures & why you'd use it

ROUGE is the default summarization tripwire. ROUGE‑1/2 count overlapping unigrams/bigrams;
**ROUGE‑L** uses the longest common subsequence, so it rewards in‑order word overlap without
requiring contiguity — a summary that keeps the reference's key phrases in order scores well
even if it inserts words between them. `rouge_l` computes all four (1, 2, L, Lsum) and
surfaces ROUGE‑L's F‑measure as the headline.

## Intuition (tiny worked example)

Reference `"the cat sat on the mat"`, output `"the cat sat"`. The LCS is `the cat sat`
(length 3). ROUGE‑L recall = 3/6 = 0.5, precision = 3/3 = 1.0, F ≈ 0.667. A short but
faithful prefix scores high on precision, lower on recall — ROUGE‑L's F balances the two.

## Formal definition

For ROUGE‑N, overlap of n‑gram multisets gives precision/recall/F. For ROUGE‑L, with LCS
length \(\ell\) between prediction (len \(m\)) and reference (len \(n\)):

\[
P = \frac{\ell}{m}, \quad R = \frac{\ell}{n}, \quad F = \frac{2PR}{P+R}.
\]

**Two implementations, one contract.** When the `rouge_score` library imports, it is used
directly (ROUGE‑1/2/L/Lsum). If it can't import (e.g. Python 3.14 blocking NLTK's regex
load), a **deterministic in‑house fallback** computes ROUGE‑1/2 via n‑gram counters and
ROUGE‑L via an LCS DP — preserving reportability. Either way the value is
`scores["rougeL"].fmeasure`.

```python
            return float(scores["rougeL"].fmeasure), detail
        except ImportError:
            # Python 3.14 can block NLTK's regex import from unsafe paths.
            # The deterministic implementation below preserves reportability.
            pass
        predicted = tokens(gen.output)
        expected = tokens(reference)
        ...
        lcs = _lcs_length(predicted, expected)
        fallback_detail["rougeL"] = _prf(lcs, len(predicted), len(expected))
        fallback_detail["rougeLsum"] = fallback_detail["rougeL"]
        return fallback_detail["rougeL"]["fmeasure"], fallback_detail
```

## Inputs & requirements

- **Task types:** `ALL_TEXT_TASKS`. **Requires:** `REFERENCE`.
- Missing output/reference → `NULL`.

## Output & aggregation

- **Per‑case value:** ROUGE‑L F ∈ `[0, 1]`; `detail` carries precision/recall/fmeasure for
  each of `rouge1`, `rouge2`, `rougeL`, `rougeLsum`.
- **Aggregate:** inherited **mean + Wilson** over the `0.5` threshold. For a mean ROUGE with
  a CI, [BCa bootstrap](../statistics/bootstrap.md) the per‑case values.

## Registered name / version & config

- **Name / version:** `rouge_l` / `1.0.0`. **Config:** inherited `threshold` (`0.5`).

## Pitfalls & gotchas

- **Overlap ≠ quality.** ROUGE rewards lexical echo; an abstractive summary that paraphrases
  well can score low. Regression tripwire, not verdict.
- **Fallback ≈ but ≠ library.** The deterministic fallback casefolds via `\w+` tokens and
  sets `rougeLsum = rougeL`; the `rouge_score` library tokenizes/stems slightly differently.
  Numbers can differ marginally across environments — pin the environment for published
  comparisons and note which path ran (the `detail` shape differs).
- **ROUGE‑L ≠ ROUGE‑Lsum in general.** The library computes Lsum (sentence‑split) distinctly;
  the fallback equates them.
- **Recall bias.** ROUGE historically emphasizes recall; a verbose output can inflate recall
  overlap — read precision too.

## How it composes

- The summarization tripwire; layer [chrF++](chrf.md) for morphology, [METEOR](meteor.md)
  for synonym‑aware alignment, and [semantic similarity](../semantic-similarity/README.md)
  or [BERTScore](bertscore.md) for meaning.

## References & code

- Code: [`RougeLMetric`](../../../src/evalharness/scoring/metrics/overlap/rouge_l.py), `_lcs_length`,
  `_prf`.
- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap).
- Lineage: Lin (2004), "ROUGE: A Package for Automatic Evaluation of Summaries."
