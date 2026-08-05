# `meteor`

## TL;DR

An overlap metric that **aligns** output and reference words allowing synonyms and stems
(via WordNet), then scores the alignment — usually correlating with humans a bit better than
BLEU. Value in `[0, 1]`, or **`NULL`** if the NLTK resources aren't installed.

## What it measures & why you'd use it

METEOR improves on pure n‑gram overlap by building an explicit word **alignment** that
matches not just exact words but stems and WordNet synonyms, then combines unigram
precision/recall (recall‑weighted) with a fragmentation penalty for word‑order differences.
Reach for it when you want an overlap metric that is more forgiving of synonymy than
BLEU/ROUGE, and you can afford the NLTK/WordNet dependency.

## Intuition (tiny worked example)

Reference `"the quick brown fox"`, output `"the fast brown fox"`. BLEU/ROUGE penalize `fast`
vs `quick` fully; METEOR aligns them as WordNet synonyms and gives near‑full credit, docking
only a little for the reordering/fragmentation penalty.

## Formal definition

METEOR = \(F_{\text{mean}} \times (1 - \text{penalty})\), where \(F_{\text{mean}}\) is a
recall‑weighted harmonic mean of unigram precision/recall over the alignment, and the penalty
grows with the number of contiguous "chunks" in the alignment. The metric wraps NLTK's
`meteor_score` on casefolded `\w+` tokens:

```453:464:src/evalharness/scoring/catalog.py
        try:
            from nltk.translate.meteor_score import meteor_score

            value = float(meteor_score([_tokens(reference)], _tokens(gen.output)))
            return value, {"language": "en", "resources": ["wordnet"]}
        except (ImportError, LookupError) as exc:
            return None, {
                "reason": "nltk_resource_unavailable",
                "language": "en",
                "resource": "wordnet",
                "error": str(exc),
            }
```

## Inputs & requirements

- **Task types:** `ALL_TEXT_TASKS`. **Requires:** `REFERENCE`.
- **Also requires NLTK + WordNet data at runtime.** If `nltk` isn't importable or the WordNet
  corpus isn't downloaded, the metric returns **`NULL`** with `detail.reason =
  "nltk_resource_unavailable"` — an honest "couldn't measure," not a `0`.
- Missing output/reference → `NULL` (`reason: missing`).

## Output & aggregation

- **Per‑case value:** METEOR ∈ `[0, 1]` or `NULL`; `detail` records `language` and
  `resources`.
- **Aggregate:** inherited **mean + Wilson** over the `0.5` threshold, `NULL`s dropped.
- **High vs low:** higher is better.

## Registered name / version & config

- **Name / version:** `meteor` / `1.0.0`.
- **Config:** `{"language": "en", "resources": ["wordnet"]}` — declares the language and the
  NLTK resource the score depends on. **Install the WordNet data in CI** or the metric
  silently no‑ops to `NULL` across the run.

## Pitfalls & gotchas

- **Silent NULLs if resources are missing.** A run can show METEOR "scored" but with every
  value `NULL` — check `n` and the `detail.reason`. This is a deliberate, honest failure mode
  (see the deterministic ROUGE fallback for the alternative philosophy).
- **English/WordNet‑centric.** Synonymy is only as good as WordNet coverage; non‑English or
  domain jargon gets less benefit.
- **Overlap metric still.** More synonym‑aware than BLEU, but not a meaning verdict — for
  that use [semantic similarity](../semantic-similarity/README.md) or
  [BERTScore](bertscore.md).
- **Tokenization is casefolded `\w+`**, not METEOR's original tokenizer — minor differences
  from reference implementations are expected.

## How it composes

- A synonym‑aware complement to [ROUGE](rouge.md)/[SacreBLEU](sacrebleu.md); report it when
  paraphrase tolerance matters but you still want a lexical‑family metric.

## References & code

- Code: [`MeteorMetric`](../../../src/evalharness/scoring/catalog.py); NLTK `meteor_score`.
- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap).
- Lineage: Banerjee & Lavie (2005), "METEOR."
