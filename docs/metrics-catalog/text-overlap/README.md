# Text overlap (summarization / translation)

**How much surface content does the generation share with the reference?** These metrics —
ROUGE, BLEU, chrF++, METEOR, and the optional BERTScore — count overlapping n‑grams,
characters, or aligned words between output and reference. They are the standard tripwires
for summarization and translation regressions.

**Read this warning first, then the metrics.** Surface‑overlap metrics **correlate weakly
with human judgment on abstractive tasks.** A faithful paraphrase that shares few words
scores low; a fluent‑but‑wrong copy that echoes the reference scores high. Treat them as
**regression detectors** ("did quality move?"), not as **quality verdicts** ("is it good?").
For meaning, cross over to [semantic similarity](../semantic-similarity/README.md); for
faithfulness you ultimately need human or LLM judgment (the latter is
[deferred](../../guide.md#84-known-gaps--deferred)).

All live in [`scoring/metrics/overlap/`](../../../src/evalharness/scoring/metrics/overlap/) except
BERTScore, which is in [`scoring/ml.py`](../../../src/evalharness/scoring/ml.py) behind the
`metrics-ml` extra.

## The metrics

| Doc | Registered name | Level | One line | Availability |
|-----|-----------------|-------|----------|--------------|
| [ROUGE](rouge.md) | `rouge_l` | word n‑gram / LCS | ROUGE‑1/2/L/Lsum; value is ROUGE‑L F | built‑in (with deterministic fallback) |
| [SacreBLEU](sacrebleu.md) | `sacrebleu` | word n‑gram (precision) | Sentence BLEU per case, **corpus BLEU** on aggregate | built‑in |
| [chrF++](chrf.md) | `chrf_pp` | character n‑gram (+word) | Robust to morphology / typos | built‑in |
| [METEOR](meteor.md) | `meteor` | aligned unigrams | Synonym/stem‑aware alignment | built‑in (NULL if NLTK data missing) |
| [BERTScore](bertscore.md) | `bertscore_f1` | contextual embeddings | Meaning‑aware token matching | **`metrics-ml` extra only** |

## Shared mechanics

- All require `REFERENCE` and apply to `ALL_TEXT_TASKS` = `generation`, `qa_short`,
  `summarization`, `rag`.
- Missing output/reference → `NULL` (`detail.reason = "missing"`); METEOR also returns
  `NULL` when its NLTK resources are unavailable.
- Except where noted (BLEU's corpus aggregate), they inherit the `ScalarMetric`
  **mean + Wilson** aggregate over the `0.5` threshold — so the aggregate "pass rate" is a
  thresholded proxy; for the *mean* score with an interval, use a
  [BCa bootstrap](../statistics/bootstrap.md).

## Pitfalls that apply to the whole family

- **Overlap ≠ quality.** The headline caveat above. Always say which metric and why.
- **BLEU corpus ≠ mean of sentence BLEUs.** The single most common reporting error — see
  [SacreBLEU](sacrebleu.md).
- **Report the signature/config.** chrF and BLEU results are only comparable with the same
  tokenization/signature; BERTScore only with the same pinned model + layers.

## Related

- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap).
- Narrative: [`metrics.md`](../../metrics.md#overlap-summarization--translation).
- [Semantic similarity](../semantic-similarity/README.md) — when you need meaning, not
  surface.
