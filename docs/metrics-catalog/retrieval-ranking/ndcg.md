# NDCG@10 (the primary value of `retrieval_ndcg_10`)

## TL;DR

A single number in `[0, 1]` for ranking quality that (a) gives **more credit for more
relevant** documents and (b) **discounts** documents the further down the list they appear.
`1.0` means your ranking is as good as the ideal ordering.

## What it measures & why you'd use it

Precision@k treats all relevant docs equally and ignores their order within the top‑k. But
a search result is better if the *most* relevant doc is *first*. **NDCG** (Normalized
Discounted Cumulative Gain) captures both: each document contributes a **gain** based on its
graded relevance, discounted by a logarithm of its rank, and the total is normalized by the
best possible ordering. It is the standard headline for graded retrieval.

## Intuition (tiny worked example)

`qrels = {d1: 3, d2: 1}`, ranking `["d1", "d2"]` (from the test). d1 (gain 3) is first, d2
(gain 1) second — this is already the ideal order, so DCG = IDCG and **NDCG = 1.0**. The
test also checks `precision@5 = 0.4` (2 relevant in top 5) and `recall@5 = 1.0` (both
relevant retrieved).

Now swap to `["d2", "d1"]`: the high‑gain doc is discounted more, DCG < IDCG, NDCG < 1 — the
metric punishes burying the best result.

## Formal definition

With exponential gain and log₂ rank discount over the top 10:

\[
\mathrm{DCG} = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \qquad
\mathrm{NDCG@10} = \frac{\mathrm{DCG}}{\mathrm{IDCG}},
\]

where IDCG is the DCG of the ideal ranking (relevant docs sorted by descending gain).
`recall_ceiling = min(1, |ranking| / |relevant|)` reports whether the list was even long
enough to retrieve everything relevant.

```310:317:src/evalharness/scoring/catalog.py
        dcg = sum(
            (2 ** relevant.get(doc, 0) - 1) / math.log2(index + 2)
            for index, doc in enumerate(ranking[:10])
        )
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        ndcg = dcg / idcg if idcg else 0.0
        detail["recall_ceiling"] = min(1.0, len(ranking) / len(relevant))
```

Note `log2(index + 2)` with 0‑based `index` = \(\log_2(\text{rank}+1)\) — standard.

## Inputs & requirements

- **Task types:** `retrieval`, `rag`. **Requires:** `QRELS`.
- **Reads:** `gen.output` (ranked ids), `case.qrels` (`{doc: graded gain}`).
- **Zero relevance** → `NULL` (`excluded: zero_relevance`).

## Output & aggregation

- **Per‑case value:** NDCG@10 ∈ `[0, 1]` (the metric's primary value); all other ranking
  numbers ride in `detail`.
- **Aggregate:** inherited **mean + Wilson**, but `config = {"threshold": 0.0, "cutoffs":
  [1,3,5,10,20]}` — with threshold `0.0`, *every* non‑null case counts as a "pass", so the
  Wilson interval here is essentially over coverage, not quality. **Report the mean NDCG (and
  a [BCa CI](../statistics/bootstrap.md) on it), not the pass rate.**
- **High vs low:** higher is better; `1.0` = ideal ordering of the retrieved relevant docs.

## Registered name / version & config

- **Name / version:** `retrieval_ndcg_10` / `1.0.0`.
- **Config:** `{"threshold": 0.0, "cutoffs": [1, 3, 5, 10, 20]}`. NDCG is fixed at cutoff
  **10** in code even though the P/R/Hit cutoffs are configurable via `cutoffs`.

## Pitfalls & gotchas

- **Exponential gain, always state it.** \((2^{rel}-1)\) ≠ linear gain \((rel)\). Two teams
  reporting "NDCG@10" with different gain formulas are not comparable. Put "exponential gain"
  in the report.
- **NDCG@10 ignores docs past rank 10** even though P/R/Hit are computed to 20. A great doc
  at rank 12 helps recall@20 but not NDCG@10.
- **Aggregate pass rate is misleading** because threshold is 0 — use the mean value with a
  bootstrap CI, not `passed`.
- **Ties = model order.** No re‑sorting; if the model emits a poor order among equally
  plausible docs, NDCG reflects that order.

## How it composes

- Headline for ranking; read with [P/R/Hit@k](precision-recall-hit-at-k.md) (set accuracy)
  and [MRR/MAP](mrr-map.md) (first‑hit / average precision) from the same payload.
- For RAG, score the *retrieval* with this and the *generated answer* with an overlap or
  semantic metric — two different questions.

## References & code

- Code: [`RetrievalMetric`](../../../src/evalharness/scoring/catalog.py).
- Test: `test_retrieval_ndcg_and_short_list_semantics` (asserts NDCG `1.0`, P@5 `0.4`, R@5
  `1.0`).
- Guide: [§6.4](../../guide.md#64-ranking--retrieval).
- Lineage: Järvelin & Kekäläinen (2002).
