# Precision@k, Recall@k, Hit@k

Sub‑metrics inside `retrieval_ndcg_10`'s `detail`, computed for each cutoff
\(k \in \{1, 3, 5, 10, 20\}\).

## TL;DR

- **Precision@k** — of the top *k* results, what fraction are relevant?
- **Recall@k** — of all relevant docs, what fraction made it into the top *k*?
- **Hit@k** — did *any* relevant doc appear in the top *k*? (1/0)

## What they measure & why

These are the intuitive, **set‑based** ranking numbers — they ignore order *within* the
top‑k and just ask about membership. Precision matters when the user only looks at the first
few results; recall matters when missing a relevant doc is costly (RAG context); Hit@k is the
"did we get at least one useful result" signal, common for QA‑over‑retrieval.

## Intuition (tiny worked example)

`qrels = {d1: 3, d2: 1}` → relevant = `{d1, d2}`. Ranking `["d1", "d2"]`:

- **@5:** top‑5 is just `[d1, d2]` (only 2 exist). hits = 2. Precision@5 = 2/5 = **0.4**;
  Recall@5 = 2/2 = **1.0**; Hit@5 = **1.0**.
- **@1:** top‑1 = `[d1]`. hits = 1. Precision@1 = 1/1 = 1.0; Recall@1 = 1/2 = 0.5; Hit@1 =
  1.0.

The `0.4` / `1.0` values match `test_retrieval_ndcg_and_short_list_semantics`.

## Formal definition

For each cutoff \(k\), with \(H_k = |\text{relevant} \cap \text{top‑}k|\):

\[
P@k = \frac{H_k}{k}, \qquad R@k = \frac{H_k}{|\text{relevant}|}, \qquad \text{Hit@}k = \mathbb{1}[H_k > 0].
\]

```39:44:src/evalharness/scoring/metrics/retrieval/ndcg.py
        for cutoff in self.config["cutoffs"]:
            selected = ranking[:cutoff]
            hits = sum(doc in relevant for doc in selected)
            detail[f"precision@{cutoff}"] = hits / cutoff
            detail[f"recall@{cutoff}"] = hits / len(relevant)
            detail[f"hit@{cutoff}"] = float(hits > 0)
```

## Inputs & requirements

Same as the family: `retrieval`/`rag`, `QRELS` required, zero‑relevance excluded. These
values are always present in `detail` when the case is scored (not `NULL`).

## Output

- `precision@k`, `recall@k` ∈ `[0, 1]`; `hit@k` ∈ `{0.0, 1.0}` — for each k in the
  configured cutoffs. Higher is better throughout.

## Pitfalls & gotchas

- **Precision@k has a built‑in ceiling when few docs are relevant.** With only 2 relevant
  docs, Precision@5 can never exceed `0.4` no matter how good the ranking — that's a property
  of *k*, not a model failure. Read it against `recall_ceiling`.
- **Binary relevance.** These collapse graded `qrels` to relevant‑vs‑not (`gain > 0`); the
  grade only matters for [NDCG](ndcg.md).
- **Denominator differences.** Precision divides by *k* (fixed), recall by |relevant|
  (varies per query) — don't average them across queries naively; prefer per‑query then mean,
  with a [CI](../statistics/bootstrap.md).
- **Order‑insensitive within k.** A relevant doc at rank 1 and at rank 5 give the same
  Precision@5 — use MRR/NDCG when position within the window matters.

## How it composes

- The interpretable companions to [NDCG](ndcg.md): NDCG for graded, position‑aware quality;
  P/R/Hit for "did the right docs show up." Report a couple of cutoffs, not all five.
- [MRR & MAP](mrr-map.md) add the order‑sensitivity that set metrics lack.

## References & code

- Code: [`RetrievalMetric`](../../../src/evalharness/scoring/metrics/retrieval/ndcg.py).
- Guide: [§6.4](../../guide.md#64-ranking--retrieval).
