# MRR & MAP

Sub‑metrics inside `retrieval_ndcg_10`'s `detail`.

## TL;DR

- **MRR** (Mean Reciprocal Rank) — how high up is the **first** relevant doc? `1 / its rank`.
- **MAP** (Mean Average Precision) — averaged precision measured **each time** a relevant
  doc appears; rewards getting *all* relevant docs high.

## What they measure & why

Set metrics ([P/R/Hit@k](precision-recall-hit-at-k.md)) ignore order within the window; MRR
and MAP put order back. **MRR** is ideal when the user wants *one* good answer fast (the
first hit is everything) — a first‑position hit scores 1.0, a fifth‑position hit scores 0.2.
**MAP** is the go‑to single‑number summary when *all* relevant docs matter and you want
credit for surfacing each of them early.

## Intuition (tiny worked example)

`relevant = {d1, d2}`, ranking `["d1", "d2"]`:

- **MRR:** first relevant is `d1` at rank 1 → `1/1 = 1.0`.
- **MAP:** precision at each relevant hit → at d1 (rank 1): 1/1 = 1.0; at d2 (rank 2): 2/2 =
  1.0. Sum = 2.0, divided by |relevant| = 2 → **1.0**.

Now ranking `["x", "d1", "d2"]` (an irrelevant doc first): MRR = `1/2 = 0.5`; MAP = (1/2 +
2/3)/2 ≈ 0.583 — both penalize the wasted top slot.

## Formal definition

Let `ranks` be the 1‑based positions of relevant docs in the ranking.

\[
\mathrm{MRR} = \frac{1}{\min(\text{ranks})}, \qquad
\mathrm{MAP} = \frac{1}{|\text{relevant}|} \sum_{\text{rank } r \text{ of a relevant doc}} \frac{|\text{relevant} \cap \text{top‑}r|}{r}.
\]

```302:309:src/evalharness/scoring/catalog.py
        ranks = [index + 1 for index, doc in enumerate(ranking) if doc in relevant]
        detail["mrr"] = 1 / min(ranks) if ranks else 0.0
        precisions = [
            sum(item in relevant for item in ranking[:rank]) / rank
            for rank, item in enumerate(ranking, start=1)
            if item in relevant
        ]
        detail["map"] = sum(precisions) / len(relevant)
```

Note the MAP denominator is `len(relevant)` (all relevant docs), so relevant docs **missing**
from the ranking implicitly contribute 0 to the average — the correct, recall‑aware
convention.

## Inputs & requirements

Same as the family: `retrieval`/`rag`, `QRELS` required, zero‑relevance excluded. Both
values are always present in `detail` when scored (MRR is `0.0` if no relevant doc appears).

## Output

- `mrr` ∈ `[0, 1]` — `1/rank` of the first hit; higher is better; `0.0` if none retrieved.
- `map` ∈ `[0, 1]` — recall‑aware mean average precision; higher is better.

## Pitfalls & gotchas

- **MRR only sees the first hit.** A ranking with one perfect top result but every other
  relevant doc missing still scores MRR = 1.0 — pair with recall / MAP.
- **MAP divides by |relevant|, not by hits.** Docs never retrieved drag MAP down (they
  should) — MAP conflates precision and recall by design; that's a feature, but don't read it
  as pure precision.
- **Unbounded list.** MRR/MAP scan the *whole* ranking (not just top‑10), unlike NDCG@10, so
  a deep relevant doc still contributes to MAP but not to NDCG@10.
- **Graded relevance ignored.** Like P/R/Hit, these use binary relevant‑vs‑not; grades only
  affect [NDCG](ndcg.md).

## How it composes

- MRR for "first good answer" tasks (QA retrieval); MAP as the order‑aware single number
  when all relevant docs count; [NDCG](ndcg.md) when relevance is graded.
- Report one or two of these deliberately rather than dumping the whole `detail` — they
  answer overlapping questions.

## References & code

- Code: [`RetrievalMetric`](../../../src/evalharness/scoring/catalog.py).
- Guide: [§6.4](../../guide.md#64-ranking--retrieval).
- Lineage: MAP/MRR are TREC‑era IR standards (Voorhees et al.).
