# Retrieval & ranking

**Are the right documents ranked highly?** When the task is to retrieve or rank — search,
RAG context selection — accuracy is meaningless; what matters is whether relevant documents
appear, and appear *near the top*. This family scores a ranked list against graded
relevance judgments (`qrels`).

`retrieval_ndcg_10` is the primary metric (NDCG@10), and three siblings publish the other
ranking signals as metrics in their own right, so each is aggregated, sliced, and gated
independently: `retrieval_precision_at_k`, `retrieval_mrr`, and `retrieval_map`. The NDCG
`detail` payload still carries the full battery across cutoffs, retained for one release
and contracted after that; read the sibling metric, not the detail, for anything you gate
on. A dataset that declares only `retrieval_ndcg_10` keeps working unchanged.

- **Code:** [`scoring/metrics/retrieval/`](../../../src/evalharness/scoring/metrics/retrieval/)
- **Requires:** `QRELS` — a `case.qrels` map of `{doc_id: graded_relevance}`.
- **Task types:** `retrieval`, `rag`.

## The signals and where they come from

| Doc | Signals | Metric | One line |
|-----|---------|--------|----------|
| [NDCG](ndcg.md) | NDCG@10, `recall_ceiling` | `retrieval_ndcg_10` | Primary value: graded, position‑discounted quality with **exponential gain** |
| [Precision / recall / hit @k](precision-recall-hit-at-k.md) | P@k, R@k, Hit@k for k∈{1,3,5,10,20} | `retrieval_precision_at_k` (P@k); R@k and Hit@k in the NDCG detail | Set‑based accuracy at each cutoff |
| [MRR & MAP](mrr-map.md) | MRR, MAP | `retrieval_mrr`, `retrieval_map` | Rank of the first hit; mean average precision |

## How the ranking is parsed (shared)

The model's output becomes a ranked list of doc ids, then **relevant** docs are those with
positive graded relevance:

```python
        try:
            ranking = json.loads(gen.output or "[]")
        except json.JSONDecodeError:
            ranking = [part.strip() for part in (gen.output or "").split(",") if part.strip()]
        ranking = list(dict.fromkeys(map(str, ranking)))
        relevant = {str(doc): int(gain) for doc, gain in case.qrels.items() if int(gain) > 0}
```

Key behaviors every sub‑metric inherits:

- **Two accepted output formats:** a JSON array of ids, or a comma‑separated string
  (fallback when JSON parsing fails).
- **De‑duplication & tie handling:** `dict.fromkeys` keeps the **first** occurrence and
  preserves original order — ties break by the order the model emitted them. There is no
  score‑based re‑sort; the *ranking is the model's order*.
- **Zero‑relevance exclusion:** if `case.qrels` is empty/falsy, the metric returns
  `value=None` with `detail={"excluded": "zero_relevance"}` — a query with no relevant docs
  is **excluded, not scored as 0** (it would otherwise unfairly punish or reward). The
  sibling metrics also exclude a case whose judgments are all graded `0`, which has no
  relevant document to rank.

## Pitfalls that apply to the whole family

- **State "exponential gain."** NDCG has two conventions; this code uses \((2^{rel}-1)\),
  which gives different numbers than the linear‑gain variant. Cross‑team disagreements
  almost always trace back to this — see [NDCG](ndcg.md).
- **Graded, not binary.** `qrels` values are graded gains (`{d1: 3, d2: 1}`), so a highly
  relevant doc contributes more. P/R/Hit collapse this to relevant‑vs‑not (any gain > 0).
- **Cutoffs are fixed** at `{1, 3, 5, 10, 20}` in config; NDCG itself is @10.

## Related

- Guide: [§6.4](../../guide.md#64-ranking--retrieval).
- Narrative: [`metrics.md`](../../metrics.md#retrieval--ranking).
- Test: `test_retrieval_ndcg_and_short_list_semantics` in
  [`tests/test_metric_catalog.py`](../../../tests/test_metric_catalog.py).
