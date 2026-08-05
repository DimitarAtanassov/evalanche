"""Context precision/recall from qrels (``qrels_context_v1``)."""

from __future__ import annotations

from typing import Any


def context_precision_recall(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean precision/recall over cases that supply qrels."""
    precision_values: list[float] = []
    recall_values: list[float] = []
    for case in cases:
        qrels = case.get("qrels")
        if not isinstance(qrels, dict):
            continue
        contexts = case.get("retrieved_contexts") or []
        retrieved_ids = [str(ctx.get("doc_id")) for ctx in contexts if isinstance(ctx, dict)]
        relevant = {str(doc_id) for doc_id, grade in qrels.items() if int(grade) > 0}
        if not retrieved_ids:
            precision_values.append(0.0)
        else:
            relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant)
            precision_values.append(relevant_retrieved / len(retrieved_ids))
        if not relevant:
            recall_values.append(0.0)
        else:
            retrieved_relevant_ids = set(retrieved_ids) & relevant
            recall_values.append(len(retrieved_relevant_ids) / len(relevant))

    if not precision_values:
        unavailable = {
            "status": "unavailable",
            "value": None,
            "n": 0,
            "reason": "QRELS_MISSING",
        }
        return {
            "method": "qrels_context_v1",
            "precision": dict(unavailable),
            "recall": dict(unavailable),
        }

    n = len(precision_values)
    return {
        "method": "qrels_context_v1",
        "precision": {
            "status": "ok",
            "value": sum(precision_values) / n,
            "n": n,
        },
        "recall": {
            "status": "ok",
            "value": sum(recall_values) / n,
            "n": n,
        },
    }
