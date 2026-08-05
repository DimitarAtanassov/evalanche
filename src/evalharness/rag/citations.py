"""Citation attribution (``cited_present_relevant_v1``)."""

from __future__ import annotations

from typing import Any, Literal

from evalharness.rag.text import EXAMPLE_LIMIT, truncate_span


def citation_attribution(
    cases: list[dict[str, Any]],
    *,
    nli_labels: dict[tuple[str, int, str], Literal["entailment", "neutral", "contradiction"]]
    | None = None,
) -> dict[str, Any]:
    """Attribute citations that are present and relevant (or NLI-entailed)."""
    total = 0
    attributed = 0
    missing: list[dict[str, Any]] = []
    for case in cases:
        citations = case.get("citations")
        if not isinstance(citations, list) or not citations:
            continue
        contexts = case.get("retrieved_contexts") or []
        retrieved = {
            str(ctx.get("doc_id")): str(ctx.get("text") or "")
            for ctx in contexts
            if isinstance(ctx, dict)
        }
        raw_qrels = case.get("qrels")
        qrels: dict[str, int] = raw_qrels if isinstance(raw_qrels, dict) else {}
        case_id = str(case.get("case_id"))
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            total += 1
            doc_id = str(citation.get("doc_id"))
            claim_index = int(citation.get("claim_index", 0))
            present = doc_id in retrieved
            relevant = int(qrels.get(doc_id, 0) or 0) > 0
            entailed = False
            if nli_labels is not None:
                entailed = nli_labels.get((case_id, claim_index, doc_id)) == "entailment"
            ok = present and (relevant or entailed)
            if ok:
                attributed += 1
            elif len(missing) < EXAMPLE_LIMIT:
                missing.append(
                    {
                        "case_id": case_id,
                        "claim_index": claim_index,
                        "doc_id": doc_id,
                        "present": present,
                        "relevant": relevant,
                        "snippet": truncate_span(retrieved.get(doc_id, "")),
                    }
                )

    if total == 0:
        return {
            "method": "cited_present_relevant_v1",
            "attribution": {
                "status": "unavailable",
                "value": None,
                "n": 0,
                "reason": "CITATIONS_MISSING",
            },
            "missing_support_examples": [],
        }

    return {
        "method": "cited_present_relevant_v1",
        "attribution": {
            "status": "ok",
            "value": attributed / total,
            "n": total,
        },
        "missing_support_examples": missing,
    }
