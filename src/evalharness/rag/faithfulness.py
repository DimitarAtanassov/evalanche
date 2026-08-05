"""Faithfulness via optional NLI provider seam (``claim_nli_v1``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from evalharness.rag.claims import split_claims
from evalharness.rag.errors import RagError
from evalharness.rag.text import EXAMPLE_LIMIT, MAX_CONTEXTS_PER_CASE, truncate_claim, truncate_span

NliLabel = Literal["entailment", "neutral", "contradiction"]


def load_mock_nli_responses(path: Path) -> dict[tuple[str, int, str], NliLabel]:
    """Load mock NLI labels keyed by ``(case_id, claim_index, doc_id)``."""
    if not path.is_file():
        raise RagError("MISSING_ARTIFACT", str(path))
    mapping: dict[tuple[str, int, str], NliLabel] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RagError("INVALID_NLI_RESPONSES", f"{path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagError("INVALID_NLI_RESPONSES", f"{path}:{index}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RagError("INVALID_NLI_RESPONSES", f"{path}:{index}: expected object")
        label = payload.get("label")
        if label not in {"entailment", "neutral", "contradiction"}:
            raise RagError("INVALID_NLI_RESPONSES", f"{path}:{index}: invalid label {label!r}")
        key = (
            str(payload["case_id"]),
            int(payload["claim_index"]),
            str(payload["doc_id"]),
        )
        mapping[key] = label
    return mapping


def build_faithfulness(
    cases: list[dict[str, Any]],
    *,
    nli_labels: dict[tuple[str, int, str], NliLabel] | None,
) -> tuple[dict[str, Any], dict[tuple[str, int, str], NliLabel], list[dict[str, Any]]]:
    """Build faithfulness section; unavailable when NLI labels are absent."""
    claim_parse_notes: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    used_labels: dict[tuple[str, int, str], NliLabel] = {}

    if nli_labels is None:
        for case in cases:
            case_id = str(case["case_id"])
            explicit = case.get("claims")
            claims, error = split_claims(
                str(case.get("answer_text") or ""),
                explicit_claims=[str(item) for item in explicit]
                if isinstance(explicit, list)
                else None,
            )
            if error:
                claim_parse_notes.append({"case_id": case_id, "reason": error})
            if len(examples) < EXAMPLE_LIMIT:
                examples.append(
                    {
                        "case_id": case_id,
                        "claims": [
                            {
                                "text": truncate_claim(claim),
                                "supported": None,
                                "evidence_spans": [],
                            }
                            for claim in claims
                        ],
                    }
                )
        return (
            {
                "method": "claim_nli_v1",
                "status": "unavailable",
                "reason": "NLI_UNAVAILABLE",
                "aggregate": {"unsupported_claim_rate": None, "n": 0},
                "examples": examples,
            },
            used_labels,
            claim_parse_notes,
        )

    unsupported = 0
    total_claims = 0
    for case in cases:
        case_id = str(case["case_id"])
        explicit = case.get("claims")
        claims, error = split_claims(
            str(case.get("answer_text") or ""),
            explicit_claims=[str(item) for item in explicit]
            if isinstance(explicit, list)
            else None,
        )
        if error:
            claim_parse_notes.append({"case_id": case_id, "reason": error})
            continue
        contexts = [ctx for ctx in (case.get("retrieved_contexts") or []) if isinstance(ctx, dict)][
            :MAX_CONTEXTS_PER_CASE
        ]
        claim_rows: list[dict[str, Any]] = []
        for claim_index, claim in enumerate(claims):
            total_claims += 1
            evidence_spans: list[dict[str, str]] = []
            supported = False
            contradicted = False
            for ctx in contexts:
                doc_id = str(ctx.get("doc_id"))
                key = (case_id, claim_index, doc_id)
                label = nli_labels.get(key)
                if label is None:
                    raise RagError(
                        "MOCK_RESPONSE_MISSING",
                        f"missing NLI label for case_id={case_id} "
                        f"claim_index={claim_index} doc_id={doc_id}",
                    )
                used_labels[key] = label
                if label == "entailment":
                    supported = True
                    evidence_spans.append(
                        {
                            "doc_id": doc_id,
                            "text": truncate_span(str(ctx.get("text") or "")),
                        }
                    )
                elif label == "contradiction":
                    contradicted = True
            if not supported:
                unsupported += 1
            claim_rows.append(
                {
                    "text": truncate_claim(claim),
                    "supported": supported,
                    "contradicted": contradicted,
                    "evidence_spans": evidence_spans,
                }
            )
        if len(examples) < EXAMPLE_LIMIT:
            examples.append({"case_id": case_id, "claims": claim_rows})

    rate = (unsupported / total_claims) if total_claims else 0.0
    return (
        {
            "method": "claim_nli_v1",
            "status": "ok",
            "aggregate": {"unsupported_claim_rate": rate, "n": total_claims},
            "examples": examples,
        },
        used_labels,
        claim_parse_notes,
    )
