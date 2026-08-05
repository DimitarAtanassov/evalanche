"""Published-text bounds for RAG evidence artifacts."""

from __future__ import annotations

from evalharness.observability import sanitize_text

CLAIM_TEXT_LIMIT = 280
EVIDENCE_SPAN_LIMIT = 280
EXAMPLE_LIMIT = 8
MAX_CLAIMS_PER_CASE = 20
MAX_CONTEXTS_PER_CASE = 20


def truncate_claim(value: str) -> str:
    return sanitize_text(value, max_chars=CLAIM_TEXT_LIMIT)


def truncate_span(value: str) -> str:
    return sanitize_text(value, max_chars=EVIDENCE_SPAN_LIMIT)
