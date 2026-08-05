"""Published-text bounds for judge artifacts."""

from __future__ import annotations

from evalharness.observability import sanitize_text

REASONING_LIMIT = 1_000
EVIDENCE_TEXT_LIMIT = 280
GALLERY_TEXT_LIMIT = 280


def truncate_reasoning(value: str) -> str:
    """Bound and redact judge reasoning for published artifacts."""
    return sanitize_text(value, max_chars=REASONING_LIMIT)


def truncate_evidence(value: str) -> str:
    """Bound and redact evidence / quoted spans."""
    return sanitize_text(value, max_chars=EVIDENCE_TEXT_LIMIT)


def truncate_gallery(value: str) -> str:
    """Bound gallery text to the Phase 3 example limit."""
    return sanitize_text(value, max_chars=GALLERY_TEXT_LIMIT)
