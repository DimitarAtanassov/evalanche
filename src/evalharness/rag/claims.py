"""Deterministic claim decomposition (``claim_split_v1``)."""

from __future__ import annotations

import re

from evalharness.rag.text import MAX_CLAIMS_PER_CASE, truncate_claim

_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "vs.",
    "etc.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Fig.",
    "No.",
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _protect_abbreviations(text: str) -> str:
    protected = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbreviation, f"__ABBR{index}__")
    return protected


def _restore_abbreviations(text: str) -> str:
    restored = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        restored = restored.replace(f"__ABBR{index}__", abbreviation)
    return restored


def split_claims(
    answer_text: str,
    *,
    explicit_claims: list[str] | None = None,
    max_claims: int = MAX_CLAIMS_PER_CASE,
) -> tuple[list[str], str | None]:
    """Return truncated claims and an optional per-case error code."""
    if explicit_claims is not None:
        claims = [truncate_claim(claim) for claim in explicit_claims if claim.strip()]
        return claims[:max_claims], None

    normalized = " ".join(answer_text.split())
    if not normalized:
        return [], None

    fragments: list[str] = []
    for line in answer_text.splitlines() or [answer_text]:
        line = line.strip()
        if not line:
            continue
        protected = _protect_abbreviations(line)
        parts = (
            _SENTENCE_SPLIT.split(protected) if _SENTENCE_SPLIT.search(protected) else [protected]
        )
        for part in parts:
            restored = _restore_abbreviations(part).strip()
            if restored:
                fragments.append(truncate_claim(restored))

    if not fragments and normalized:
        return [], "CLAIM_PARSE_FAILED"
    return fragments[:max_claims], None
