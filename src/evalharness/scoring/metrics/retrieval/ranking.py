"""Ranking parsing shared by the retrieval metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping

# Serializes to the same JSON array the metric configs hashed before the split, so
# ``metric_config_sha256`` stays comparable across the refactor.
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20)
DEFAULT_PRIMARY_CUTOFF = 10


def parse_ranking(output: str | None) -> list[str]:
    """Ranked doc ids from a JSON array, falling back to a comma-separated list.

    Duplicates are dropped keeping first position: a rerun of the same document cannot
    earn credit twice.
    """
    try:
        ranking = json.loads(output or "[]")
    except json.JSONDecodeError:
        ranking = [part.strip() for part in (output or "").split(",") if part.strip()]
    return list(dict.fromkeys(map(str, ranking)))


def graded_relevance(qrels: Mapping[str, int]) -> dict[str, int]:
    """Doc id to gain for the positively graded judgements only."""
    return {str(doc): int(gain) for doc, gain in qrels.items() if int(gain) > 0}
