"""METEOR, which needs the NLTK wordnet corpus at runtime."""

from __future__ import annotations

from typing import Any

from evalharness.domain import Case, Generation, ScoringContext
from evalharness.scoring.base import ScalarMetric, reference_text, tokens

try:
    from nltk.translate.meteor_score import meteor_score as _meteor_score
except ImportError:  # pragma: no cover - optional path when nltk is unavailable
    _meteor_score = None


class MeteorMetric(ScalarMetric):
    name = "meteor"
    config = {"language": "en", "resources": ["wordnet"]}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = reference_text(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        if _meteor_score is None:
            return None, {
                "reason": "nltk_resource_unavailable",
                "language": "en",
                "resource": "wordnet",
                "error": "nltk.translate.meteor_score is not installed",
            }
        try:
            value = float(_meteor_score([tokens(reference)], tokens(gen.output)))
            return value, {"language": "en", "resources": ["wordnet"]}
        except LookupError as exc:
            return None, {
                "reason": "nltk_resource_unavailable",
                "language": "en",
                "resource": "wordnet",
                "error": str(exc),
            }
