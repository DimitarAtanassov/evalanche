"""Agreement metrics for judge calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from evalharness.judge.errors import JudgeError
from evalharness.judge.models import AgreementMetric


def _as_float_pairs(
    human: Sequence[int | float | str],
    judge: Sequence[int | float | str],
) -> tuple[list[float], list[float]]:
    if len(human) != len(judge):
        raise JudgeError("INVALID_LABELS", "human and judge values must be aligned")
    try:
        return [float(value) for value in human], [float(value) for value in judge]
    except (TypeError, ValueError) as exc:
        raise JudgeError(
            "INVALID_LABELS", f"non-numeric values for ordinal agreement: {exc}"
        ) from exc


def _krippendorff_alpha_nominal(human: Sequence[str], judge: Sequence[str]) -> float | None:
    """Two-rater Krippendorff alpha for nominal labels, or None when undefined."""
    pairs = list(zip(human, judge, strict=True))
    n = len(pairs) * 2
    values = [value for pair in pairs for value in pair]
    categories = sorted(set(values))
    if len(categories) < 2:
        return None
    observed = sum(1.0 for left, right in pairs if left != right) / len(pairs)
    marginal = {category: values.count(category) / n for category in categories}
    expected = 1.0 - sum(weight * weight for weight in marginal.values())
    if expected <= 0.0:
        return None
    return float(1.0 - (observed / expected))


def _krippendorff_alpha_ordinal(human: Sequence[float], judge: Sequence[float]) -> float | None:
    """Two-rater ordinal Krippendorff alpha, or None when undefined."""
    pairs = list(zip(human, judge, strict=True))
    values = [value for pair in pairs for value in pair]
    if len(set(values)) < 2:
        return None
    observed = float(np.mean([(a - b) ** 2 for a, b in pairs]))
    # Expected disagreement over all value pairs with replacement.
    expected_terms = [(a - b) ** 2 for a in values for b in values]
    expected = float(np.mean(expected_terms))
    if expected <= 0.0:
        return None
    return float(1.0 - (observed / expected))


def compute_agreement(
    metric: AgreementMetric,
    human: Sequence[int | float | str],
    judge: Sequence[int | float | str],
) -> float | None:
    """Compute the requested agreement metric, or None where it is undefined.

    Chance-corrected agreement needs at least two observed categories to have a
    chance term at all. Reporting 1.0 for a degenerate single-category holdout
    would clear the calibration threshold on no evidence, so those cases return
    None and the caller fails the gate closed.
    """
    if not human:
        return None
    if metric is AgreementMetric.COHEN_KAPPA:
        human_s = [str(value) for value in human]
        judge_s = [str(value) for value in judge]
        if len(set(human_s) | set(judge_s)) < 2:
            return None
        return float(cohen_kappa_score(human_s, judge_s))
    if metric is AgreementMetric.SPEARMAN:
        human_f, judge_f = _as_float_pairs(human, judge)
        if len(set(human_f)) < 2 or len(set(judge_f)) < 2:
            return None
        corr, _ = spearmanr(human_f, judge_f)
        if corr is None or np.isnan(corr):
            return None
        return float(corr)
    if metric is AgreementMetric.KRIPPENDORFF_ALPHA:
        if all(isinstance(value, str) for value in human) and all(
            isinstance(value, str) for value in judge
        ):
            return _krippendorff_alpha_nominal(
                [str(value) for value in human],
                [str(value) for value in judge],
            )
        human_f, judge_f = _as_float_pairs(human, judge)
        return _krippendorff_alpha_ordinal(human_f, judge_f)
    raise JudgeError("INVALID_RUBRIC", f"unsupported agreement metric: {metric}")
