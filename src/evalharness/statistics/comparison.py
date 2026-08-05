"""Paired run comparison models and calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evalharness.statistics.core import (
    benjamini_hochberg,
    effect_sizes,
    exact_mcnemar,
    paired_bootstrap,
)


@dataclass(frozen=True)
class ComparisonResult:
    metric: str
    n: int
    baseline: float
    candidate: float
    absolute_delta: float
    relative_delta: float | None
    cohens_h: float
    ci_low: float
    ci_high: float
    p_value: float
    significant_bh: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_binary(
    metric: str,
    baseline: list[bool],
    candidate: list[bool],
    *,
    seed: int = 0,
) -> ComparisonResult:
    if not baseline:
        raise ValueError("No aligned, non-flaky cases to compare")
    old_rate = sum(baseline) / len(baseline)
    new_rate = sum(candidate) / len(candidate)
    delta, interval = paired_bootstrap(
        [float(value) for value in baseline],
        [float(value) for value in candidate],
        seed=seed,
    )
    _, _, p_value = exact_mcnemar(baseline, candidate)
    effects = effect_sizes(old_rate, new_rate)
    return ComparisonResult(
        metric=metric,
        n=len(baseline),
        baseline=old_rate,
        candidate=new_rate,
        absolute_delta=delta,
        relative_delta=effects["relative_delta"],
        cohens_h=float(effects["cohens_h"] or 0.0),
        ci_low=interval[0],
        ci_high=interval[1],
        p_value=p_value,
    )


def apply_multiplicity(results: list[ComparisonResult], q: float = 0.05) -> list[ComparisonResult]:
    decisions = benjamini_hochberg([result.p_value for result in results], q)
    return [
        ComparisonResult(**{**result.to_dict(), "significant_bh": significant})
        for result, significant in zip(results, decisions, strict=True)
    ]
