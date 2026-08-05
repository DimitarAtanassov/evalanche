"""Seeded statistical inference for evaluation results."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bca_bootstrap(
    values: list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        value = float(statistic(data)) if data.size else 0.0
        return value, value
    result = stats.bootstrap(
        (data,),
        statistic,
        method="BCa",
        n_resamples=resamples,
        confidence_level=confidence,
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    left = np.asarray(baseline, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired samples must have identical shape")
    delta = right - left
    return float(np.mean(delta)), bca_bootstrap(delta.tolist(), resamples=resamples, seed=seed)


def exact_mcnemar(baseline: list[bool], candidate: list[bool]) -> tuple[int, int, float]:
    if len(baseline) != len(candidate):
        raise ValueError("Paired samples must have identical length")
    b = sum(old and not new for old, new in zip(baseline, candidate, strict=True))
    c = sum(not old and new for old, new in zip(baseline, candidate, strict=True))
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return b, c, p


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    cutoff = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / count:
            cutoff = rank
    rejected = [False] * count
    for rank, index in enumerate(order, start=1):
        rejected[index] = rank <= cutoff
    return rejected


def effect_sizes(baseline: float, candidate: float) -> dict[str, float | None]:
    absolute = candidate - baseline
    relative = absolute / baseline if baseline else None
    h = 2 * (math.asin(math.sqrt(candidate)) - math.asin(math.sqrt(baseline)))
    return {"absolute_delta": absolute, "relative_delta": relative, "cohens_h": h}


def between_repeat_variance(values_by_case: list[list[float]]) -> float:
    variances = [float(np.var(values, ddof=1)) for values in values_by_case if len(values) > 1]
    return float(np.mean(variances)) if variances else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or k < 1:
        raise ValueError("Require 0 <= c <= n and k >= 1")
    if n - c < k:
        return 1.0
    log_failure = (
        math.lgamma(n - c + 1)
        - math.lgamma(n - c - k + 1)
        - math.lgamma(n + 1)
        + math.lgamma(n - k + 1)
    )
    return -math.expm1(log_failure)


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    candidate = min(1 - 1e-9, max(1e-9, baseline_rate + minimum_detectable_effect))
    h = abs(2 * (math.asin(math.sqrt(candidate)) - math.asin(math.sqrt(baseline_rate))))
    if h == 0:
        raise ValueError("minimum_detectable_effect must be non-zero")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return int(math.ceil(float(2 * ((z_alpha + z_power) / h) ** 2)))


@dataclass(frozen=True)
class FlakyCase:
    case_id: str
    outcomes: tuple[bool, ...]


def find_flaky_cases(outcomes: dict[str, list[bool]]) -> list[FlakyCase]:
    return [
        FlakyCase(case_id, tuple(values))
        for case_id, values in outcomes.items()
        if len(set(values)) > 1
    ]
