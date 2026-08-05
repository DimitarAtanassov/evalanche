"""Public statistical API."""

from evalharness.statistics.comparison import (
    ComparisonResult,
    apply_multiplicity,
    compare_binary,
)
from evalharness.statistics.core import (
    bca_bootstrap,
    benjamini_hochberg,
    between_repeat_variance,
    effect_sizes,
    exact_mcnemar,
    find_flaky_cases,
    paired_bootstrap,
    pass_at_k,
    required_sample_size,
    wilson_interval,
)

__all__ = [
    "ComparisonResult",
    "apply_multiplicity",
    "bca_bootstrap",
    "benjamini_hochberg",
    "between_repeat_variance",
    "compare_binary",
    "effect_sizes",
    "exact_mcnemar",
    "find_flaky_cases",
    "paired_bootstrap",
    "pass_at_k",
    "required_sample_size",
    "wilson_interval",
]
