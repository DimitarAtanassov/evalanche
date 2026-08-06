from __future__ import annotations

import math

from evalharness.scoring.calibration import calibrate_threshold, calibration_metrics
from evalharness.statistics import (
    benjamini_hochberg,
    compare_binary,
    exact_mcnemar,
    pass_at_k,
    required_sample_size,
    wilson_interval,
)


def test_wilson_and_pass_at_k_golden() -> None:
    low, high = wilson_interval(50, 100)
    assert math.isclose(low, 0.4038315, rel_tol=1e-5)
    assert math.isclose(high, 0.5961685, rel_tol=1e-5)
    assert math.isclose(pass_at_k(10, 2, 3), 0.5333333333)


def test_exact_mcnemar_and_bh() -> None:
    b, c, p = exact_mcnemar([True, True, False], [False, True, True])
    assert (b, c, p) == (1, 1, 1.0)
    assert benjamini_hochberg([0.001, 0.02, 0.5]) == [True, True, False]


def test_identical_paired_results_have_finite_zero_width_interval() -> None:
    result = compare_binary(
        "exact_match",
        [True, False, True, False],
        [True, False, True, False],
    )

    assert result.absolute_delta == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0


def test_power_and_calibration() -> None:
    assert required_sample_size(0.5, 0.1) > 100
    metrics = calibration_metrics([True, False, True], [0.9, 0.2, 0.7], bins=3)
    assert 0 <= metrics["adaptive_ece"] <= 1
    result = calibrate_threshold([True, True, False, False], [0.9, 0.8, 0.3, 0.1])
    assert result["dev_f1"] == 1.0
