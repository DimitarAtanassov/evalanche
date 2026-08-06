import pytest

from evalharness.scoring.stats import percentile, wilson_interval
from evalharness.statistics import percentile as public_percentile
from evalharness.statistics import wilson_interval as public_wilson_interval


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low <= high <= 1.0
    assert low < 0.8 < high


def test_wilson_interval_zero_n() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_public_wilson_interval_matches_pinned_values() -> None:
    assert public_wilson_interval(50, 100) == (
        pytest.approx(0.4038315303659956, abs=1e-12),
        pytest.approx(0.5961684696340044, abs=1e-12),
    )
    assert public_wilson_interval(0, 0) == (0.0, 0.0)
    assert public_wilson_interval(1, 1) == (
        pytest.approx(0.20654931437723745, abs=1e-12),
        1.0,
    )


def test_public_percentile_matches_pinned_values() -> None:
    # k = 3.6 lands between the 4th and 5th sample, so this exercises interpolation.
    assert public_percentile([10.0, 20.0, 30.0, 40.0, 50.0], 0.9) == pytest.approx(46.0, abs=1e-12)
    assert public_percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5, abs=1e-12)
    assert public_percentile([], 0.5) == 0.0
