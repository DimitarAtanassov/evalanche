from evalharness.scoring.stats import percentile, wilson_interval


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low <= high <= 1.0
    assert low < 0.8 < high


def test_wilson_interval_zero_n() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
