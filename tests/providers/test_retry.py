"""Characterization tests for shared retry backoff / Retry-After delay helpers."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from evalharness.providers.retry import (
    exponential_jitter_seconds,
    retry_after_seconds,
    retry_delay_seconds,
)


class _ExcWithResponse(Exception):
    def __init__(self, headers: dict[str, str]) -> None:
        self.response = SimpleNamespace(headers=headers)
        super().__init__("provider error")


def test_exponential_jitter_is_uniform_in_capped_window() -> None:
    rng = random.Random(7)
    # attempt=2, base=0.5 => raw = 2.0; cap=1.5 => window [0, 1.5]
    delays = [exponential_jitter_seconds(2, base_s=0.5, cap_s=1.5, rng=rng) for _ in range(50)]
    assert all(0.0 <= d <= 1.5 for d in delays)
    # Seeded draw must be deterministic across runs.
    rng2 = random.Random(7)
    expected = [exponential_jitter_seconds(2, base_s=0.5, cap_s=1.5, rng=rng2) for _ in range(50)]
    assert delays == expected


def test_retry_delay_raises_to_retry_after_then_clamps_budget() -> None:
    rng = random.Random(0)
    # Force jitter below Retry-After by using zero base (window is [0, 0]).
    delay = retry_delay_seconds(
        0,
        base_s=0.0,
        cap_s=10.0,
        retry_after_s=4.0,
        remaining_budget_s=2.5,
        rng=rng,
    )
    assert delay == 2.5


def test_retry_delay_without_budget_matches_max_of_jitter_and_retry_after() -> None:
    rng = random.Random(42)
    jitter = exponential_jitter_seconds(1, base_s=1.0, cap_s=8.0, rng=random.Random(42))
    delay = retry_delay_seconds(
        1,
        base_s=1.0,
        cap_s=8.0,
        retry_after_s=0.25,
        rng=rng,
    )
    assert delay == max(jitter, 0.25)


def test_retry_delay_none_retry_after_behaves_like_zero() -> None:
    rng = random.Random(99)
    delay = retry_delay_seconds(
        0,
        base_s=0.0,
        cap_s=1.0,
        retry_after_s=None,
        remaining_budget_s=5.0,
        rng=rng,
    )
    assert delay == 0.0


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("3", 3.0),
        ("0", 0.0),
        ("-2", 0.0),
    ],
)
def test_retry_after_seconds_numeric_header(header: str, expected: float) -> None:
    assert retry_after_seconds(_ExcWithResponse({"Retry-After": header})) == expected


def test_retry_after_seconds_missing_response_or_header_returns_none() -> None:
    assert retry_after_seconds(Exception("no response")) is None
    assert retry_after_seconds(_ExcWithResponse({})) is None
