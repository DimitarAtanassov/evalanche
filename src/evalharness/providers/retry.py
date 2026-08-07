"""Shared reading of Retry-After and exponential backoff with jitter."""

from __future__ import annotations

import random
from datetime import UTC, datetime


def retry_after_seconds(exc: Exception) -> float | None:
    """Delay requested by the ``Retry-After`` header on the exception's response.

    Accepts either form the header allows, a delay in seconds or an HTTP-date, and
    never returns a negative delay. Returns ``None`` when the exception carries no
    response, the response carries no header, or the value parses as neither form.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=UTC)
        except ValueError:
            return None
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def exponential_jitter_seconds(
    attempt: int,
    *,
    base_s: float,
    cap_s: float,
    rng: random.Random | None = None,
) -> float:
    """Uniform draw in ``[0, min(cap_s, base_s * 2**attempt)]``."""
    draw = rng if rng is not None else random.Random()
    window = min(cap_s, base_s * (2**attempt))
    return draw.uniform(0.0, window)


def retry_delay_seconds(
    attempt: int,
    *,
    base_s: float,
    cap_s: float,
    retry_after_s: float | None = None,
    remaining_budget_s: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Backoff delay: max(jitter, Retry-After), then clamp to the remaining case budget."""
    delay = max(
        exponential_jitter_seconds(attempt, base_s=base_s, cap_s=cap_s, rng=rng),
        retry_after_s or 0.0,
    )
    if remaining_budget_s is not None:
        delay = min(delay, max(0.0, remaining_budget_s))
    return delay
