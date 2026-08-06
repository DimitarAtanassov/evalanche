"""Shared reading of the HTTP ``Retry-After`` hint carried on provider exceptions."""

from __future__ import annotations

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
