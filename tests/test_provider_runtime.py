from __future__ import annotations

import pytest

from evalharness.providers.runtime import CircuitBreaker, CircuitOpenError, CircuitState


def test_circuit_breaker_open_and_half_open(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("evalharness.providers.runtime.time.monotonic", lambda: now)
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=10)
    breaker.failure()
    breaker.failure()
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    now = 111.0
    assert breaker.before_call() == CircuitState.HALF_OPEN
    breaker.success()
    assert breaker.state == CircuitState.CLOSED
