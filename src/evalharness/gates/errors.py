"""Caller-actionable failures for gates manifest loading and evaluation."""

from __future__ import annotations


class GatesValidationError(ValueError):
    """Fail-closed error while validating a gates.yaml or its bound artifacts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
