"""Caller-actionable failures for matrix and baseline manifests."""

from __future__ import annotations


class MatrixValidationError(ValueError):
    """Fail-closed error while validating matrix.yaml or baseline.yaml."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
