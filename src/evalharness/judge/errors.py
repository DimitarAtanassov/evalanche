"""Domain errors for the judge and calibration subsystem."""

from __future__ import annotations


class JudgeError(ValueError):
    """Caller-actionable failure while building or calibrating judgments."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
