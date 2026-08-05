"""Domain errors for the RAG evidence subsystem."""

from __future__ import annotations


class RagError(ValueError):
    """Caller-actionable failure while building RAG evidence artifacts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
