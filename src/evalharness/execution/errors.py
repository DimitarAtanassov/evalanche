"""Domain errors raised by the execution layer."""

from __future__ import annotations


class ResumeError(Exception):
    """The requested run cannot be resumed with the supplied inputs."""


class DecodeParamsError(ValueError):
    """Decode parameters are illegal and must not reach case execution."""
