"""Pure functions shared by the execution pipeline: validation, rendering, classification."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from jinja2 import Environment

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome, FinishReason
from evalharness.execution.errors import DecodeParamsError


def validate_decode_params(decode_params: Mapping[str, Any]) -> None:
    """Fail fast on decode params that would crash mid-case (e.g. non-numeric temperature)."""
    if "temperature" not in decode_params:
        return
    raw = decode_params["temperature"]
    # bool is an int subclass; reject it so True/False never become 1.0/0.0 silently.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DecodeParamsError(f"decode_params.temperature must be a finite number, got {raw!r}")
    if not math.isfinite(raw):
        raise DecodeParamsError(f"decode_params.temperature must be finite, got {raw!r}")


def render_prompt(template: str, case: Case) -> str:
    """Render a trusted local template against case inputs."""
    return Environment(autoescape=False).from_string(template).render(**case.inputs)


def classify_outcome(
    *,
    output: str | None,
    finish_reason: FinishReason | None,
    harness_error: bool,
    harness_timeout: bool,
) -> FailureOutcome:
    if harness_timeout:
        return FailureOutcome.HARNESS_TIMEOUT
    if harness_error:
        return FailureOutcome.HARNESS_ERROR
    if not output or not output.strip():
        return FailureOutcome.EMPTY_OUTPUT
    if finish_reason == FinishReason.LENGTH:
        return FailureOutcome.TRUNCATED
    if finish_reason == FinishReason.CONTENT_FILTER:
        return FailureOutcome.CONTENT_FILTERED
    if output.strip().lower().startswith("i can't") or output.strip().lower().startswith(
        "i cannot"
    ):
        return FailureOutcome.REFUSED
    return FailureOutcome.PASSED
