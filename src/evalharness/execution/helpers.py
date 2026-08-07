"""Pure helpers and shared value types for execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment

from evalharness.core.enums import FailureOutcome, FinishReason
from evalharness.core.models import Case, GenerationResponse, ToolCall
from evalharness.execution.errors import DecodeParamsError
from evalharness.hashing import sha256_canonical


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


def response_cache_key(
    *,
    provider: str,
    resolved_version: str,
    rendered_prompt: str,
    decode_params: dict[str, Any],
) -> str:
    """Key for the shared response cache; callers that purge must use this same derivation."""
    return sha256_canonical(
        {
            "provider": provider,
            "model_version": resolved_version,
            "prompt": rendered_prompt,
            "decode": decode_params,
            "adapter": f"{provider}-v1",
        }
    )


def response_from_cache(payload: dict[str, Any]) -> GenerationResponse:
    """Rebuild a GenerationResponse from a cached payload dict."""
    return GenerationResponse(
        text=payload["text"],
        tool_calls=[ToolCall(**call) for call in payload.get("tool_calls", [])],
        finish_reason=FinishReason(payload["finish_reason"]),
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        logprobs=None,
        ttft_ms=payload.get("ttft_ms"),
        total_ms=payload["total_ms"],
        raw=payload.get("raw", {}),
    )


@dataclass(frozen=True)
class RunPlanItem:
    case_db_id: int
    case: Case
    repeat_idx: int


@dataclass(frozen=True)
class RunConfig:
    dataset_id: int
    prompt_template_id: int
    model_version_id: int
    config_sha256: str
    decode_params: dict[str, Any]
    repeats: int
    concurrency: int
    case_timeout_s: float
    request_timeout_s: float
    run_timeout_s: float
    drain_timeout_s: float
    max_retries: int


@dataclass(frozen=True)
class ExecutionResult:
    case_id: int
    external_id: str
    repeat_idx: int
    outcome: FailureOutcome
    attempts: int
    cached: bool
    duration_ms: float | None
    # False when shutdown skipped persist; do not count as completed work.
    persisted: bool = True


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
