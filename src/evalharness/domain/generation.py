"""Generation and provider request/response domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from evalharness.domain.enums import FailureOutcome, FinishReason


class Capabilities(TypedDict):
    supports_seed: bool
    supports_logprobs: bool
    supports_tools: bool
    supports_json_schema: bool
    supports_streaming: bool
    supports_system_role: bool
    max_context_tokens: int


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[Message]
    max_tokens: int | None
    temperature: float
    top_p: float | None
    top_k: int | None
    seed: int | None
    stop: list[str]
    response_format: dict[str, Any] | None
    tools: list[ToolSpec] | None
    timeout_s: float


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    prompt_tokens: int | None
    completion_tokens: int | None
    logprobs: list[TokenLogprob] | None
    ttft_ms: float | None
    total_ms: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelVersion:
    provider: str
    model: str
    resolved_version: str
    quantization: str | None = None
    params_b: float | None = None
    context_window: int | None = None
    capabilities: Capabilities | None = None


@dataclass(frozen=True)
class Generation:
    id: int | None
    run_id: str
    case_external_id: str
    repeat_idx: int
    output: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: FinishReason | None
    outcome: FailureOutcome
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    ttft_ms: float | None
    total_ms: float | None
    queue_wait_ms: float | None
    attempts: int
    attempt_log: list[dict[str, Any]]
    cached: bool
    raw_response: dict[str, Any] | None
    trace_id: str | None


@dataclass(frozen=True)
class StoredGeneration:
    """A persisted generation with FK case_id for store-side joins."""

    id: int
    run_id: str
    case_id: int
    case_external_id: str
    repeat_idx: int
    output: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: FinishReason | None
    outcome: FailureOutcome
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    ttft_ms: float | None
    total_ms: float | None
    queue_wait_ms: float | None
    attempts: int
    attempt_log: list[dict[str, Any]]
    cached: bool
    raw_response: dict[str, Any] | None
    trace_id: str | None

    def as_generation(self) -> Generation:
        return Generation(
            id=self.id,
            run_id=self.run_id,
            case_external_id=self.case_external_id,
            repeat_idx=self.repeat_idx,
            output=self.output,
            tool_calls=self.tool_calls,
            finish_reason=self.finish_reason,
            outcome=self.outcome,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
            ttft_ms=self.ttft_ms,
            total_ms=self.total_ms,
            queue_wait_ms=self.queue_wait_ms,
            attempts=self.attempts,
            attempt_log=self.attempt_log,
            cached=self.cached,
            raw_response=self.raw_response,
            trace_id=self.trace_id,
        )
