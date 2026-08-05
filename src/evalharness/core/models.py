"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from evalharness.core.enums import FailureOutcome, FinishReason, TaskType


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
class Case:
    external_id: str
    task_type: TaskType
    inputs: dict[str, Any]
    reference_answer: str | None = None
    references: list[str] = field(default_factory=list)
    expected_label: str | None = None
    expected_json: dict[str, Any] | None = None
    qrels: dict[str, int] | None = None
    slices: dict[str, str] = field(default_factory=dict)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    canary: str | None = None
    weight: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    normalized_prompt: str | None = None


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
    raw_uri: str | None
    trace_id: str | None


@dataclass(frozen=True)
class ScoreValue:
    metric_name: str
    metric_version: str
    metric_config_sha256: str
    value: float | None
    passed: bool | None
    detail: dict[str, Any]


@dataclass(frozen=True)
class AggregateValue:
    metric_name: str
    slice_key: str
    n: int
    value: float
    ci_low: float | None
    ci_high: float | None
    stddev: float | None
    method: str


@dataclass
class ScoringContext:
    normalizer_id: str
    extra: dict[str, Any] = field(default_factory=dict)
