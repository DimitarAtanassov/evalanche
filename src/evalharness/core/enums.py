"""Enumerations for the evaluation harness."""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    QA_SHORT = "qa_short"
    RETRIEVAL = "retrieval"
    RAG = "rag"
    TOOL_USE = "tool_use"
    AGENT_TRAJECTORY = "agent_trajectory"
    SAFETY = "safety"
    PAIRWISE = "pairwise"


class ErrorClass(StrEnum):
    RETRYABLE_TRANSIENT = "retryable_transient"
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"
    NON_RETRYABLE_REQUEST = "non_retryable_request"
    NON_RETRYABLE_AUTH = "non_retryable_auth"
    MODEL_REFUSAL = "model_refusal"
    CONTENT_FILTER = "content_filter"


class FailureOutcome(StrEnum):
    PASSED = "passed"
    FAILED_SCORE = "failed_score"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    SCHEMA_INVALID = "schema_invalid"
    EMPTY_OUTPUT = "empty_output"
    CONTENT_FILTERED = "content_filtered"
    MODEL_ERROR = "model_error"
    HARNESS_TIMEOUT = "harness_timeout"
    HARNESS_ERROR = "harness_error"
    SKIPPED = "skipped"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class Requirement(StrEnum):
    REFERENCE = "reference"
    QRELS = "qrels"
    EMBEDDINGS = "embeddings"
    JUDGE = "judge"
    LOGPROBS = "logprobs"
