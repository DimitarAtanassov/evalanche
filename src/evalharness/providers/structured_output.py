"""Strict versioned JSON output contracts for model-scored workflows."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

STRUCTURED_OUTPUT_VERSION = "1.0"
_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(?P<body>.*?)\n?```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)

POINTWISE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "reasoning", "score"],
    "properties": {
        "schema_version": {"const": STRUCTURED_OUTPUT_VERSION},
        "reasoning": {"type": "string", "minLength": 1},
        "score": {"type": "integer"},
    },
}

PAIRWISE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "reasoning", "preference"],
    "properties": {
        "schema_version": {"const": STRUCTURED_OUTPUT_VERSION},
        "reasoning": {"type": "string", "minLength": 1},
        "preference": {"enum": ["A", "B", "tie"]},
    },
}

NLI_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "label"],
    "properties": {
        "schema_version": {"const": STRUCTURED_OUTPUT_VERSION},
        "label": {"enum": ["entailment", "neutral", "contradiction"]},
    },
}


class StructuredOutputError(ValueError):
    """Model output did not satisfy its declared JSON contract."""


class _StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PointwiseOutput(_StrictOutput):
    schema_version: Literal["1.0"]
    reasoning: str = Field(min_length=1)
    score: int


class PairwiseOutput(_StrictOutput):
    schema_version: Literal["1.0"]
    reasoning: str = Field(min_length=1)
    preference: Literal["A", "B", "tie"]


class NliOutput(_StrictOutput):
    schema_version: Literal["1.0"]
    label: Literal["entailment", "neutral", "contradiction"]


def pointwise_json_schema(*, score_min: int, score_max: int) -> dict[str, object]:
    """Build the pointwise schema with rubric-specific score bounds."""

    return {
        **POINTWISE_JSON_SCHEMA,
        "properties": {
            "schema_version": {"const": STRUCTURED_OUTPUT_VERSION},
            "reasoning": {"type": "string", "minLength": 1},
            "score": {"type": "integer", "minimum": score_min, "maximum": score_max},
        },
    }


def strip_optional_markdown_fence(text: str) -> str:
    """Remove one optional outer JSON markdown fence."""

    match = _MARKDOWN_FENCE.fullmatch(text)
    return match.group("body").strip() if match else text.strip()


def _parse_json_object(text: str) -> object:
    try:
        return json.loads(strip_optional_markdown_fence(text))
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON: {exc.msg}") from exc


def parse_pointwise_output(text: str, *, score_min: int, score_max: int) -> PointwiseOutput:
    """Parse a pointwise score and enforce the rubric range."""

    try:
        output = PointwiseOutput.model_validate(_parse_json_object(text))
    except ValidationError as exc:
        raise StructuredOutputError(f"invalid pointwise output: {exc}") from exc
    if not output.reasoning.strip():
        raise StructuredOutputError("pointwise reasoning must not be blank")
    if output.score < score_min or output.score > score_max:
        raise StructuredOutputError(
            f"pointwise score {output.score} outside rubric range [{score_min}, {score_max}]"
        )
    return output


def parse_pairwise_output(text: str) -> PairwiseOutput:
    """Parse a pairwise preference from strict JSON."""

    try:
        output = PairwiseOutput.model_validate(_parse_json_object(text))
    except ValidationError as exc:
        raise StructuredOutputError(f"invalid pairwise output: {exc}") from exc
    if not output.reasoning.strip():
        raise StructuredOutputError("pairwise reasoning must not be blank")
    return output


def parse_nli_output(text: str) -> NliOutput:
    """Parse one NLI label from strict JSON."""

    try:
        return NliOutput.model_validate(_parse_json_object(text))
    except ValidationError as exc:
        raise StructuredOutputError(f"invalid NLI output: {exc}") from exc
