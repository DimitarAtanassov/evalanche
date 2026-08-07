"""Decode-params boundary validation (temperature)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.execution import DecodeParamsError, Executor, validate_decode_params
from evalharness.hashing import sha256_hex
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository


@pytest.mark.parametrize(
    "decode_params",
    [
        {},
        {"temperature": 0.0},
        {"temperature": 0},
        {"temperature": 1.5},
        {"temperature": 0.0, "max_tokens": 32},
    ],
)
def test_validate_decode_params_accepts_numeric_temperature(
    decode_params: dict[str, Any],
) -> None:
    validate_decode_params(decode_params)


@pytest.mark.parametrize(
    "temperature",
    [
        "0.0",
        "hot",
        None,
        True,
        False,
        [0.0],
        {"value": 0.0},
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_validate_decode_params_rejects_illegal_temperature(temperature: object) -> None:
    with pytest.raises(DecodeParamsError, match="temperature"):
        validate_decode_params({"temperature": temperature})


class _StubProvider:
    name = "mock"

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version="mock-digest",
            quantization="Q4",
            capabilities=self.capabilities(model),
        )

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_seed=True,
            supports_logprobs=False,
            supports_tools=False,
            supports_json_schema=False,
            supports_streaming=False,
            supports_system_role=True,
            max_context_tokens=4096,
        )

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="ok",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=1,
            completion_tokens=1,
            logprobs=None,
            ttft_ms=1.0,
            total_ms=1.0,
            raw={},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.RETRYABLE_TRANSIENT

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_create_run_rejects_non_numeric_temperature(db_ready) -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    assert validate_dataset(bundle).valid
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)
    provider = _StubProvider()
    model_version = await provider.resolve_version("stub")
    executor = Executor(
        provider=provider,
        model="stub",
        model_version=model_version,
        template_body=template_body,
    )

    async with session_scope() as session:
        repo = RunRepository(session)
        dataset_id = await repo.upsert_dataset(**dataset_upsert_fields(bundle))
        prompt_template_id = await repo.upsert_prompt_template(
            name="t", version="1", body=template_body, sha256=template_sha
        )
        model_version_id = await repo.upsert_model_version(
            provider=model_version.provider,
            model=model_version.model,
            resolved_version=model_version.resolved_version,
            quantization=model_version.quantization,
            capabilities=model_version.capabilities or {},
        )

    with pytest.raises(DecodeParamsError, match="temperature"):
        await executor.create_run(
            bundle_dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            dataset_sha256=bundle.content_sha256,
            prompt_template_sha256=template_sha,
            decode_params={"temperature": "not-a-number", "max_tokens": 32, "stop": []},
            repeats=1,
            tenant_id="test-decode-params",
        )
