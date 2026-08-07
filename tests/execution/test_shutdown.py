"""Shutdown progress honesty: non-persisted early returns are not completed work."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from evalharness.config import Settings, get_settings
from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.execution.executor import Executor, render_prompt
from evalharness.hashing import sha256_hex
from evalharness.observability import PipelineStage, ProgressEvent
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository


def _settings(**overrides: float | int | str) -> Settings:
    base = get_settings()
    values: dict[str, float | int | str] = {
        "database_url": base.database_url,
        "default_max_retries": 0,
        "default_retry_base_s": 0.0,
        "default_retry_cap_s": 0.0,
        "default_concurrency": 1,
        "default_run_timeout_s": 30.0,
        "default_shutdown_drain_timeout_s": 5.0,
        "default_case_timeout_s": 10.0,
        "default_request_timeout_s": 5.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class _GateProvider:
    """Blocks the first generate until released; later calls succeed immediately."""

    name = "mock"

    def __init__(self, answers: dict[str, str], digest: str) -> None:
        self.answers = answers
        self.digest = digest
        self.calls = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version=self.digest,
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
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await self.release_first.wait()
        prompt = req.messages[-1].content
        text = self.answers.get(prompt, "unknown")
        return GenerationResponse(
            text=text,
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=10,
            completion_tokens=5,
            logprobs=None,
            ttft_ms=1.0,
            total_ms=5.0,
            raw={"mock": True, "cost_usd": 0.0},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.RETRYABLE_TRANSIENT

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_shutdown_before_persist_does_not_count_as_completed(db_ready) -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    assert validate_dataset(bundle).valid
    assert len(bundle.cases) >= 3
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)
    answers = {
        render_prompt(template_body, case): case.reference_answer or "unknown"
        for case in bundle.cases
    }
    provider = _GateProvider(answers, digest=f"shutdown-{uuid.uuid4().hex}")
    model_version = await provider.resolve_version("mock-model")
    executor = Executor(
        provider=provider,
        model="mock-model",
        model_version=model_version,
        template_body=template_body,
        settings=_settings(),
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

    run_id = await executor.create_run(
        bundle_dataset_id=dataset_id,
        prompt_template_id=prompt_template_id,
        model_version_id=model_version_id,
        dataset_sha256=bundle.content_sha256,
        prompt_template_sha256=template_sha,
        decode_params={"temperature": 0.0, "max_tokens": 32, "seed": 9201, "stop": []},
        repeats=1,
        tenant_id="test-shutdown",
    )

    progress: list[ProgressEvent] = []
    run_task = asyncio.create_task(
        executor.execute_run(run_id, concurrency=1, progress=progress.append)
    )
    await asyncio.wait_for(provider.first_started.wait(), timeout=5.0)
    executor.shutdown.request("test")
    provider.release_first.set()
    await asyncio.wait_for(run_task, timeout=10.0)

    generating = [event for event in progress if event.stage == PipelineStage.GENERATING]
    assert generating
    final_progress = generating[-1]
    assert final_progress.completed == 1
    assert final_progress.total == len(bundle.cases)

    async with session_scope() as session:
        repo = RunRepository(session)
        gens = await repo.get_generations_for_run(run_id)
        run = await repo.get_run(run_id)

    assert run is not None
    assert run.status == "cancelled"
    assert len(gens) == 1
