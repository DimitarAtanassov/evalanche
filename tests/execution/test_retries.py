"""Request-timeout retries vs case-budget expiry through Executor.execute_run."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from evalharness.config import Settings, get_settings
from evalharness.core.enums import ErrorClass, FailureOutcome, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.execution.executor import Executor, render_prompt
from evalharness.hashing import sha256_hex
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository


def _settings(**overrides: float | int | str) -> Settings:
    base = get_settings()
    values: dict[str, float | int | str] = {
        "database_url": base.database_url,
        "default_max_retries": 5,
        "default_retry_base_s": 0.0,
        "default_retry_cap_s": 0.0,
        "default_concurrency": 1,
        "default_run_timeout_s": 30.0,
        "default_shutdown_drain_timeout_s": 2.0,
        "default_case_timeout_s": 5.0,
        "default_request_timeout_s": 0.2,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class _ScriptedProvider:
    """Injectable provider: timeout once per prompt, then answer; or hang forever."""

    name = "mock"

    def __init__(
        self,
        *,
        answers: dict[str, str],
        digest: str,
        timeouts_before_ok: int = 0,
        hang: bool = False,
    ) -> None:
        self.answers = answers
        self.digest = digest
        self.timeouts_before_ok = timeouts_before_ok
        self.hang = hang
        self.calls = 0
        self._timeouts_by_prompt: dict[str, int] = {}

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
        prompt = req.messages[-1].content
        if self.hang:
            await asyncio.sleep(60.0)
        seen = self._timeouts_by_prompt.get(prompt, 0)
        if seen < self.timeouts_before_ok:
            self._timeouts_by_prompt[prompt] = seen + 1
            raise TimeoutError("simulated request timeout")
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


async def _create_run(executor: Executor, *, seed: int) -> object:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    assert validate_dataset(bundle).valid
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)

    async with session_scope() as session:
        repo = RunRepository(session)
        dataset_id = await repo.upsert_dataset(**dataset_upsert_fields(bundle))
        prompt_template_id = await repo.upsert_prompt_template(
            name="t", version="1", body=template_body, sha256=template_sha
        )
        model_version_id = await repo.upsert_model_version(
            provider=executor.model_version.provider,
            model=executor.model_version.model,
            resolved_version=executor.model_version.resolved_version,
            quantization=executor.model_version.quantization,
            capabilities=executor.model_version.capabilities or {},
        )

    return await executor.create_run(
        bundle_dataset_id=dataset_id,
        prompt_template_id=prompt_template_id,
        model_version_id=model_version_id,
        dataset_sha256=bundle.content_sha256,
        prompt_template_sha256=template_sha,
        decode_params={"temperature": 0.0, "max_tokens": 32, "seed": seed, "stop": []},
        repeats=1,
        tenant_id="test-retries",
    )


@pytest.mark.asyncio
async def test_request_timeout_then_success_persists_passed(db_ready) -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    answers = {
        render_prompt(template_body, case): case.reference_answer or "unknown"
        for case in bundle.cases
    }
    # Unique digest keeps response-cache keys isolated across suite runs.
    provider = _ScriptedProvider(
        answers=answers,
        digest=f"retry-ok-{uuid.uuid4().hex}",
        timeouts_before_ok=1,
    )
    model_version = await provider.resolve_version("mock-model")
    executor = Executor(
        provider=provider,
        model="mock-model",
        model_version=model_version,
        template_body=template_body,
        settings=_settings(default_case_timeout_s=5.0, default_max_retries=3),
    )

    run_id = await _create_run(executor, seed=9101)
    await executor.execute_run(run_id, concurrency=1)

    async with session_scope() as session:
        repo = RunRepository(session)
        gens = await repo.get_generations_for_run(run_id)
        run = await repo.get_run(run_id)

    assert run is not None
    assert run.status == "completed"
    assert len(gens) == len(bundle.cases)
    assert {g.outcome for g in gens} == {FailureOutcome.PASSED.value}
    assert all(g.attempts == 2 for g in gens)
    assert provider.calls == len(bundle.cases) * 2


@pytest.mark.asyncio
async def test_case_budget_expiry_writes_harness_timeout_without_exhausting_retries(
    db_ready,
) -> None:
    """Outer case timeout is terminal; retries must not continue past the budget."""
    max_retries = 50
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    answers = {
        render_prompt(template_body, case): case.reference_answer or "unknown"
        for case in bundle.cases
    }
    provider = _ScriptedProvider(
        answers=answers,
        digest=f"retry-budget-{uuid.uuid4().hex}",
        hang=True,
    )
    model_version = await provider.resolve_version("mock-model")
    executor = Executor(
        provider=provider,
        model="mock-model",
        model_version=model_version,
        template_body=template_body,
        settings=_settings(
            default_case_timeout_s=0.25,
            default_request_timeout_s=0.05,
            default_max_retries=max_retries,
        ),
    )

    run_id = await _create_run(executor, seed=9102)
    await executor.execute_run(run_id, concurrency=1)

    async with session_scope() as session:
        repo = RunRepository(session)
        gens = await repo.get_generations_for_run(run_id)
        run = await repo.get_run(run_id)

    assert run is not None
    assert run.status == "completed"
    assert len(gens) == len(bundle.cases)
    assert {g.outcome for g in gens} == {FailureOutcome.HARNESS_TIMEOUT.value}
    # ADR 001 (b): case budget must allow retries (distinct from terminal-on-first-timeout),
    # then cut off well before exhausting max_retries. attempt_log must survive expiry.
    assert provider.calls > len(bundle.cases)
    assert provider.calls < len(bundle.cases) * (max_retries + 1)
    assert all(len(g.attempt_log) > 1 for g in gens)
    assert all(any(entry.get("error_class") == "timeout" for entry in g.attempt_log) for g in gens)
    # Budget expiry must keep real request-timeout attempts, not a synthetic case_timeout row.
    for gen in gens:
        assert isinstance(gen.attempt_log, list)
        assert len(gen.attempt_log) >= 1
        assert gen.attempt_log != [{"attempt": 1, "error_class": "case_timeout"}]
        assert any(
            isinstance(entry, dict) and entry.get("error_class") == "timeout"
            for entry in gen.attempt_log
        )
