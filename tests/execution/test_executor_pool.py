"""Executor orchestration against an in-memory store: no database required.

The database-backed suites in this package prove the same contracts end to end. These
run everywhere, so the case-budget, cost, and progress-honesty rules stay covered when
Postgres is absent.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from evalharness.app.settings import Settings, get_settings
from evalharness.domain.dataset import Case
from evalharness.domain.enums import ErrorClass, FailureOutcome, FinishReason, TaskType
from evalharness.domain.generation import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.domain.run import RunRecord
from evalharness.execution.executor import Executor
from evalharness.observability import PipelineStage, ProgressEvent

TEMPLATE = "Q: {{ question }}"
RUN_ID = uuid.UUID("30000000-0000-4000-8000-000000000001")


def _settings(**overrides: float | int | str) -> Settings:
    values: dict[str, float | int | str] = {
        "database_url": get_settings().database_url,
        "default_max_retries": 0,
        "default_retry_base_s": 0.0,
        "default_retry_cap_s": 0.0,
        "default_concurrency": 1,
        "default_run_timeout_s": 30.0,
        "default_shutdown_drain_timeout_s": 5.0,
        "default_case_timeout_s": 5.0,
        "default_request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _cases(count: int) -> list[tuple[int, Case]]:
    return [
        (
            index,
            Case(
                external_id=f"case-{index}",
                task_type=TaskType.QA_SHORT,
                inputs={"question": f"q{index}"},
                reference_answer=f"a{index}",
            ),
        )
        for index in range(1, count + 1)
    ]


@dataclass
class _StoreState:
    """Shared across the per-session store instances the factory hands out."""

    cases: list[tuple[int, Case]]
    generations: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: str = "running"


class _FakeStore:
    """Implements only the RunStore calls the generation path makes."""

    def __init__(self, state: _StoreState) -> None:
        self.state = state

    async def get_run(self, run_id: uuid.UUID) -> RunRecord:
        return RunRecord(
            id=run_id,
            dataset_id=1,
            prompt_template_id=1,
            model_version_id=1,
            decode_params={"temperature": 0.0, "stop": []},
            config_sha256="c" * 64,
            harness_version="test",
            git_sha="deadbeef",
            repeats=1,
            status=self.state.status,
            tenant_id="test",
            started_at=None,
            finished_at=None,
            baseline_run_id=None,
        )

    async def get_cases_for_dataset(self, dataset_id: int) -> list[tuple[int, Case]]:
        return self.state.cases

    async def get_completed_keys(self, run_id: uuid.UUID) -> set[tuple[int, int]]:
        return {(gen["case_id"], gen["repeat_idx"]) for gen in self.state.generations}

    async def update_run_status(self, run_id: uuid.UUID, status: str) -> None:
        self.state.status = status

    async def save_generation(self, **kwargs: Any) -> int:
        self.state.generations.append(kwargs)
        return len(self.state.generations)

    async def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        return self.state.cache.get(cache_key)

    async def put_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        self.state.cache[cache_key] = response


class _Provider:
    """Scripted provider: optional per-prompt timeouts, an indefinite hang, or a gate."""

    name = "mock"

    def __init__(
        self,
        *,
        cost_usd: float | None = 0.0,
        timeouts_before_ok: int = 0,
        hang: bool = False,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.cost_usd = cost_usd
        self.timeouts_before_ok = timeouts_before_ok
        self.hang = hang
        self.gate = gate
        self.first_started = asyncio.Event()
        self.calls = 0
        self._timeouts: dict[str, int] = {}

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version="digest-1",
            quantization=None,
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
        if self.gate is not None and self.calls == 1:
            self.first_started.set()
            await self.gate.wait()
        if self.hang:
            await asyncio.sleep(60.0)
        seen = self._timeouts.get(prompt, 0)
        if seen < self.timeouts_before_ok:
            self._timeouts[prompt] = seen + 1
            raise TimeoutError("simulated request timeout")
        raw: dict[str, Any] = {"mock": True}
        if self.cost_usd is not None:
            raw["cost_usd"] = self.cost_usd
        return GenerationResponse(
            text=f"answer for {prompt}",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            prompt_tokens=10,
            completion_tokens=5,
            logprobs=None,
            ttft_ms=1.0,
            total_ms=5.0,
            raw=raw,
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.RETRYABLE_TRANSIENT

    async def aclose(self) -> None:
        return None


def _executor(provider: _Provider, state: _StoreState, settings: Settings) -> Executor:
    return Executor(
        provider=provider,
        model="mock-model",
        model_version=ModelVersion(
            provider="mock",
            model="mock-model",
            resolved_version=f"digest-{uuid.uuid4().hex}",
            quantization=None,
            capabilities=None,
        ),
        template_body=TEMPLATE,
        settings=settings,
        run_store=lambda _session: _FakeStore(state),  # type: ignore[arg-type, return-value]
    )


@pytest.mark.asyncio
async def test_every_planned_case_is_generated_and_run_completes() -> None:
    state = _StoreState(cases=_cases(3))
    provider = _Provider()
    progress: list[ProgressEvent] = []

    await _executor(provider, state, _settings()).execute_run(
        RUN_ID, concurrency=2, progress=progress.append
    )

    assert state.status == "completed"
    assert len(state.generations) == 3
    assert {gen["outcome"] for gen in state.generations} == {FailureOutcome.PASSED}
    generating = [event for event in progress if event.stage == PipelineStage.GENERATING]
    assert generating[-1].completed == 3
    assert generating[-1].counters == {
        "valid_outputs": 3,
        "other_outcomes": 0,
        "retries": 0,
        "cache_hits": 0,
    }


@pytest.mark.asyncio
async def test_cost_is_none_when_provider_omits_it_and_zero_when_reported() -> None:
    unpriced = _StoreState(cases=_cases(2))
    await _executor(_Provider(cost_usd=None), unpriced, _settings()).execute_run(RUN_ID)

    priced = _StoreState(cases=_cases(2))
    await _executor(_Provider(cost_usd=0.0), priced, _settings()).execute_run(RUN_ID)

    assert all(gen["cost_usd"] is None for gen in unpriced.generations)
    assert all(gen["cost_usd"] == 0.0 for gen in priced.generations)


@pytest.mark.asyncio
async def test_request_timeout_is_retried_and_the_successful_attempt_is_persisted() -> None:
    state = _StoreState(cases=_cases(2))
    provider = _Provider(timeouts_before_ok=1)

    await _executor(provider, state, _settings(default_max_retries=3)).execute_run(RUN_ID)

    assert state.status == "completed"
    assert {gen["outcome"] for gen in state.generations} == {FailureOutcome.PASSED}
    assert all(gen["attempts"] == 2 for gen in state.generations)
    assert provider.calls == 4


@pytest.mark.asyncio
async def test_case_budget_expiry_keeps_the_real_timeout_attempts() -> None:
    """The budget must cut retries off early without discarding the attempt log."""
    max_retries = 50
    state = _StoreState(cases=_cases(2))
    provider = _Provider(hang=True)

    await _executor(
        provider,
        state,
        _settings(
            default_case_timeout_s=0.25,
            default_request_timeout_s=0.05,
            default_max_retries=max_retries,
        ),
    ).execute_run(RUN_ID)

    assert {gen["outcome"] for gen in state.generations} == {FailureOutcome.HARNESS_TIMEOUT}
    assert provider.calls > len(state.cases)
    assert provider.calls < len(state.cases) * (max_retries + 1)
    for gen in state.generations:
        assert len(gen["attempt_log"]) > 1
        assert gen["attempt_log"] != [{"attempt": 1, "error_class": "case_timeout"}]
        assert all(entry["error_class"] == "timeout" for entry in gen["attempt_log"])


@pytest.mark.asyncio
async def test_shutdown_skips_remaining_cases_without_counting_them_as_progress() -> None:
    gate = asyncio.Event()
    state = _StoreState(cases=_cases(3))
    provider = _Provider(gate=gate)
    executor = _executor(provider, state, _settings())
    progress: list[ProgressEvent] = []

    run = asyncio.create_task(executor.execute_run(RUN_ID, concurrency=1, progress=progress.append))
    await asyncio.wait_for(provider.first_started.wait(), timeout=5.0)
    executor.shutdown.request("test")
    gate.set()
    await asyncio.wait_for(run, timeout=10.0)

    assert state.status == "cancelled"
    assert len(state.generations) == 1
    generating = [event for event in progress if event.stage == PipelineStage.GENERATING]
    assert generating[-1].completed == 1
    assert generating[-1].total == 3
