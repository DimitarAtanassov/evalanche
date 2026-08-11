"""Executor persists provider cost through the public execute_run seam (ADR 003)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from evalharness.app.settings import Settings, get_settings
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.domain.constants import GATES_SCHEMA_VERSION
from evalharness.domain.enums import ErrorClass, FailureOutcome, FinishReason
from evalharness.domain.generation import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
)
from evalharness.execution.executor import Executor, render_prompt
from evalharness.gates.evaluate import evaluate_gates
from evalharness.gates.models import (
    CostGate,
    GateArtifacts,
    GateSeverity,
    GatesManifest,
    LoadedGates,
)
from evalharness.hashing import sha256_hex
from evalharness.reporting.report import build_report, report_to_json
from evalharness.db.session import session_scope
from evalharness.repositories import RunStoreUow
from evalharness.suite.models import RunArtifact


def _settings(**overrides: float | int | str) -> Settings:
    base = get_settings()
    values: dict[str, float | int | str] = {
        "database_url": base.database_url,
        "default_max_retries": 0,
        "default_retry_base_s": 0.0,
        "default_retry_cap_s": 0.0,
        "default_concurrency": 1,
        "default_run_timeout_s": 30.0,
        "default_shutdown_drain_timeout_s": 2.0,
        "default_case_timeout_s": 5.0,
        "default_request_timeout_s": 2.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class _CostProvider:
    """Answers the sample dataset; cost_usd is omitted from raw when None."""

    name = "mock"

    def __init__(
        self,
        *,
        answers: dict[str, str],
        digest: str,
        cost_usd: float | None,
    ) -> None:
        self.answers = answers
        self.digest = digest
        self.cost_usd = cost_usd
        self.calls = 0

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
        text = self.answers.get(prompt, "unknown")
        raw: dict[str, object] = {"mock": True, "prompt": prompt}
        if self.cost_usd is not None:
            raw["cost_usd"] = self.cost_usd
        return GenerationResponse(
            text=text,
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
        return [[0.1, 0.2] for _ in texts]

    def classify_error(self, exc: Exception) -> ErrorClass:
        return ErrorClass.NON_RETRYABLE_REQUEST

    async def aclose(self) -> None:
        return None


async def _create_run(executor: Executor, *, seed: int) -> object:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    assert validate_dataset(bundle).valid
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)

    async with session_scope() as session:
        repo = RunStoreUow(session)
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
        tenant_id="test-cost-persist",
    )


async def _run_with_cost(
    *,
    cost_usd: float | None,
    seed: int,
) -> tuple[uuid.UUID, list[object], int]:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    answers = {
        render_prompt(template_body, case): case.reference_answer or "unknown"
        for case in bundle.cases
    }
    provider = _CostProvider(
        answers=answers,
        digest=f"cost-{uuid.uuid4().hex}",
        cost_usd=cost_usd,
    )
    model_version = await provider.resolve_version("mock-model")
    executor = Executor(
        provider=provider,
        model="mock-model",
        model_version=model_version,
        template_body=template_body,
        settings=_settings(),
    )
    run_id = await _create_run(executor, seed=seed)
    assert isinstance(run_id, uuid.UUID)
    await executor.execute_run(run_id, concurrency=1)

    async with session_scope() as session:
        repo = RunStoreUow(session)
        gens = await repo.get_generations_for_run(run_id)

    return run_id, list(gens), len(bundle.cases)


@pytest.mark.asyncio
async def test_executor_persists_none_cost_when_provider_raw_omits_cost(db_ready) -> None:
    _run_id, gens, case_count = await _run_with_cost(cost_usd=None, seed=9301)

    assert len(gens) == case_count
    assert {g.outcome for g in gens} == {FailureOutcome.PASSED.value}
    assert all(g.cost_usd is None for g in gens)


@pytest.mark.asyncio
async def test_executor_persists_explicit_zero_cost_from_provider_raw(db_ready) -> None:
    _run_id, gens, case_count = await _run_with_cost(cost_usd=0.0, seed=9302)

    assert len(gens) == case_count
    assert {g.outcome for g in gens} == {FailureOutcome.PASSED.value}
    assert all(g.cost_usd is not None for g in gens)
    assert all(float(g.cost_usd) == 0.0 for g in gens)


@pytest.mark.asyncio
async def test_unpriced_run_report_fails_blocking_cost_gate(db_ready) -> None:
    run_id, gens, case_count = await _run_with_cost(cost_usd=None, seed=9303)
    assert {g.outcome for g in gens} == {FailureOutcome.PASSED.value}
    assert all(g.cost_usd is None for g in gens)

    report = await build_report(run_id, coverage_floor=0.0)
    assert report.cost_usd_total == 0.0
    assert report.cost_unpriced_generations == case_count

    artifact = RunArtifact.model_validate(report_to_json(report))
    loaded = LoadedGates(
        manifest_path="in-memory",
        manifest=GatesManifest(
            schema_version=GATES_SCHEMA_VERSION,
            name="cost-unpriced",
            artifacts=GateArtifacts(run_report="report.json"),
            gates=[
                CostGate(
                    name="cost-cap",
                    severity=GateSeverity.BLOCKING,
                    max_usd=1.0,
                )
            ],
        ),
        run_report=artifact,
    )
    result = evaluate_gates(loaded)

    assert result.blocking_failed is True
    assert result.results[0].passed is False
    assert result.results[0].blocking_failure is True
    assert result.results[0].evidence["cost_unpriced_generations"] == case_count
