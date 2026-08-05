"""Integration test for resumable execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.datasets import load_dataset, validate_dataset
from evalharness.execution.executor import Executor, render_prompt
from evalharness.hashing import sha256_hex
from evalharness.reporting.report import build_report
from evalharness.store.db import session_scope
from evalharness.store.repository import RunRepository
from tests.conftest import MockProvider


@pytest.mark.asyncio
async def test_resume_produces_same_outputs(db_ready) -> None:
    bundle = load_dataset(Path("fixtures/sample_dataset"))
    assert validate_dataset(bundle).valid
    template_body = Path("fixtures/templates/qa.jinja").read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)

    responses = {}
    for case in bundle.cases:
        rendered = render_prompt(template_body, case)
        responses[rendered] = case.reference_answer or "unknown"

    provider = MockProvider(responses)
    model_version = await provider.resolve_version("mock-model")
    executor = Executor(
        provider=provider,
        model="mock-model",
        model_version=model_version,
        template_body=template_body,
    )

    async with session_scope() as session:
        repo = RunRepository(session)
        dataset_id = await repo.upsert_dataset(bundle)
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
        decode_params={"temperature": 0.0, "max_tokens": 32, "seed": 42, "stop": []},
        repeats=1,
        tenant_id="test",
    )

    # Simulate partial run: only first 2 cases
    config, items = await executor.plan(run_id)
    partial_items = items[:2]
    import asyncio

    sem = asyncio.Semaphore(2)
    await asyncio.gather(*[executor._run_one(run_id, config, item, sem) for item in partial_items])

    async with session_scope() as session:
        repo = RunRepository(session)
        partial_gens = await repo.get_generations_for_run(run_id)
        partial_pairs = {(g.case_id, g.repeat_idx, g.output) for g in partial_gens}

    # Resume remaining
    await executor.execute_run(run_id, concurrency=2)

    async with session_scope() as session:
        repo = RunRepository(session)
        all_gens = await repo.get_generations_for_run(run_id)
        all_pairs = {(g.case_id, g.repeat_idx, g.output) for g in all_gens}
        assert len(all_gens) == len(bundle.cases)
        assert partial_pairs.issubset(all_pairs)
        assert all(g.raw_response and g.raw_response["mock"] is True for g in all_gens)

    report = await build_report(run_id, coverage_floor=0.0)
    assert report.config_sha256
    assert report.model_digest == "mock-digest-abc123"
    assert "p50" in report.latency
    assert report.pass_rate_ci[0] <= report.pass_rate <= report.pass_rate_ci[1]
