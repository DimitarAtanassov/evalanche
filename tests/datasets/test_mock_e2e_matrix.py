"""In-process mock E2E: scoring exactness and real report assembly seam."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalharness.core.enums import FailureOutcome
from evalharness.core.models import GenerationRequest, Message
from evalharness.datasets import load_dataset, validate_dataset
from evalharness.execution.executor import render_prompt
from evalharness.hashing import sha256_hex
from evalharness.reporting.report import (
    EXAMPLE_TEXT_LIMIT,
    assemble_run_report,
    report_to_json,
)
from evalharness.scoring.engine import OVERALL_SLICE, ScoringEngine
from tests.conftest import MockProvider
from tests.datasets._helpers import (
    SMOKE_ROOT,
    perfect_generation,
    perfect_output,
    smoke_paths,
    template_for_dataset,
)


@pytest.mark.parametrize("dataset_path", smoke_paths(), ids=lambda path: path.name)
@pytest.mark.asyncio
async def test_mock_provider_scores_exact_task_fit_values(dataset_path: Path) -> None:
    bundle = load_dataset(dataset_path)
    assert validate_dataset(bundle).valid
    assert bundle.manifest.task_metrics is not None
    metric_names = list(bundle.manifest.task_metrics)
    template_body = template_for_dataset(dataset_path).read_text(encoding="utf-8")
    responses = {render_prompt(template_body, case): perfect_output(case) for case in bundle.cases}
    provider = MockProvider(responses)
    engine = ScoringEngine()

    for case in bundle.cases:
        rendered = render_prompt(template_body, case)
        generation_response = await provider.generate("mock-model", _request_for(rendered))
        assert generation_response.text == perfect_output(case)
        generation = replace(
            perfect_generation(case),
            output=generation_response.text,
            raw_response=generation_response.raw,
        )
        scores = engine.score_one(generation, case, metric_names)
        assert {score.metric_name: score.value for score in scores} == {
            name: 1.0 for name in metric_names
        }
        assert all(score.passed is True for score in scores)

    assert len(provider.calls) == len(bundle.cases)


def _report_inputs(dataset_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Build `assemble_run_report` kwargs for an all-passing run over a smoke dataset."""
    bundle = load_dataset(dataset_path)
    assert bundle.manifest.task_metrics is not None
    metric_names = list(bundle.manifest.task_metrics)
    template_path = template_for_dataset(dataset_path)
    template_body = template_path.read_text(encoding="utf-8")
    engine = ScoringEngine()
    primary_metric = metric_names[0]

    generations: list[SimpleNamespace] = []
    score_rows: list[SimpleNamespace] = []
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    cases_by_id: dict[int, object] = {}

    for index, case in enumerate(bundle.cases, start=1):
        generation = replace(perfect_generation(case), raw_response={"mock": True})
        # Overlong output proves assemble_run_report truncates via production helpers.
        long_output = (generation.output or "") + (" pad" * 200)
        scores = engine.score_one(
            replace(generation, output=perfect_output(case)),
            case,
            metric_names,
        )
        cases_by_id[index] = case
        generations.append(
            SimpleNamespace(
                id=index,
                case_id=index,
                repeat_idx=0,
                output=long_output,
                outcome=FailureOutcome.PASSED.value,
                total_ms=12.0,
                trace_id=None,
                finish_reason="stop",
                cost_usd=0.0,
                attempts=1,
                cached=False,
            )
        )
        for score in scores:
            grouped[(score.metric_name, OVERALL_SLICE)].append(score)
            score_rows.append(
                SimpleNamespace(
                    generation_id=index,
                    metric_name=score.metric_name,
                    value=score.value,
                    passed=score.passed,
                )
            )

    aggregates = []
    for (metric_name, slice_key), values in sorted(grouped.items()):
        aggregate = engine.registry.get(metric_name).aggregate(values)
        aggregates.append(
            SimpleNamespace(
                metric_name=aggregate.metric_name,
                metric_version=aggregate.metric_version,
                metric_config_sha256=values[0].metric_config_sha256,
                slice_key=slice_key,
                n=aggregate.n,
                value=aggregate.value,
                ci_low=aggregate.ci_low,
                ci_high=aggregate.ci_high,
                method=aggregate.method,
            )
        )

    common = dict(
        run_id="00000000-0000-4000-8000-00000000p401",
        config_sha256="c" * 64,
        model_digest="mock-digest-abc123",
        dataset_sha256=bundle.content_sha256,
        model={
            "provider": "mock",
            "model": "mock-model",
            "resolved_version": "mock-digest-abc123",
            "quantization": "Q4",
            "params_b": None,
            "context_window": None,
            "capabilities": {},
        },
        dataset={
            "name": bundle.manifest.name,
            "version": bundle.manifest.version,
            "split": bundle.manifest.split,
            "content_sha256": bundle.content_sha256,
            "case_count": len(bundle.cases),
            "license": bundle.manifest.license,
            "pii_scrubbed": bundle.manifest.pii_scrubbed,
            "slice_dimensions": list(bundle.manifest.slices),
        },
        prompt_template={
            "name": template_path.stem,
            "version": "1",
            "content_sha256": sha256_hex(template_body),
            "body": template_body[:720],
        },
        decode_params={"temperature": 0.0, "max_tokens": 64, "seed": 42},
        planned_generations=len(bundle.cases),
        generations=generations,
        scores=score_rows,
        aggregates=aggregates,
        cases=cases_by_id,
        coverage_floor=0.98,
        primary_metric=primary_metric,
    )
    return common, metric_names


@pytest.mark.parametrize("dataset_path", smoke_paths(), ids=lambda path: path.name)
def test_assemble_run_report_publishable_gate_and_schema(
    dataset_path: Path,
) -> None:
    common, metric_names = _report_inputs(dataset_path)

    publishable = assemble_run_report(run_status="completed", **common)
    blocked = assemble_run_report(run_status="running", **common)
    under_covered = assemble_run_report(
        run_status="completed",
        **{**common, "planned_generations": len(common["generations"]) + 1},
    )
    payload = report_to_json(publishable)

    assert payload["schema_version"] == "2.1"
    assert publishable.publishable is True
    assert blocked.publishable is False
    assert under_covered.publishable is False
    assert under_covered.coverage < 0.98
    overall = {
        row["metric"]: row["value"]
        for row in payload["metric_aggregates"]
        if row["slice"] == OVERALL_SLICE
    }
    assert overall == {name: 1.0 for name in metric_names}
    if dataset_path.name == "synthetic-retrieval-smoke":
        assert publishable.headline_kind == "mean"
        assert publishable.pass_rate == pytest.approx(1.0)
        assert publishable.pass_rate_ci == (None, None)
    else:
        assert publishable.headline_kind == "pass_rate"
        assert publishable.pass_rate == pytest.approx(1.0)
    assert payload["case_examples"]
    for example in payload["case_examples"]:
        assert "raw_response" not in example
        assert example["output"] is not None
        assert example["output"].endswith("…")
        assert len(example["output"]) == EXAMPLE_TEXT_LIMIT + 1


@pytest.mark.parametrize("dataset_path", smoke_paths(), ids=lambda path: path.name)
def test_assemble_run_report_blocks_publish_when_harness_failures_sink_coverage(
    dataset_path: Path,
) -> None:
    common, _ = _report_inputs(dataset_path)
    generations = list(common["generations"])
    harness_outcomes = (FailureOutcome.HARNESS_ERROR.value, FailureOutcome.HARNESS_TIMEOUT.value)
    assert len(generations) > len(harness_outcomes)
    degraded = [
        SimpleNamespace(**{**vars(generation), "outcome": harness_outcomes[index]})
        if index < len(harness_outcomes)
        else generation
        for index, generation in enumerate(generations)
    ]
    expected_coverage = (len(generations) - len(harness_outcomes)) / len(generations)

    report = assemble_run_report(
        run_status="completed",
        **{**common, "generations": degraded},
    )

    assert report.publishable is False
    # Only the coverage conjunct may block: status and written == planned both hold.
    assert report.run_status == "completed"
    assert report.written_generations == report.planned_generations
    assert report.harness_failures == len(harness_outcomes)
    assert report.coverage == pytest.approx(expected_coverage)
    assert report.coverage < report.coverage_floor


@pytest.mark.parametrize("dataset_path", smoke_paths(), ids=lambda path: path.name)
def test_assemble_run_report_blocks_publish_when_written_below_planned_at_adequate_coverage(
    dataset_path: Path,
) -> None:
    common, _ = _report_inputs(dataset_path)
    written = len(common["generations"])
    planned = written + 1
    expected_coverage = written / planned

    report = assemble_run_report(
        run_status="completed",
        **{**common, "planned_generations": planned, "coverage_floor": expected_coverage},
    )

    assert report.publishable is False
    # Only the written == planned conjunct may block: status holds and coverage clears the floor.
    assert report.run_status == "completed"
    assert report.harness_failures == 0
    assert report.written_generations == written
    assert report.coverage == pytest.approx(expected_coverage)
    assert report.coverage >= report.coverage_floor


def test_assemble_run_report_retrieval_smoke_headlines_mean_not_false_pass_rate() -> None:
    """Wrong rankings score NDCG 0 with passed=True (threshold 0); headline must not be 100%."""
    dataset_path = SMOKE_ROOT / "synthetic-retrieval-smoke"
    bundle = load_dataset(dataset_path)
    assert bundle.manifest.task_metrics == ["retrieval_ndcg_10"]
    engine = ScoringEngine()
    metric_names = list(bundle.manifest.task_metrics)

    generations: list[SimpleNamespace] = []
    score_rows: list[SimpleNamespace] = []
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    cases_by_id: dict[int, object] = {}

    for index, case in enumerate(bundle.cases, start=1):
        # Irrelevant ranking: every relevant doc is absent → NDCG 0.0, passed True.
        generation = replace(perfect_generation(case), output='["unrelated-doc"]')
        scores = engine.score_one(generation, case, metric_names)
        assert all(score.metric_name == "retrieval_ndcg_10" for score in scores)
        assert all(score.value == 0.0 for score in scores)
        assert all(score.passed is True for score in scores)

        cases_by_id[index] = case
        generations.append(
            SimpleNamespace(
                id=index,
                case_id=index,
                repeat_idx=0,
                output=generation.output,
                outcome=FailureOutcome.PASSED.value,
                total_ms=12.0,
                trace_id=None,
                finish_reason="stop",
                cost_usd=0.0,
                attempts=1,
                cached=False,
            )
        )
        for score in scores:
            grouped[(score.metric_name, OVERALL_SLICE)].append(score)
            score_rows.append(
                SimpleNamespace(
                    generation_id=index,
                    metric_name=score.metric_name,
                    value=score.value,
                    passed=score.passed,
                )
            )

    aggregates = []
    for (metric_name, slice_key), values in sorted(grouped.items()):
        aggregate = engine.registry.get(metric_name).aggregate(values)
        aggregates.append(
            SimpleNamespace(
                metric_name=aggregate.metric_name,
                metric_version=aggregate.metric_version,
                metric_config_sha256=values[0].metric_config_sha256,
                slice_key=slice_key,
                n=aggregate.n,
                value=aggregate.value,
                ci_low=aggregate.ci_low,
                ci_high=aggregate.ci_high,
                method=aggregate.method,
            )
        )

    report = assemble_run_report(
        run_id="00000000-0000-4000-8000-00000000ret1",
        run_status="completed",
        config_sha256="c" * 64,
        model_digest="mock-digest",
        dataset_sha256=bundle.content_sha256,
        model={
            "provider": "mock",
            "model": "mock-model",
            "resolved_version": "mock-digest",
            "quantization": None,
            "params_b": None,
            "context_window": None,
            "capabilities": {},
        },
        dataset={
            "name": bundle.manifest.name,
            "version": bundle.manifest.version,
            "split": bundle.manifest.split,
            "content_sha256": bundle.content_sha256,
            "case_count": len(bundle.cases),
            "license": bundle.manifest.license,
            "pii_scrubbed": bundle.manifest.pii_scrubbed,
            "slice_dimensions": list(bundle.manifest.slices),
        },
        prompt_template={
            "name": "retrieval",
            "version": "1",
            "content_sha256": "e" * 64,
            "body": "{{ query }}",
        },
        decode_params={"temperature": 0.0},
        planned_generations=len(bundle.cases),
        generations=generations,
        scores=score_rows,
        aggregates=aggregates,
        cases=cases_by_id,
        coverage_floor=0.98,
        primary_metric="retrieval_ndcg_10",
    )

    assert report.headline_kind == "mean"
    assert report.pass_rate == pytest.approx(0.0)
    assert report.pass_rate_n == len(bundle.cases)
    assert report.pass_rate_ci == (None, None)
    assert report.cost_per_correct is None


def _request_for(prompt: str) -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content=prompt)],
        max_tokens=64,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=42,
        stop=[],
        response_format=None,
        tools=None,
        timeout_s=5.0,
    )
