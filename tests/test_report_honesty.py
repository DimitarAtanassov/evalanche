"""Report contract honesty: flaky flag and JUnit publishability clauses."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from evalharness.core.enums import FailureOutcome
from evalharness.core.models import Case, TaskType
from evalharness.reporting.report import assemble_run_report, report_to_junit


def _base_kwargs(**overrides: object) -> dict[str, object]:
    case = Case(
        external_id="c1",
        task_type=TaskType.QA_SHORT,
        inputs={"question": "2+2?"},
        reference_answer="4",
    )
    generation = SimpleNamespace(
        id=1,
        case_id=1,
        repeat_idx=0,
        output="4",
        outcome=FailureOutcome.PASSED.value,
        total_ms=5.0,
        trace_id=None,
        finish_reason="stop",
        cost_usd=0.0,
        attempts=1,
        cached=False,
    )
    score = SimpleNamespace(
        generation_id=1,
        metric_name="exact_match",
        value=1.0,
        passed=True,
    )
    aggregate = SimpleNamespace(
        metric_name="exact_match",
        metric_version="1.0.0",
        metric_config_sha256="a" * 64,
        slice_key="__overall__",
        n=1,
        value=1.0,
        ci_low=0.0,
        ci_high=1.0,
        method="wilson",
    )
    kwargs: dict[str, object] = {
        "run_id": "00000000-0000-4000-8000-0000000000h1",
        "run_status": "completed",
        "config_sha256": "c" * 64,
        "model_digest": "mock-digest",
        "dataset_sha256": "d" * 64,
        "model": {
            "provider": "mock",
            "model": "mock-model",
            "resolved_version": "mock-digest",
        },
        "dataset": {
            "name": "sample",
            "version": "1.0.0",
            "split": "test",
            "content_sha256": "d" * 64,
            "case_count": 1,
        },
        "prompt_template": {
            "name": "default",
            "version": "1",
            "content_sha256": "e" * 64,
            "body": "Answer: {{ question }}",
        },
        "decode_params": {"temperature": 0.0, "max_tokens": 32},
        "planned_generations": 1,
        "generations": [generation],
        "scores": [score],
        "aggregates": [aggregate],
        "cases": {1: case},
        "coverage_floor": 0.98,
        "primary_metric": "exact_match",
    }
    kwargs.update(overrides)
    return kwargs


def test_assemble_run_report_sets_flaky_cases_excluded_false() -> None:
    report = assemble_run_report(**_base_kwargs())  # type: ignore[arg-type]
    assert report.flaky_cases_excluded is False


@pytest.mark.parametrize(
    ("overrides", "expected_clause"),
    [
        ({"run_status": "failed"}, "Run status is completed"),
        (
            {
                "planned_generations": 2,
                "coverage_floor": 0.0,
                "generations": [
                    SimpleNamespace(
                        id=1,
                        case_id=1,
                        repeat_idx=0,
                        output="4",
                        outcome=FailureOutcome.PASSED.value,
                        total_ms=5.0,
                        trace_id=None,
                        finish_reason="stop",
                        cost_usd=0.0,
                        attempts=1,
                        cached=False,
                    )
                ],
            },
            "All planned generations written",
        ),
        (
            {
                "coverage_floor": 1.0,
                "generations": [
                    SimpleNamespace(
                        id=1,
                        case_id=1,
                        repeat_idx=0,
                        output=None,
                        outcome=FailureOutcome.HARNESS_ERROR.value,
                        total_ms=None,
                        trace_id=None,
                        finish_reason=None,
                        cost_usd=None,
                        attempts=1,
                        cached=False,
                    )
                ],
                "scores": [],
                "aggregates": [
                    SimpleNamespace(
                        metric_name="exact_match",
                        metric_version="1.0.0",
                        metric_config_sha256="a" * 64,
                        slice_key="__overall__",
                        n=0,
                        value=0.0,
                        ci_low=None,
                        ci_high=None,
                        method="wilson",
                    )
                ],
            },
            "Coverage \u2265 floor (100%)",
        ),
    ],
)
def test_report_to_junit_names_failing_publishability_clause(
    overrides: dict[str, object],
    expected_clause: str,
) -> None:
    report = assemble_run_report(**_base_kwargs(**overrides))  # type: ignore[arg-type]
    assert report.publishable is False
    xml = report_to_junit(report)
    root = ET.fromstring(xml)
    failure = root.find("./testcase[@name='coverage']/failure")
    assert failure is not None
    assert failure.get("message") == expected_clause


def test_assemble_run_report_sums_known_costs_and_counts_unpriced() -> None:
    priced = SimpleNamespace(
        id=1,
        case_id=1,
        repeat_idx=0,
        output="4",
        outcome=FailureOutcome.PASSED.value,
        total_ms=5.0,
        trace_id=None,
        finish_reason="stop",
        cost_usd=0.25,
        attempts=1,
        cached=False,
    )
    also_priced = SimpleNamespace(
        id=2,
        case_id=1,
        repeat_idx=1,
        output="4",
        outcome=FailureOutcome.PASSED.value,
        total_ms=5.0,
        trace_id=None,
        finish_reason="stop",
        cost_usd=0.10,
        attempts=1,
        cached=False,
    )
    unpriced = SimpleNamespace(
        id=3,
        case_id=1,
        repeat_idx=2,
        output="4",
        outcome=FailureOutcome.PASSED.value,
        total_ms=5.0,
        trace_id=None,
        finish_reason="stop",
        cost_usd=None,
        attempts=1,
        cached=False,
    )
    report = assemble_run_report(
        **_base_kwargs(  # type: ignore[arg-type]
            planned_generations=3,
            generations=[priced, also_priced, unpriced],
            scores=[
                SimpleNamespace(generation_id=1, metric_name="exact_match", value=1.0, passed=True),
                SimpleNamespace(generation_id=2, metric_name="exact_match", value=1.0, passed=True),
                SimpleNamespace(generation_id=3, metric_name="exact_match", value=1.0, passed=True),
            ],
        )
    )
    assert report.cost_usd_total == pytest.approx(0.35)
    assert report.cost_unpriced_generations == 1
