"""Guards against non-reproducible report artifacts."""

from __future__ import annotations

from evalharness.reporting.report import (
    LATENCY_CHART_DIV_ID,
    METRIC_CHART_DIV_ID,
    OUTCOME_CHART_DIV_ID,
    SLICE_CHART_DIV_ID,
    RunReport,
    report_to_html,
    slice_aggregates,
)

RUN_ID = "00000000-0000-4000-8000-0000000000c1"


def _aggregate(slice_key: str, value: float, n: int) -> dict[str, object]:
    return {
        "metric": "exact_match",
        "version": "1.0.0",
        "config_sha256": "a" * 64,
        "slice": slice_key,
        "n": n,
        "value": value,
        "ci_low": max(0.0, value - 0.1),
        "ci_high": min(1.0, value + 0.1),
        "method": "wilson",
    }


def _report() -> RunReport:
    return RunReport(
        schema_version="2.2",
        run_id=RUN_ID,
        run_status="completed",
        config_sha256="c" * 64,
        model_digest="mock-digest",
        dataset_sha256="d" * 64,
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
            "name": "sample",
            "version": "1.0.0",
            "split": "test",
            "content_sha256": "d" * 64,
            "case_count": 5,
            "license": None,
            "pii_scrubbed": None,
            "slice_dimensions": ["difficulty", "lang"],
        },
        prompt_template={
            "name": "default",
            "version": "1",
            "content_sha256": "e" * 64,
            "body": "Answer: {{ input }}",
        },
        decode_params={"max_tokens": 64, "temperature": 0.0},
        coverage=1.0,
        planned_generations=5,
        written_generations=5,
        coverage_floor=0.98,
        publishable=True,
        primary_metric="exact_match",
        headline_kind="pass_rate",
        pass_rate=0.8,
        pass_rate_n=5,
        pass_rate_ci=(0.5, 1.0),
        confidence_method="Wilson 95%",
        flaky_cases_excluded=True,
        outcome_histogram={"passed": 4, "harness_error": 1},
        harness_failures=1,
        latency={"p50": 5.0, "p90": 5.0, "p95": 5.0, "p99": 5.0, "max": 5.0, "mean": 5.0},
        finish_reasons={"stop": 5},
        metric_aggregates=[
            _aggregate("__overall__", 0.8, 5),
            _aggregate("difficulty=easy", 0.9, 3),
            _aggregate("difficulty=hard", 0.6, 2),
        ],
        case_examples=[
            {
                "case_id": "case-1",
                "repeat_idx": 0,
                "task_type": "generation",
                "slices": {"difficulty": "hard"},
                "input": "What is 2+2?",
                "reference": "4",
                "output": "four",
                "outcome": "passed",
                "metric": "exact_match",
                "metric_value": 0.0,
                "passed": False,
                "latency_ms": 5.0,
                "trace_id": None,
            }
        ],
        cost_usd_total=0.0,
        cost_unpriced_generations=0,
        cost_per_correct=0.0,
        retries=0,
        cache_hits=0,
        cache_rate=0.0,
        trace_ids_sample=[],
    )


def test_report_html_is_byte_identical_across_renders() -> None:
    assert report_to_html(_report()) == report_to_html(_report())


def test_report_html_uses_stable_chart_div_ids() -> None:
    html = report_to_html(_report())
    # Altair falls back to a random altair-viz-<uuid> div id when output_div is omitted.
    assert "altair-viz-" not in html
    for div_id in (
        METRIC_CHART_DIV_ID,
        SLICE_CHART_DIV_ID,
        OUTCOME_CHART_DIV_ID,
        LATENCY_CHART_DIV_ID,
    ):
        assert f'id="{div_id}"' in html


def test_report_html_is_offline_self_contained() -> None:
    html = report_to_html(_report())
    assert 'src="http' not in html


def test_slice_aggregates_are_ordered_worst_first() -> None:
    rows = slice_aggregates(_report())
    assert [row["slice"] for row in rows] == ["difficulty=hard", "difficulty=easy"]


def test_report_html_includes_run_context() -> None:
    html = report_to_html(_report())
    assert "What was evaluated, and on what?" in html
    assert "mock-model" in html
    assert "What is 2+2?" in html
    assert "Sampled inputs/outputs" in html


def test_sections_without_data_are_omitted() -> None:
    report = _report()
    report.metric_aggregates = []
    html = report_to_html(report)
    assert f'id="{SLICE_CHART_DIV_ID}"' not in html
    assert f'id="{METRIC_CHART_DIV_ID}"' not in html
