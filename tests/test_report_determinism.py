"""Guards against non-reproducible report artifacts."""

from __future__ import annotations

import re

from evalharness.reporting.report import (
    LATENCY_CHART_DIV_ID,
    RunReport,
    report_to_html,
)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RUN_ID = "00000000-0000-4000-8000-0000000000c1"


def _report() -> RunReport:
    return RunReport(
        schema_version="1.0",
        run_id=RUN_ID,
        run_status="completed",
        config_sha256="c" * 64,
        model_digest="mock-digest",
        dataset_sha256="d" * 64,
        coverage=1.0,
        planned_generations=5,
        written_generations=5,
        coverage_floor=0.98,
        publishable=True,
        pass_rate=1.0,
        pass_rate_ci=(0.5, 1.0),
        outcome_histogram={"passed": 5},
        latency={"p50": 5.0, "p90": 5.0, "p95": 5.0, "p99": 5.0, "max": 5.0, "mean": 5.0},
        finish_reasons={"stop": 5},
        metric_aggregates=[],
        trace_ids_sample=[],
        views={
            "leadership": {"cost_per_correct": 0.0, "coverage": 1.0},
            "research": {"confidence_method": "Wilson 95%", "flaky_cases_excluded": True},
            "engineering": {"retries": 0, "cache_hits": 0, "cache_rate": 0.0},
        },
    )


def test_report_html_is_byte_identical_across_renders() -> None:
    assert report_to_html(_report()) == report_to_html(_report())


def test_report_html_uses_stable_chart_div_id() -> None:
    html = report_to_html(_report())
    assert f'id="{LATENCY_CHART_DIV_ID}"' in html
    # Plotly falls back to a random UUID div id when div_id is not supplied.
    assert set(UUID_RE.findall(html)) == {RUN_ID}
