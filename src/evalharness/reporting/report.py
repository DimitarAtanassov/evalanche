"""Stable reporting façade: assemble, I/O, and render stay importable from one module.

Implementation lives in ``assemble``, ``io``, and ``render``. Callers keep importing
from ``evalharness.reporting.report`` (and the package root) so Wave-1 split does not
churn every import site.
"""

from __future__ import annotations

from evalharness.reporting.assemble import (
    EXAMPLE_LIMIT,
    EXAMPLE_TEXT_LIMIT,
    HARNESS_OUTCOMES,
    HEADLINE_MEAN,
    HEADLINE_PASS_RATE,
    PRIMARY_METRIC,
    SCHEMA_VERSION,
    RunReport,
    _dataset_context,
    _model_context,
    _prompt_context,
    _stable_decode_params,
    assemble_run_report,
)
from evalharness.reporting.io import build_report, write_report
from evalharness.reporting.render import (
    LATENCY_CHART_DIV_ID,
    LATENCY_PERCENTILES,
    METRIC_CHART_DIV_ID,
    OUTCOME_CHART_DIV_ID,
    SLICE_CHART_DIV_ID,
    overall_aggregates,
    report_to_html,
    report_to_json,
    report_to_junit,
    slice_aggregates,
)

__all__ = [
    "EXAMPLE_LIMIT",
    "EXAMPLE_TEXT_LIMIT",
    "HARNESS_OUTCOMES",
    "HEADLINE_MEAN",
    "HEADLINE_PASS_RATE",
    "LATENCY_CHART_DIV_ID",
    "LATENCY_PERCENTILES",
    "METRIC_CHART_DIV_ID",
    "OUTCOME_CHART_DIV_ID",
    "PRIMARY_METRIC",
    "SCHEMA_VERSION",
    "SLICE_CHART_DIV_ID",
    "RunReport",
    "_dataset_context",
    "_model_context",
    "_prompt_context",
    "_stable_decode_params",
    "assemble_run_report",
    "build_report",
    "overall_aggregates",
    "report_to_html",
    "report_to_json",
    "report_to_junit",
    "slice_aggregates",
    "write_report",
]
