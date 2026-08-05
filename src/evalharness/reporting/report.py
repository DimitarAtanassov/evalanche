"""Versioned, read-only multi-audience report generation."""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, PackageLoader, select_autoescape

from evalharness.core.enums import FailureOutcome
from evalharness.statistics import wilson_interval
from evalharness.store.db import session_scope
from evalharness.store.models import DatasetRow, ModelVersionRow
from evalharness.store.repository import RunRepository

SCHEMA_VERSION = "1.0"
HARNESS_OUTCOMES = {FailureOutcome.HARNESS_ERROR.value, FailureOutcome.HARNESS_TIMEOUT.value}
LATENCY_CHART_DIV_ID = "chart-latency-percentiles"
_templates = Environment(
    loader=PackageLoader("evalharness.reporting", "templates"),
    autoescape=select_autoescape(["html"]),
)


@dataclass
class RunReport:
    schema_version: str
    run_id: str
    run_status: str
    config_sha256: str
    model_digest: str
    dataset_sha256: str
    coverage: float
    planned_generations: int
    written_generations: int
    coverage_floor: float
    publishable: bool
    pass_rate: float
    pass_rate_ci: tuple[float, float]
    outcome_histogram: dict[str, int]
    latency: dict[str, float]
    finish_reasons: dict[str, int]
    metric_aggregates: list[dict[str, Any]]
    trace_ids_sample: list[str]
    views: dict[str, Any]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def build_report(run_id: uuid.UUID, coverage_floor: float = 0.98) -> RunReport:
    """Read stored scores/aggregates; reporting never mutates evaluation state."""
    async with session_scope() as session:
        repo = RunRepository(session)
        run = await repo.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        dataset = await session.get(DatasetRow, run.dataset_id)
        model = await session.get(ModelVersionRow, run.model_version_id)
        generations = await repo.get_generations_for_run(run_id)
        scores = await repo.get_scores_for_run(run_id)
        aggregates = await repo.get_metric_aggregates(run_id)
        planned = await repo.get_planned_generation_count(run_id)

    harness_failures = sum(row.outcome in HARNESS_OUTCOMES for row in generations)
    covered = max(0, len(generations) - harness_failures)
    coverage = covered / planned if planned else 0.0
    exact = [
        score for score in scores if score.metric_name == "exact_match" and score.passed is not None
    ]
    passed = sum(bool(score.passed) for score in exact)
    pass_rate = passed / len(exact) if exact else 0.0
    latencies = [float(row.total_ms) for row in generations if row.total_ms is not None]
    latency = {
        key: round(value, 2)
        for key, value in {
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
        }.items()
    }
    total_cost = sum(float(row.cost_usd or 0) for row in generations)
    retries = sum(max(0, row.attempts - 1) for row in generations)
    cached = sum(row.cached for row in generations)
    written = len(generations)
    return RunReport(
        schema_version=SCHEMA_VERSION,
        run_id=str(run_id),
        run_status=run.status,
        config_sha256=run.config_sha256,
        model_digest=model.resolved_version if model else "",
        dataset_sha256=dataset.content_sha256 if dataset else "",
        coverage=coverage,
        planned_generations=planned,
        written_generations=written,
        coverage_floor=coverage_floor,
        publishable=(
            run.status == "completed" and written == planned and coverage >= coverage_floor
        ),
        pass_rate=pass_rate,
        pass_rate_ci=wilson_interval(passed, len(exact)),
        outcome_histogram=dict(sorted(Counter(row.outcome for row in generations).items())),
        latency=latency,
        finish_reasons=dict(
            sorted(Counter(row.finish_reason or "unknown" for row in generations).items())
        ),
        metric_aggregates=[
            {
                "metric": row.metric_name,
                "version": row.metric_version,
                "config_sha256": row.metric_config_sha256,
                "slice": row.slice_key,
                "n": row.n,
                "value": row.value,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "method": row.method,
            }
            for row in aggregates
        ],
        trace_ids_sample=[row.trace_id for row in generations if row.trace_id][:10],
        views={
            "leadership": {
                "cost_per_correct": total_cost / passed if passed else None,
                "coverage": coverage,
            },
            "research": {"confidence_method": "Wilson 95%", "flaky_cases_excluded": True},
            "engineering": {
                "retries": retries,
                "cache_hits": cached,
                "cache_rate": cached / len(generations) if generations else 0.0,
            },
        },
    )


def report_to_json(report: RunReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["pass_rate_ci"] = {
        "low": report.pass_rate_ci[0],
        "high": report.pass_rate_ci[1],
    }
    payload["latency_ms"] = report.latency
    return payload


def _render_chart(figure: go.Figure, div_id: str, *, inline_plotlyjs: bool) -> str:
    """Render a figure to a div.

    ``div_id`` must be supplied by the caller: plotly generates a random UUID when it is
    omitted, which makes the HTML report non-reproducible.
    """
    return cast(
        str,
        pio.to_html(
            figure,
            include_plotlyjs="inline" if inline_plotlyjs else False,
            full_html=False,
            div_id=div_id,
        ),
    )


def _latency_figure(report: RunReport) -> go.Figure:
    figure = go.Figure(
        data=[go.Bar(x=list(report.latency), y=list(report.latency.values()), name="Latency")]
    )
    figure.update_layout(title="Latency percentiles", xaxis_title="Statistic", yaxis_title="ms")
    return figure


def report_to_html(report: RunReport) -> str:
    chart = _render_chart(_latency_figure(report), LATENCY_CHART_DIV_ID, inline_plotlyjs=True)
    rendered = _templates.get_template("report_v1.html.j2").render(
        report=report_to_json(report), plot=chart
    )
    return "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"


def report_to_junit(report: RunReport) -> str:
    suite = ET.Element(
        "testsuite",
        name="evalanche",
        tests="2",
        failures=str(int(not report.publishable)),
    )
    coverage = ET.SubElement(suite, "testcase", name="coverage")
    if not report.publishable:
        failure = ET.SubElement(coverage, "failure", message="coverage gate failed")
        failure.text = f"{report.coverage:.6f} < {report.coverage_floor:.6f}"
    pass_rate = ET.SubElement(suite, "testcase", name="pass_rate")
    ET.SubElement(pass_rate, "system-out").text = f"{report.pass_rate:.6f}"
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)


async def write_report(
    run_id: uuid.UUID, output_dir: Path, coverage_floor: float = 0.98
) -> RunReport:
    report = await build_report(run_id, coverage_floor)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / str(run_id)
    stem.with_suffix(".json").write_text(
        json.dumps(report_to_json(report), indent=2), encoding="utf-8"
    )
    stem.with_suffix(".html").write_text(report_to_html(report), encoding="utf-8")
    stem.with_suffix(".xml").write_text(report_to_junit(report), encoding="utf-8")
    return report
