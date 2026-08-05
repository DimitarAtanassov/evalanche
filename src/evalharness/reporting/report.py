"""Report generation."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jinja2 import Template

from evalharness.core.enums import FailureOutcome
from evalharness.core.models import ScoreValue
from evalharness.scoring.exact_match import ExactMatchMetric
from evalharness.scoring.normalizer import Normalizer, NormalizerConfig
from evalharness.scoring.stats import percentile, wilson_interval
from evalharness.store.db import session_scope
from evalharness.store.models import (
    DatasetRow,
    ModelVersionRow,
)
from evalharness.store.repository import RunRepository

HARNESS_OUTCOMES = {FailureOutcome.HARNESS_ERROR, FailureOutcome.HARNESS_TIMEOUT}


@dataclass
class RunReport:
    run_id: str
    config_sha256: str
    model_digest: str
    dataset_sha256: str
    coverage: float
    coverage_floor: float
    publishable: bool
    pass_rate: float
    pass_rate_ci: tuple[float, float]
    outcome_histogram: dict[str, int]
    latency: dict[str, float]
    finish_reasons: dict[str, int]
    metric_aggregates: list[dict[str, Any]]
    trace_ids_sample: list[str]


HTML_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Eval Report {{ run_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #111; }
    h1, h2 { margin-bottom: 0.5rem; }
    table { border-collapse: collapse; margin: 1rem 0; }
    th, td { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }
    .warn { color: #a00; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <p><strong>Run ID:</strong> {{ run_id }}</p>
  <p><strong>Config SHA256:</strong> {{ config_sha256 }}</p>
  <p><strong>Model digest:</strong> {{ model_digest }}</p>
  <p><strong>Dataset SHA256:</strong> {{ dataset_sha256 }}</p>
  {% if not publishable %}
  <p class="warn">Coverage {{ "%.2f"|format(coverage) }} is below floor {{ coverage_floor }} — report withheld.</p>
  {% else %}
  <h2>Pass Rate</h2>
  <p>{{ "%.2f%%"|format(pass_rate * 100) }} (95% CI: {{ "%.2f%%"|format(pass_rate_ci[0] * 100) }} – {{ "%.2f%%"|format(pass_rate_ci[1] * 100) }})</p>
  <h2>Latency (ms)</h2>
  <table>
    <tr><th>p50</th><th>p90</th><th>p95</th><th>p99</th><th>max</th><th>mean</th></tr>
    <tr>
      <td>{{ latency.p50 }}</td><td>{{ latency.p90 }}</td><td>{{ latency.p95 }}</td>
      <td>{{ latency.p99 }}</td><td>{{ latency.max }}</td><td>{{ latency.mean }}</td>
    </tr>
  </table>
  <h2>Outcome Histogram</h2>
  <table>
    <tr><th>Outcome</th><th>Count</th></tr>
    {% for outcome, count in outcome_histogram.items() %}
    <tr><td>{{ outcome }}</td><td>{{ count }}</td></tr>
    {% endfor %}
  </table>
  <h2>Finish Reasons</h2>
  <table>
    <tr><th>Reason</th><th>Count</th></tr>
    {% for reason, count in finish_reasons.items() %}
    <tr><td>{{ reason }}</td><td>{{ count }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}
</body>
</html>
"""
)


async def build_report(run_id: uuid.UUID, coverage_floor: float = 0.98) -> RunReport:
    async with session_scope() as session:
        repo = RunRepository(session)
        run = await repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        dataset = await session.get(DatasetRow, run.dataset_id)
        model_version = await session.get(ModelVersionRow, run.model_version_id)
        generations = await repo.get_generations_for_run(run_id)
        scores = await repo.get_scores_for_run(run_id)

        total = len(generations)
        harness_failures = sum(
            1 for g in generations if g.outcome in {o.value for o in HARNESS_OUTCOMES}
        )
        coverage = 1.0 - (harness_failures / total if total else 0.0)

        outcome_hist = Counter(g.outcome for g in generations)
        finish_hist = Counter(g.finish_reason or "unknown" for g in generations)

        latencies = [float(g.total_ms) for g in generations if g.total_ms is not None]
        latency_stats = {
            "p50": round(percentile(latencies, 0.50), 2),
            "p90": round(percentile(latencies, 0.90), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "max": round(max(latencies) if latencies else 0.0, 2),
            "mean": round(sum(latencies) / len(latencies) if latencies else 0.0, 2),
        }

        eligible = [g for g in generations if g.outcome not in {o.value for o in HARNESS_OUTCOMES}]
        passed = sum(1 for g in eligible if g.outcome == FailureOutcome.PASSED.value)
        n_eligible = len(eligible)
        pass_rate = passed / n_eligible if n_eligible else 0.0
        ci = wilson_interval(passed, n_eligible)

        metric = ExactMatchMetric(Normalizer(NormalizerConfig()))
        score_values: list[ScoreValue] = []
        for s in scores:
            if s.metric_name == metric.name:
                score_values.append(
                    ScoreValue(
                        metric_name=s.metric_name,
                        metric_version=s.metric_version,
                        metric_config_sha256=s.metric_config_sha256,
                        value=s.value,
                        passed=s.passed,
                        detail=s.detail or {},
                    )
                )
        agg = metric.aggregate(score_values)
        await repo.save_metric_aggregate(
            run_id=run_id,
            metric_name=agg.metric_name,
            metric_version=metric.version,
            slice_key=agg.slice_key,
            n=agg.n,
            value=agg.value,
            ci_low=agg.ci_low,
            ci_high=agg.ci_high,
            stddev=agg.stddev,
            method=agg.method,
        )

        publishable = coverage >= coverage_floor
        trace_ids = [g.trace_id for g in generations if g.trace_id][:10]

        return RunReport(
            run_id=str(run_id),
            config_sha256=run.config_sha256,
            model_digest=model_version.resolved_version if model_version else "",
            dataset_sha256=dataset.content_sha256 if dataset else "",
            coverage=coverage,
            coverage_floor=coverage_floor,
            publishable=publishable,
            pass_rate=pass_rate,
            pass_rate_ci=ci,
            outcome_histogram=dict(outcome_hist),
            latency=latency_stats,
            finish_reasons=dict(finish_hist),
            metric_aggregates=[
                {
                    "metric": agg.metric_name,
                    "version": agg.metric_version
                    if hasattr(agg, "metric_version")
                    else metric.version,
                    "value": agg.value,
                    "ci_low": agg.ci_low,
                    "ci_high": agg.ci_high,
                    "method": agg.method,
                }
            ],
            trace_ids_sample=trace_ids,
        )


def report_to_json(report: RunReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "config_sha256": report.config_sha256,
        "model_digest": report.model_digest,
        "dataset_sha256": report.dataset_sha256,
        "coverage": report.coverage,
        "coverage_floor": report.coverage_floor,
        "publishable": report.publishable,
        "pass_rate": report.pass_rate,
        "pass_rate_ci": {"low": report.pass_rate_ci[0], "high": report.pass_rate_ci[1]},
        "outcome_histogram": report.outcome_histogram,
        "latency_ms": report.latency,
        "finish_reasons": report.finish_reasons,
        "metric_aggregates": report.metric_aggregates,
        "trace_ids_sample": report.trace_ids_sample,
    }


def report_to_html(report: RunReport) -> str:
    return cast(
        str,
        HTML_TEMPLATE.render(
        run_id=report.run_id,
        config_sha256=report.config_sha256,
        model_digest=report.model_digest,
        dataset_sha256=report.dataset_sha256,
        coverage=report.coverage,
        coverage_floor=report.coverage_floor,
        publishable=report.publishable,
        pass_rate=report.pass_rate,
        pass_rate_ci=report.pass_rate_ci,
        outcome_histogram=report.outcome_histogram,
        latency=report.latency,
        finish_reasons=report.finish_reasons,
        ),
    )


async def write_report(
    run_id: uuid.UUID, output_dir: Path, coverage_floor: float = 0.98
) -> RunReport:
    report = await build_report(run_id, coverage_floor=coverage_floor)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    html_path = output_dir / f"{run_id}.html"
    json_path.write_text(json.dumps(report_to_json(report), indent=2), encoding="utf-8")
    html_path.write_text(report_to_html(report), encoding="utf-8")
    return report
