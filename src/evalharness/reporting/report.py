"""Versioned, read-only run report generation.

The JSON artifact is the contract; the HTML dashboard is a view over it. Both are
byte-reproducible for a given run so the committed golden fixtures stay meaningful.
"""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import altair as alt
import vl_convert
from jinja2 import Environment, PackageLoader, select_autoescape

from evalharness.core.enums import FailureOutcome
from evalharness.core.models import Case
from evalharness.observability import (
    PipelineStage,
    ProgressCallback,
    ProgressEvent,
    StageTimer,
    emit_progress,
    get_logger,
    log_context,
)
from evalharness.scoring.engine import OVERALL_SLICE
from evalharness.scoring.registry import MetricRegistry
from evalharness.statistics import wilson_interval
from evalharness.store.db import session_scope
from evalharness.store.models import DatasetRow, ModelVersionRow, PromptTemplateRow
from evalharness.store.repository import RunRepository

SCHEMA_VERSION = "2.1"
PRIMARY_METRIC = "exact_match"
HARNESS_OUTCOMES = {FailureOutcome.HARNESS_ERROR.value, FailureOutcome.HARNESS_TIMEOUT.value}
EXAMPLE_LIMIT = 8
EXAMPLE_TEXT_LIMIT = 280
# Headline kinds: Bernoulli metrics publish a pass rate; continuous metrics
# (explicit threshold <= 0, e.g. retrieval_ndcg_10) publish the overall mean.
HEADLINE_PASS_RATE = "pass_rate"
HEADLINE_MEAN = "mean"

METRIC_CHART_DIV_ID = "chart-metric-scores"
SLICE_CHART_DIV_ID = "chart-slice-pass-rate"
OUTCOME_CHART_DIV_ID = "chart-outcome-breakdown"
LATENCY_CHART_DIV_ID = "chart-latency-percentiles"

LATENCY_PERCENTILES = ("p50", "p90", "p95", "p99", "max")

INK = "#111820"
MUTED = "#5b6675"
LINE = "#e4e8ee"
ACCENT = "#2f5bd7"
BAD = "#b42318"
WARN = "#b25e09"
FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"

_EMBED_OPTIONS = {"actions": False, "renderer": "svg"}
_CHART_THEME: dict[str, Any] = {
    "font": FONT,
    "background": "transparent",
    "view": {"stroke": None},
    "axis": {
        "labelColor": MUTED,
        "titleColor": MUTED,
        "labelFontSize": 11,
        "titleFontSize": 12,
        "titleFontWeight": "normal",
        "titlePadding": 10,
        "gridColor": LINE,
        "domainColor": LINE,
        "tickColor": LINE,
    },
    "legend": {
        "labelColor": MUTED,
        "titleColor": MUTED,
        "labelFontSize": 11,
        "labelLimit": 260,
        "orient": "top",
        "direction": "horizontal",
        "symbolType": "square",
    },
    "bar": {"color": ACCENT},
}

_templates = Environment(
    loader=PackageLoader("evalharness.reporting", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
)
logger = get_logger(__name__)


@dataclass
class RunReport:
    schema_version: str
    run_id: str
    run_status: str
    config_sha256: str
    model_digest: str
    dataset_sha256: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    prompt_template: dict[str, Any]
    decode_params: dict[str, Any]
    coverage: float
    planned_generations: int
    written_generations: int
    coverage_floor: float
    publishable: bool
    primary_metric: str
    headline_kind: str
    pass_rate: float | None
    pass_rate_n: int
    pass_rate_ci: tuple[float | None, float | None]
    confidence_method: str
    flaky_cases_excluded: bool
    outcome_histogram: dict[str, int]
    harness_failures: int
    latency: dict[str, float]
    finish_reasons: dict[str, int]
    metric_aggregates: list[dict[str, Any]]
    case_examples: list[dict[str, Any]]
    cost_usd_total: float
    cost_per_correct: float | None
    retries: int
    cache_hits: int
    cache_rate: float
    trace_ids_sample: list[str]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _truncate(value: str | None, limit: int = EXAMPLE_TEXT_LIMIT) -> str | None:
    """Single-line text at most ``limit`` chars, counting the ellipsis inside the cap.

    Matches ``observability.sanitize_text`` and the suite, judge, and RAG bounds so a
    published limit means the same number of characters everywhere.
    """
    if value is None:
        return None
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _format_inputs(inputs: dict[str, Any]) -> str:
    if not inputs:
        return ""
    if len(inputs) == 1:
        only = next(iter(inputs.values()))
        return _truncate(str(only)) or ""
    rendered = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _truncate(rendered) or ""


def _reference_text(case: Case) -> str | None:
    if case.reference_answer is not None:
        return _truncate(case.reference_answer)
    if case.expected_label is not None:
        return _truncate(case.expected_label)
    if case.expected_json is not None:
        return _truncate(json.dumps(case.expected_json, sort_keys=True, ensure_ascii=False))
    if case.references:
        return _truncate("; ".join(str(item) for item in case.references))
    return None


def _model_context(model: ModelVersionRow | None) -> dict[str, Any]:
    if model is None:
        return {
            "provider": "",
            "model": "",
            "resolved_version": "",
            "quantization": None,
            "params_b": None,
            "context_window": None,
            "capabilities": {},
        }
    return {
        "provider": model.provider,
        "model": model.model,
        "resolved_version": model.resolved_version,
        "quantization": model.quantization,
        "params_b": model.params_b,
        "context_window": model.context_window,
        "capabilities": dict(model.capabilities or {}),
    }


def _dataset_context(dataset: DatasetRow | None, case_count: int) -> dict[str, Any]:
    if dataset is None:
        return {
            "name": "",
            "version": "",
            "split": "",
            "content_sha256": "",
            "case_count": case_count,
            "license": None,
            "pii_scrubbed": None,
            "slice_dimensions": [],
        }
    manifest = dataset.manifest or {}
    slice_dimensions = manifest.get("slices") or []
    if not isinstance(slice_dimensions, list):
        slice_dimensions = list(slice_dimensions)
    return {
        "name": dataset.name,
        "version": dataset.version,
        "split": dataset.split,
        "content_sha256": dataset.content_sha256,
        "case_count": case_count,
        "license": manifest.get("license"),
        "pii_scrubbed": manifest.get("pii_scrubbed"),
        "slice_dimensions": [str(item) for item in slice_dimensions],
    }


def _prompt_context(template: PromptTemplateRow | None) -> dict[str, Any]:
    if template is None:
        return {"name": "", "version": "", "content_sha256": "", "body": ""}
    return {
        "name": template.name,
        "version": template.version,
        "content_sha256": template.content_sha256,
        "body": _truncate(template.body, limit=720) or "",
    }


def _stable_decode_params(decode_params: dict[str, Any] | None) -> dict[str, Any]:
    """Stable key order so JSON artifacts stay byte-reproducible."""
    if not decode_params:
        return {}
    return cast(dict[str, Any], json.loads(json.dumps(decode_params, sort_keys=True)))


def _case_examples(
    *,
    generations: list[Any],
    cases: dict[int, Case],
    scores: list[Any],
    primary_metric: str,
    limit: int = EXAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Bounded examples for the dashboard.

    Failures lead so a reader immediately sees what went wrong. Text is truncated and
    raw provider payloads are never included — lean reports stay lean.
    """
    primary_by_generation = {
        score.generation_id: score for score in scores if score.metric_name == primary_metric
    }
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for generation in generations:
        case = cases.get(generation.case_id)
        if case is None:
            continue
        score = primary_by_generation.get(generation.id)
        passed = None if score is None else score.passed
        if generation.outcome in HARNESS_OUTCOMES:
            priority = 0
        elif passed is False:
            priority = 1
        elif passed is True:
            priority = 3
        else:
            priority = 2
        candidates.append(
            (
                priority,
                generation.case_id,
                generation.repeat_idx,
                {
                    "case_id": case.external_id,
                    "repeat_idx": generation.repeat_idx,
                    "task_type": case.task_type.value,
                    "slices": dict(sorted((case.slices or {}).items())),
                    "input": _format_inputs(dict(case.inputs or {})),
                    "reference": _reference_text(case),
                    "output": _truncate(generation.output),
                    "outcome": generation.outcome,
                    "metric": primary_metric if score is not None else None,
                    "metric_value": score.value if score is not None else None,
                    "passed": passed,
                    "latency_ms": generation.total_ms,
                    "trace_id": generation.trace_id,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]


def _primary_uses_pass_rate(metric_name: str) -> bool:
    """True when ``passed`` is a meaningful quality gate for the headline.

    Metrics that set an explicit ``threshold <= 0`` (notably ``retrieval_ndcg_10``)
    mark every scored case as passed, so a Bernoulli pass rate is not a quality
    signal. Those primaries headline the overall aggregate mean instead.
    """
    try:
        metric = MetricRegistry.defaults().get(metric_name)
    except ValueError:
        return True
    config = getattr(metric, "config", None)
    if not isinstance(config, dict) or "threshold" not in config:
        return True
    return float(config["threshold"]) > 0.0


def _headline_quality(
    *,
    primary_metric: str,
    scores: list[Any],
    aggregates: list[Any],
) -> tuple[str, float | None, int, tuple[float | None, float | None], str, int]:
    """Return headline_kind, value, n, CI, confidence_method, and Bernoulli pass count."""
    primary = [
        score
        for score in scores
        if score.metric_name == primary_metric and score.passed is not None
    ]
    passed = sum(bool(score.passed) for score in primary)
    if _primary_uses_pass_rate(primary_metric):
        rate = passed / len(primary) if primary else 0.0
        return (
            HEADLINE_PASS_RATE,
            rate,
            len(primary),
            wilson_interval(passed, len(primary)),
            "Wilson 95%",
            passed,
        )
    overall = next(
        (
            row
            for row in aggregates
            if row.metric_name == primary_metric and row.slice_key == OVERALL_SLICE
        ),
        None,
    )
    if overall is None:
        return HEADLINE_MEAN, None, 0, (None, None), "mean (continuous primary)", 0
    return (
        HEADLINE_MEAN,
        float(overall.value),
        int(overall.n),
        (None, None),
        f"mean of {primary_metric}",
        0,
    )


def assemble_run_report(
    *,
    run_id: str,
    run_status: str,
    config_sha256: str,
    model_digest: str,
    dataset_sha256: str,
    model: dict[str, Any],
    dataset: dict[str, Any],
    prompt_template: dict[str, Any],
    decode_params: dict[str, Any],
    planned_generations: int,
    generations: list[Any],
    scores: list[Any],
    aggregates: list[Any],
    cases: dict[int, Case],
    coverage_floor: float = 0.98,
    primary_metric: str = PRIMARY_METRIC,
) -> RunReport:
    """Assemble a versioned run report from already-loaded run artifacts.

    Public pure seam for publishability, coverage, example truncation, and
    aggregate packaging. ``build_report`` loads rows then delegates here.
    Generation/score/aggregate objects are attribute-duck-typed (ORM rows or
    plain namespaces).

    The headline quality field is a Bernoulli pass rate when the primary metric
    has a positive threshold; continuous primaries (threshold ``<= 0``) publish
    the overall aggregate mean instead, with no invented pass-rate CI.
    """
    harness_failures = sum(row.outcome in HARNESS_OUTCOMES for row in generations)
    covered = max(0, len(generations) - harness_failures)
    coverage = covered / planned_generations if planned_generations else 0.0
    headline_kind, pass_rate, pass_rate_n, pass_rate_ci, confidence_method, passed = (
        _headline_quality(
            primary_metric=primary_metric,
            scores=scores,
            aggregates=aggregates,
        )
    )
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
    case_examples = _case_examples(
        generations=generations,
        cases=cases,
        scores=scores,
        primary_metric=primary_metric,
    )
    return RunReport(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_status=run_status,
        config_sha256=config_sha256,
        model_digest=model_digest,
        dataset_sha256=dataset_sha256,
        model=model,
        dataset=dataset,
        prompt_template=prompt_template,
        decode_params=decode_params,
        coverage=coverage,
        planned_generations=planned_generations,
        written_generations=written,
        coverage_floor=coverage_floor,
        publishable=(
            run_status == "completed"
            and written == planned_generations
            and coverage >= coverage_floor
        ),
        primary_metric=primary_metric,
        headline_kind=headline_kind,
        pass_rate=pass_rate,
        pass_rate_n=pass_rate_n,
        pass_rate_ci=pass_rate_ci,
        confidence_method=confidence_method,
        flaky_cases_excluded=True,
        outcome_histogram=dict(sorted(Counter(row.outcome for row in generations).items())),
        harness_failures=harness_failures,
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
        case_examples=case_examples,
        cost_usd_total=total_cost,
        cost_per_correct=(
            total_cost / passed if headline_kind == HEADLINE_PASS_RATE and passed else None
        ),
        retries=retries,
        cache_hits=cached,
        cache_rate=cached / len(generations) if generations else 0.0,
        trace_ids_sample=list(dict.fromkeys(row.trace_id for row in generations if row.trace_id))[
            :10
        ],
    )


async def build_report(
    run_id: uuid.UUID,
    coverage_floor: float = 0.98,
    primary_metric: str = PRIMARY_METRIC,
) -> RunReport:
    """Read stored scores/aggregates; reporting never mutates evaluation state.

    ``primary_metric`` names the metric the headline quality number is computed from.
    Bernoulli primaries publish a pass rate; continuous primaries (threshold ``<= 0``)
    publish the overall aggregate mean. Callers that scored a task-fit metric list must
    pass its head, or the headline reports zero observations against a metric the run
    never scored.
    """
    timer = StageTimer()
    logger.info(
        "report_build_started",
        run_id=str(run_id),
        coverage_floor=coverage_floor,
        primary_metric=primary_metric,
    )
    async with session_scope() as session:
        repo = RunRepository(session)
        run = await repo.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        dataset = await session.get(DatasetRow, run.dataset_id)
        model = await session.get(ModelVersionRow, run.model_version_id)
        template = await session.get(PromptTemplateRow, run.prompt_template_id)
        cases = {
            case_id: case for case_id, case in await repo.get_cases_for_dataset(run.dataset_id)
        }
        generations = await repo.get_generations_for_run(run_id)
        scores = await repo.get_scores_for_run(run_id)
        aggregates = await repo.get_metric_aggregates(run_id)
        planned = await repo.get_planned_generation_count(run_id)
        decode_params = _stable_decode_params(dict(run.decode_params or {}))
        run_status = run.status
        config_sha256 = run.config_sha256

    report = assemble_run_report(
        run_id=str(run_id),
        run_status=run_status,
        config_sha256=config_sha256,
        model_digest=model.resolved_version if model else "",
        dataset_sha256=dataset.content_sha256 if dataset else "",
        model=_model_context(model),
        dataset=_dataset_context(dataset, case_count=len(cases)),
        prompt_template=_prompt_context(template),
        decode_params=decode_params,
        planned_generations=planned,
        generations=generations,
        scores=scores,
        aggregates=aggregates,
        cases=cases,
        coverage_floor=coverage_floor,
        primary_metric=primary_metric,
    )
    logger.info(
        "report_build_finished",
        run_id=str(run_id),
        publishable=report.publishable,
        coverage=report.coverage,
        primary_metric=report.primary_metric,
        pass_rate=report.pass_rate,
        pass_rate_n=report.pass_rate_n,
        metric_aggregates=len(report.metric_aggregates),
        case_examples=len(report.case_examples),
        duration_ms=timer.elapsed_ms,
    )
    return report


def report_to_json(report: RunReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["pass_rate_ci"] = {
        "low": report.pass_rate_ci[0],
        "high": report.pass_rate_ci[1],
    }
    payload["latency_ms"] = report.latency
    return payload


def overall_aggregates(report: RunReport) -> list[dict[str, Any]]:
    return [row for row in report.metric_aggregates if row["slice"] == OVERALL_SLICE]


def slice_aggregates(report: RunReport) -> list[dict[str, Any]]:
    """Primary-metric rows per slice, worst first — the weakest slice leads."""
    rows = [
        row
        for row in report.metric_aggregates
        if row["slice"] != OVERALL_SLICE and row["metric"] == report.primary_metric
    ]
    return sorted(rows, key=lambda row: (row["value"], row["slice"]))


@lru_cache(maxsize=1)
def _vega_runtime() -> str:
    """Vega + Vega-Lite + Vega-Embed with no external references.

    Emitted once per document so a report with four charts still carries one copy.
    """
    # The stub marks ``snippet`` required; omitting it selects vl-convert's default
    # snippet, which is what binds vegaEmbed/vegaLite/vega onto window.
    return vl_convert.javascript_bundle()  # type: ignore[call-arg]


def _data(rows: list[dict[str, Any]]) -> alt.Data:
    """Inline chart data. Altair ships untyped constructors, so the boundary is here."""
    return alt.Data(values=rows)  # type: ignore[no-untyped-call]


def _labels(rows: list[dict[str, Any]], field: str) -> list[str]:
    """Category order for an axis, as the string sequence altair's ``sort`` expects."""
    return [str(row[field]) for row in rows]


def _render_chart(chart: alt.Chart | None, div_id: str) -> str:
    """Serialize one chart to a div plus its embed call.

    The div id is supplied by the caller rather than left to Altair, which otherwise
    emits a random ``altair-viz-<uuid>`` and makes the report non-reproducible.
    """
    if chart is None:
        return ""
    spec = json.dumps(chart.configure(**_CHART_THEME).to_dict(), sort_keys=True)
    options = json.dumps(_EMBED_OPTIONS, sort_keys=True)
    return (
        f'<div id="{div_id}" class="chart"></div>\n'
        f'<script>vegaEmbed("#{div_id}", {spec}, {options});</script>'
    )


def _metric_figure(report: RunReport) -> alt.Chart | None:
    rows: list[dict[str, Any]] = [
        {
            "metric": row["metric"],
            "value": row["value"] * 100,
            "low": (row["ci_low"] if row["ci_low"] is not None else row["value"]) * 100,
            "high": (row["ci_high"] if row["ci_high"] is not None else row["value"]) * 100,
            "n": row["n"],
        }
        for row in overall_aggregates(report)
    ]
    if not rows:
        return None
    axis = alt.Y("metric:N", sort=_labels(rows, "metric"), title=None)
    bars = (
        alt.Chart(_data(rows))
        .mark_bar(height=16)
        .encode(
            x=alt.X(
                "value:Q",
                title="Score (%) with 95% CI",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(tickCount=5, labelExpr="datum.value + '%'"),
            ),
            y=axis,
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Score (%)", format=".2f"),
                alt.Tooltip("low:Q", title="CI low (%)", format=".2f"),
                alt.Tooltip("high:Q", title="CI high (%)", format=".2f"),
                alt.Tooltip("n:Q", title="n"),
            ],
        )
    )
    whiskers = (
        alt.Chart(_data(rows))
        .mark_rule(color=INK, strokeWidth=1.2)
        .encode(x=alt.X("low:Q", title=""), x2="high:Q", y=axis)
    )
    chart = (bars + whiskers).properties(width="container", height=42 * len(rows) + 30)
    return cast(alt.Chart, chart)


def _slice_figure(report: RunReport) -> alt.Chart | None:
    rows: list[dict[str, Any]] = [
        {
            "slice": row["slice"],
            "value": row["value"] * 100,
            "low": (row["ci_low"] if row["ci_low"] is not None else row["value"]) * 100,
            "high": (row["ci_high"] if row["ci_high"] is not None else row["value"]) * 100,
            "n": row["n"],
            "band": _band(row["value"] * 100),
        }
        for row in slice_aggregates(report)
    ]
    if not rows:
        return None
    observed = {str(row["band"]) for row in rows}
    present = [band for band in _BANDS if band in observed]
    axis = alt.X(
        "slice:N",
        sort=_labels(rows, "slice"),
        title="Slice",
        axis=alt.Axis(labelAngle=0),
    )
    bars = (
        alt.Chart(_data(rows))
        .mark_bar(size=40)
        .encode(
            x=axis,
            y=alt.Y(
                "value:Q",
                title=(
                    f"{report.primary_metric} (%)"
                    if report.headline_kind == HEADLINE_MEAN
                    else f"{report.primary_metric} pass rate (%)"
                ),
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(labelExpr="datum.value + '%'"),
            ),
            color=alt.Color(
                "band:N",
                title=None,
                sort=present,
                # Only bands that occur, so the legend never advertises an empty category.
                scale=alt.Scale(domain=present, range=[_BAND_COLORS[band] for band in present]),
            ),
            tooltip=[
                alt.Tooltip("slice:N", title="Slice"),
                alt.Tooltip("value:Q", title="Pass rate (%)", format=".2f"),
                alt.Tooltip("low:Q", title="CI low (%)", format=".2f"),
                alt.Tooltip("high:Q", title="CI high (%)", format=".2f"),
                alt.Tooltip("n:Q", title="n"),
            ],
        )
    )
    whiskers = (
        alt.Chart(_data(rows))
        .mark_rule(color=INK, strokeWidth=1.2)
        .encode(x=axis, y=alt.Y("low:Q", title=""), y2="high:Q")
    )
    overall = (
        alt.Chart(_data([{"overall": report.pass_rate * 100}]))
        .mark_rule(color=MUTED, strokeDash=[3, 3])
        .encode(y=alt.Y("overall:Q", title=""))
        if report.pass_rate is not None
        else None
    )
    layers = bars + whiskers
    if overall is not None:
        layers = layers + overall
    chart = layers.properties(width="container", height=280)
    return cast(alt.Chart, chart)


_BANDS = ["below 75%", "75–85%", "above 85%"]
_BAND_COLORS = dict(zip(_BANDS, [BAD, WARN, ACCENT], strict=True))


def _band(value: float) -> str:
    if value < 75:
        return _BANDS[0]
    if value < 85:
        return _BANDS[1]
    return _BANDS[2]


def _outcome_figure(report: RunReport) -> alt.Chart | None:
    rows: list[dict[str, Any]] = [
        {
            "outcome": outcome,
            "count": count,
            "category": (
                "Harness failure (excluded)"
                if outcome in HARNESS_OUTCOMES
                else "Model outcome (in denominator)"
            ),
        }
        for outcome, count in report.outcome_histogram.items()
    ]
    if not rows:
        return None
    observed = {str(row["category"]) for row in rows}
    categories = [
        category
        for category in ("Model outcome (in denominator)", "Harness failure (excluded)")
        if category in observed
    ]
    colors = {
        "Model outcome (in denominator)": ACCENT,
        "Harness failure (excluded)": MUTED,
    }
    chart = (
        alt.Chart(_data(rows))
        .mark_bar(size=48)
        .encode(
            x=alt.X(
                "outcome:N",
                sort=_labels(rows, "outcome"),
                title="Provider outcome",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "count:Q",
                title="Generations (count)",
                axis=alt.Axis(tickMinStep=1, format="d"),
            ),
            color=alt.Color(
                "category:N",
                title=None,
                sort=categories,
                # Only categories that occur, so the legend never advertises an empty one.
                scale=alt.Scale(
                    domain=categories, range=[colors[category] for category in categories]
                ),
            ),
            tooltip=[
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Generations"),
                alt.Tooltip("category:N", title="Counted as"),
            ],
        )
        .properties(width="container", height=270)
    )
    return cast(alt.Chart, chart)


def _latency_figure(report: RunReport) -> alt.Chart | None:
    rows: list[dict[str, Any]] = [
        {"stat": key, "ms": report.latency[key]}
        for key in LATENCY_PERCENTILES
        if key in report.latency
    ]
    if not rows or all(row["ms"] == 0 for row in rows):
        return None
    axis = alt.X(
        "stat:N",
        sort=_labels(rows, "stat"),
        title="Percentile",
        axis=alt.Axis(labelAngle=0),
    )
    bars = (
        alt.Chart(_data(rows))
        .mark_bar(size=34)
        .encode(
            x=axis,
            y=alt.Y(
                "ms:Q",
                title="End-to-end latency (ms)",
                scale=alt.Scale(domain=[0, max(row["ms"] for row in rows) * 1.18]),
            ),
            tooltip=[
                alt.Tooltip("stat:N", title="Statistic"),
                alt.Tooltip("ms:Q", title="Latency (ms)", format=",.0f"),
            ],
        )
    )
    labels = (
        alt.Chart(_data(rows))
        .mark_text(dy=-8, fontSize=11, color=MUTED)
        .encode(x=axis, y="ms:Q", text=alt.Text("ms:Q", format=",.0f"))
    )
    chart = (bars + labels).properties(width="container", height=250)
    return cast(alt.Chart, chart)


def _gates(report: RunReport) -> list[dict[str, Any]]:
    """The publishability gate, itemized so the verdict explains itself."""
    return [
        {
            "name": "Run status is completed",
            "ok": report.run_status == "completed",
            "value": report.run_status,
        },
        {
            "name": "All planned generations written",
            "ok": report.written_generations == report.planned_generations,
            "value": f"{report.written_generations:,} / {report.planned_generations:,}",
        },
        {
            "name": f"Coverage \u2265 floor ({report.coverage_floor * 100:.0f}%)",
            "ok": report.coverage >= report.coverage_floor,
            "value": f"{report.coverage * 100:.2f}%",
        },
    ]


def report_to_html(report: RunReport) -> str:
    charts = {
        "metric": _render_chart(_metric_figure(report), METRIC_CHART_DIV_ID),
        "slice": _render_chart(_slice_figure(report), SLICE_CHART_DIV_ID),
        "outcome": _render_chart(_outcome_figure(report), OUTCOME_CHART_DIV_ID),
        "latency": _render_chart(_latency_figure(report), LATENCY_CHART_DIV_ID),
    }
    rendered = _templates.get_template("report_v1.html.j2").render(
        report=report_to_json(report),
        charts=charts,
        runtime=_vega_runtime(),
        gates=_gates(report),
        overall_aggregates=overall_aggregates(report),
        slice_aggregates=slice_aggregates(report),
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
    value = "n/a" if report.pass_rate is None else f"{report.pass_rate:.6f}"
    ET.SubElement(pass_rate, "system-out").text = value
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)


async def write_report(
    run_id: uuid.UUID,
    output_dir: Path,
    coverage_floor: float = 0.98,
    progress: ProgressCallback | None = None,
    primary_metric: str = PRIMARY_METRIC,
) -> RunReport:
    timer = StageTimer()
    emit_progress(
        progress,
        ProgressEvent(PipelineStage.REPORTING, 0, 3, "Building report artifacts"),
    )
    with log_context(run_id=str(run_id)):
        report = await build_report(run_id, coverage_floor, primary_metric)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / str(run_id)
        artifacts = (
            ("json", stem.with_suffix(".json"), json.dumps(report_to_json(report), indent=2)),
            ("html", stem.with_suffix(".html"), report_to_html(report)),
            ("junit", stem.with_suffix(".xml"), report_to_junit(report)),
        )
        for index, (format_name, path, content) in enumerate(artifacts, start=1):
            path.write_text(content, encoding="utf-8")
            logger.info(
                "report_artifact_written",
                format=format_name,
                path=str(path),
                bytes=len(content.encode("utf-8")),
            )
            emit_progress(
                progress,
                ProgressEvent(
                    PipelineStage.REPORTING,
                    index,
                    len(artifacts),
                    f"Wrote {format_name}",
                    {"path": str(path)},
                ),
            )
        logger.info(
            "reporting_finished",
            artifacts=len(artifacts),
            output_dir=str(output_dir),
            duration_ms=timer.elapsed_ms,
        )
        return report
