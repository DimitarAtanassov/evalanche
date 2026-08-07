"""HTML, Altair charts, and JUnit rendering over a RunReport."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Any, cast

import altair as alt
from jinja2 import Environment, PackageLoader, select_autoescape

from evalharness.charts import (
    ACCENT,
    FONT,
    LINE,
    MUTED,
    chart_data,
    render_chart,
    vega_runtime,
)
from evalharness.core.constants import OVERALL_SLICE
from evalharness.reporting.assemble import HARNESS_OUTCOMES, HEADLINE_MEAN, RunReport

METRIC_CHART_DIV_ID = "chart-metric-scores"
SLICE_CHART_DIV_ID = "chart-slice-pass-rate"
OUTCOME_CHART_DIV_ID = "chart-outcome-breakdown"
LATENCY_CHART_DIV_ID = "chart-latency-percentiles"

LATENCY_PERCENTILES = ("p50", "p90", "p95", "p99", "max")

INK = "#111820"
BAD = "#b42318"
WARN = "#b25e09"

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

_BANDS = ["below 75%", "75–85%", "above 85%"]
_BAND_COLORS = dict(zip(_BANDS, [BAD, WARN, ACCENT], strict=True))

_templates = Environment(
    loader=PackageLoader("evalharness.reporting", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
)


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


def _labels(rows: list[dict[str, Any]], field: str) -> list[str]:
    """Category order for an axis, as the string sequence altair's ``sort`` expects."""
    return [str(row[field]) for row in rows]


def _band(value: float) -> str:
    if value < 75:
        return _BANDS[0]
    if value < 85:
        return _BANDS[1]
    return _BANDS[2]


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
        alt.Chart(chart_data(rows))
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
        alt.Chart(chart_data(rows))
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
        alt.Chart(chart_data(rows))
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
        alt.Chart(chart_data(rows))
        .mark_rule(color=INK, strokeWidth=1.2)
        .encode(x=axis, y=alt.Y("low:Q", title=""), y2="high:Q")
    )
    overall = (
        alt.Chart(chart_data([{"overall": report.pass_rate * 100}]))
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
        alt.Chart(chart_data(rows))
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
        alt.Chart(chart_data(rows))
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
        alt.Chart(chart_data(rows))
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
        "metric": render_chart(_metric_figure(report), METRIC_CHART_DIV_ID, theme=_CHART_THEME),
        "slice": render_chart(_slice_figure(report), SLICE_CHART_DIV_ID, theme=_CHART_THEME),
        "outcome": render_chart(_outcome_figure(report), OUTCOME_CHART_DIV_ID, theme=_CHART_THEME),
        "latency": render_chart(_latency_figure(report), LATENCY_CHART_DIV_ID, theme=_CHART_THEME),
    }
    rendered = _templates.get_template("report_v1.html.j2").render(
        report=report_to_json(report),
        charts=charts,
        runtime=vega_runtime(),
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
        failed_gates = [gate for gate in _gates(report) if not gate["ok"]]
        message = (
            "; ".join(str(gate["name"]) for gate in failed_gates)
            if failed_gates
            else "publishability gate failed"
        )
        failure = ET.SubElement(coverage, "failure", message=message)
        details = "; ".join(f"{gate['name']}: {gate['value']}" for gate in failed_gates)
        failure.text = details or (f"{report.coverage:.6f} < {report.coverage_floor:.6f}")
    pass_rate = ET.SubElement(suite, "testcase", name="pass_rate")
    value = "n/a" if report.pass_rate is None else f"{report.pass_rate:.6f}"
    ET.SubElement(pass_rate, "system-out").text = value
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)
