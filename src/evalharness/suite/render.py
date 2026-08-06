"""Deterministic offline HTML rendering for benchmark suites."""

from __future__ import annotations

import re
from typing import Any, cast

import altair as alt
from jinja2 import Environment, PackageLoader, select_autoescape

from evalharness.charts import ACCENT, FONT, LINE, MUTED, chart_data, render_chart, vega_runtime
from evalharness.suite.models import SuiteReport

LEADERBOARD_CHART_DIV_ID = "suite-chart-leaderboard"
SLICE_CHART_DIV_ID = "suite-chart-slices"
LATENCY_CHART_DIV_ID = "suite-chart-latency"

# Deliberately a minimal subset of the run report theme: the golden suite HTML is
# compared byte for byte, so adding config here changes every emitted chart spec.
_THEME: dict[str, Any] = {
    "font": FONT,
    "background": "transparent",
    "view": {"stroke": None},
    "axis": {"gridColor": LINE, "labelColor": MUTED, "titleColor": MUTED},
    "legend": {"labelColor": MUTED, "titleColor": MUTED},
    "bar": {"color": ACCENT},
}
_templates = Environment(
    loader=PackageLoader("evalharness.suite", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
)


def _bar_chart(
    rows: list[dict[str, Any]],
    *,
    category: str,
    value: str,
    value_title: str,
    color: str | None = None,
) -> alt.Chart | None:
    if not rows:
        return None
    encoding: dict[str, Any] = {
        "x": alt.X(f"{value}:Q", title=value_title),
        "y": alt.Y(f"{category}:N", title=None, sort=None),
        "tooltip": [
            alt.Tooltip(f"{category}:N"),
            alt.Tooltip(f"{value}:Q", format=".4f"),
        ],
    }
    if color is not None:
        encoding["color"] = alt.Color(f"{color}:N", title=None)
    chart = (
        alt.Chart(chart_data(rows))
        .mark_bar()
        .encode(**encoding)
        .properties(width="container", height=max(160, 30 * len(rows)))
    )
    return cast(alt.Chart, chart)


def _leaderboard_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **entry,
            "dataset": board["dataset"],
            "metric": board["metric"],
            "group": f"{board['dataset']} / {board['metric']}",
        }
        for board in view["leaderboards"]
        for entry in board["entries"]
    ]


def _slice_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_digests = {member["run_id"]: member["dataset_sha256"] for member in view["members"]}
    return [
        {**row, "dataset_sha256": dataset_digests.get(row["run_id"], "")}
        for row in view["slices"]
        if not row["overall"]
    ]


def _latency_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": row["label"], "p95_ms": row["latency_ms"]["p95"]}
        for row in view["ops"]["members"]
        if row["latency_ms"].get("p95", 0) > 0
    ]


def suite_to_html(report: SuiteReport) -> str:
    """Render a byte-reproducible self-contained suite dashboard."""
    view = report.model_dump(mode="json")
    leaderboard_rows = _leaderboard_rows(view)
    slice_rows = _slice_rows(view)
    latency_rows = _latency_rows(view)
    charts = {
        "leaderboard": render_chart(
            _bar_chart(
                leaderboard_rows,
                category="label",
                value="value",
                value_title="Primary metric",
                color="group",
            ),
            LEADERBOARD_CHART_DIV_ID,
            theme=_THEME,
        ),
        "slices": render_chart(
            _bar_chart(
                slice_rows,
                category="slice",
                value="value",
                value_title="Primary metric",
                color="label",
            ),
            SLICE_CHART_DIV_ID,
            theme=_THEME,
        ),
        "latency": render_chart(
            _bar_chart(
                latency_rows,
                category="label",
                value="p95_ms",
                value_title="p95 latency (ms)",
            ),
            LATENCY_CHART_DIV_ID,
            theme=_THEME,
        ),
    }
    rendered = _templates.get_template("suite.html.j2").render(
        suite=view,
        charts=charts,
        runtime=vega_runtime() if any(charts.values()) else "",
        leaderboard_rows=leaderboard_rows,
        slice_rows=slice_rows,
        latency_rows=latency_rows,
    )
    normalized = "\n".join(line.rstrip() for line in rendered.splitlines())
    return re.sub(r"\n{3,}", "\n\n", normalized) + "\n"
