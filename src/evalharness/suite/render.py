"""Deterministic offline HTML rendering for benchmark suites."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, cast

import altair as alt
import vl_convert
from jinja2 import Environment, PackageLoader, select_autoescape

from evalharness.suite.models import SuiteReport

LEADERBOARD_CHART_DIV_ID = "suite-chart-leaderboard"
SLICE_CHART_DIV_ID = "suite-chart-slices"
LATENCY_CHART_DIV_ID = "suite-chart-latency"

_EMBED_OPTIONS = {"actions": False, "renderer": "svg"}
_THEME: dict[str, Any] = {
    "font": "system-ui,-apple-system,'Segoe UI',sans-serif",
    "background": "transparent",
    "view": {"stroke": None},
    "axis": {"gridColor": "#e4e8ee", "labelColor": "#5b6675", "titleColor": "#5b6675"},
    "legend": {"labelColor": "#5b6675", "titleColor": "#5b6675"},
    "bar": {"color": "#2f5bd7"},
}
_templates = Environment(
    loader=PackageLoader("evalharness.suite", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
)


@lru_cache(maxsize=1)
def _vega_runtime() -> str:
    return vl_convert.javascript_bundle()  # type: ignore[call-arg]


def _data(rows: list[dict[str, Any]]) -> alt.Data:
    return alt.Data(values=rows)  # type: ignore[no-untyped-call]


_SCRIPT_UNSAFE = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


def _script_json(value: Any) -> str:
    """Serialize JSON for inline embedding inside a <script> element.

    Chart specs carry run labels and slice names, so a member run could otherwise
    close the script element with ``</script>`` or break the statement with a raw
    U+2028/U+2029, which JSON permits in strings but JavaScript treats as a line
    terminator. Escaping to \\uXXXX keeps the value JSON-identical after parsing.
    """
    encoded = json.dumps(value, sort_keys=True)
    return "".join(_SCRIPT_UNSAFE.get(character, character) for character in encoded)


def _render_chart(chart: alt.Chart | None, div_id: str) -> str:
    if chart is None:
        return ""
    spec = _script_json(chart.configure(**_THEME).to_dict())
    options = _script_json(_EMBED_OPTIONS)
    return (
        f'<div id="{div_id}" class="chart"></div>\n'
        f'<script>vegaEmbed("#{div_id}", {spec}, {options});</script>'
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
        alt.Chart(_data(rows))
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
        "leaderboard": _render_chart(
            _bar_chart(
                leaderboard_rows,
                category="label",
                value="value",
                value_title="Primary metric",
                color="group",
            ),
            LEADERBOARD_CHART_DIV_ID,
        ),
        "slices": _render_chart(
            _bar_chart(
                slice_rows,
                category="slice",
                value="value",
                value_title="Primary metric",
                color="label",
            ),
            SLICE_CHART_DIV_ID,
        ),
        "latency": _render_chart(
            _bar_chart(
                latency_rows,
                category="label",
                value="p95_ms",
                value_title="p95 latency (ms)",
            ),
            LATENCY_CHART_DIV_ID,
        ),
    }
    rendered = _templates.get_template("suite.html.j2").render(
        suite=view,
        charts=charts,
        runtime=_vega_runtime() if any(charts.values()) else "",
        leaderboard_rows=leaderboard_rows,
        slice_rows=slice_rows,
        latency_rows=latency_rows,
    )
    normalized = "\n".join(line.rstrip() for line in rendered.splitlines())
    return re.sub(r"\n{3,}", "\n\n", normalized) + "\n"
