"""Offline, byte-reproducible Vega-Lite embedding shared by the report and suite views.

Each view keeps its own theme, so ``render_chart`` takes one; everything else about
turning an Altair chart into self-contained HTML is identical and lives here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import altair as alt
import vl_convert

FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"
LINE = "#e4e8ee"
MUTED = "#5b6675"
ACCENT = "#2f5bd7"

EMBED_OPTIONS = {"actions": False, "renderer": "svg"}

_SCRIPT_UNSAFE = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


@lru_cache(maxsize=1)
def vega_runtime() -> str:
    """Vega + Vega-Lite + Vega-Embed with no external references.

    Emitted once per document so a page with four charts still carries one copy.
    """
    # The stub marks ``snippet`` required; omitting it selects vl-convert's default
    # snippet, which is what binds vegaEmbed/vegaLite/vega onto window.
    return vl_convert.javascript_bundle()  # type: ignore[call-arg]


def chart_data(rows: list[dict[str, Any]]) -> alt.Data:
    """Inline chart data. Altair ships untyped constructors, so the boundary is here."""
    return alt.Data(values=rows)  # type: ignore[no-untyped-call]


def script_json(value: Any) -> str:
    """Serialize JSON for inline embedding inside a <script> element.

    Chart specs carry run labels and slice names, so a member run could otherwise
    close the script element with ``</script>`` or break the statement with a raw
    U+2028/U+2029, which JSON permits in strings but JavaScript treats as a line
    terminator. Escaping to \\uXXXX keeps the value JSON-identical after parsing.
    """
    encoded = json.dumps(value, sort_keys=True)
    return "".join(_SCRIPT_UNSAFE.get(character, character) for character in encoded)


def render_chart(chart: alt.Chart | None, div_id: str, *, theme: dict[str, Any]) -> str:
    """Serialize one themed chart to a div plus its embed call.

    The div id is supplied by the caller rather than left to Altair, which otherwise
    emits a random ``altair-viz-<uuid>`` and makes the document non-reproducible.
    """
    if chart is None:
        return ""
    spec = script_json(chart.configure(**theme).to_dict())
    options = script_json(EMBED_OPTIONS)
    return (
        f'<div id="{div_id}" class="chart"></div>\n'
        f'<script>vegaEmbed("#{div_id}", {spec}, {options});</script>'
    )
