"""The evalctl `metrics` command group: what this install can actually score."""

from __future__ import annotations

from dataclasses import asdict

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console

metrics_app = typer.Typer(no_args_is_help=True)


@metrics_app.command("list")
def metrics_list() -> None:
    """List discovered metrics, and for each disabled one why it is unavailable."""
    try:
        registry = build_container().metric_registry()
    except (ImportError, ValueError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    statuses = [asdict(status) for status in registry.statuses()]
    _emit_json(
        {
            "metrics": statuses,
            "enabled": sorted(status["name"] for status in statuses if status["enabled"]),
        }
    )
