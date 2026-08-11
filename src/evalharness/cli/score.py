"""The evalctl `score` command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console


def score_file(
    inputs: Path = typer.Argument(..., help="JSONL with output and reference fields"),
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Score supplied outputs without inference."""
    try:
        context = build_container()
        lines = inputs.read_text(encoding="utf-8").splitlines()
        scored = context.scoring.score_supplied_rows(
            (json.loads(line) for line in lines),
            [name.strip() for name in metrics.split(",") if name.strip()],
        )
        for row in scored:
            _emit_json(
                {
                    "id": row.external_id,
                    "scores": [
                        {
                            "metric": value.metric_name,
                            "value": value.value,
                            "passed": value.passed,
                            "detail": value.detail,
                        }
                        for value in row.scores
                    ],
                }
            )
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
