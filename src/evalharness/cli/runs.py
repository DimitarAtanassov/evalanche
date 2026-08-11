"""The evalctl `runs` command group: rescore and compare."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console
from evalharness.cli_progress import PipelineProgress
from evalharness.observability import setup_logging

runs_app = typer.Typer(no_args_is_help=True)


@runs_app.command("rescore")
def runs_rescore(
    run_id: str,
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Idempotently rescore stored generations with zero inference."""
    setup_logging()
    try:
        context = build_container()
        with PipelineProgress(console) as pipeline_progress:
            count = asyncio.run(
                context.scoring.rescore_run(
                    uuid.UUID(run_id),
                    [name.strip() for name in metrics.split(",") if name.strip()],
                    progress=pipeline_progress,
                )
            )
        _emit_json({"run_id": run_id, "scores_processed": count, "inference_calls": 0})
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    except (KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc


@runs_app.command("compare")
def runs_compare(
    baseline_run_id: str,
    candidate_run_id: str,
    metric: str = typer.Option("exact_match", "--metric"),
    allow_compatible: bool = typer.Option(False, "--allow-compatible"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Compare aligned case/repeat outcomes with paired inference."""
    try:
        context = build_container()
        artifact = asyncio.run(
            context.compare.compare_runs(
                uuid.UUID(baseline_run_id),
                uuid.UUID(candidate_run_id),
                metric,
                allow_compatible,
            )
        )
        payload = json.dumps(artifact, indent=2, allow_nan=False)
        if output:
            output.write_text(payload, encoding="utf-8")
        _emit_json(artifact)
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    except (KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
