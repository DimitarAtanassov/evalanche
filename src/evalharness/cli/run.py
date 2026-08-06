"""The evalctl `run` command."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import ExitStack
from pathlib import Path

import typer
from rich.table import Table

from evalharness.cli._common import console
from evalharness.cli_progress import PipelineProgress
from evalharness.pipeline import DatasetValidationError, ResumeError, RunResult, run_evaluation


def run_eval(
    dataset_dir: Path = typer.Option(..., "--dataset", help="Dataset directory"),
    template: Path = typer.Option(..., "--template", help="Prompt template file"),
    model: str = typer.Option(..., "--model", help="Model name"),
    provider: str = typer.Option("ollama", "--provider", help="Provider name"),
    output_dir: Path = typer.Option(Path("reports"), "--output", help="Report output dir"),
    repeats: int = typer.Option(1, "--repeats", help="Number of repeats per case"),
    concurrency: int = typer.Option(2, "--concurrency", help="Max concurrent requests"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    seed: int | None = typer.Option(None, "--seed"),
    resume: str | None = typer.Option(None, "--resume", help="Resume existing run ID"),
    final_eval: bool = typer.Option(False, "--i-am-doing-a-final-eval"),
    coverage_floor: float = typer.Option(0.98, "--coverage-floor"),
    tenant_id: str = typer.Option("default", "--tenant"),
) -> None:
    """Run evaluation against a dataset."""
    try:
        with ExitStack() as display:
            pipeline_progress = PipelineProgress(console)

            def announce_run(run_id: uuid.UUID, resumed: bool) -> None:
                label = "Resuming run" if resumed else "Created run"
                console.print(f"[cyan]{label}[/cyan] {run_id}")
                # Hand the terminal to the live display only once the long phases start,
                # so setup failures above it still render on a plain console.
                display.enter_context(pipeline_progress)

            result = asyncio.run(
                run_evaluation(
                    dataset_dir=dataset_dir,
                    template=template,
                    model=model,
                    provider=provider,
                    output_dir=output_dir,
                    repeats=repeats,
                    concurrency=concurrency,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    resume=resume,
                    final_eval=final_eval,
                    coverage_floor=coverage_floor,
                    tenant_id=tenant_id,
                    progress=pipeline_progress,
                    on_run_started=announce_run,
                )
            )
    except DatasetValidationError as exc:
        for err in exc.errors:
            console.print(f"[red]ERROR[/red] {err}")
        raise typer.Exit(1) from exc
    except ResumeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--resume") from exc

    console.print(_run_summary_table(result))

    if not result.report.publishable:
        raise typer.Exit(2)


def _run_summary_table(result: RunResult) -> Table:
    report = result.report
    table = Table(title="Run Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", str(result.run_id))
    table.add_row("Config SHA256", report.config_sha256[:16] + "...")
    table.add_row("Model digest", report.model_digest[:16] + "...")
    table.add_row("Coverage", f"{report.coverage:.2%}")
    table.add_row(
        (
            f"Mean ({report.primary_metric}, n={report.pass_rate_n})"
            if report.headline_kind == "mean"
            else f"Pass rate ({report.primary_metric}, n={report.pass_rate_n})"
        ),
        (
            "n/a"
            if report.pass_rate is None
            else (
                f"{report.pass_rate:.2%}"
                if report.pass_rate_ci[0] is None or report.pass_rate_ci[1] is None
                else (
                    f"{report.pass_rate:.2%} "
                    f"[{report.pass_rate_ci[0]:.2%}, {report.pass_rate_ci[1]:.2%}]"
                )
            )
        ),
    )
    table.add_row("Publishable", str(report.publishable))
    return table
