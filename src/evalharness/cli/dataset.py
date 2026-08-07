"""The evalctl `dataset-validate` command and the `dataset` command group."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.markup import escape

from evalharness.cli._common import _emit_json, console
from evalharness.datasets import (
    DatasetCaseError,
    DatasetManifestError,
    DatasetTier,
    load_dataset,
    validate_dataset,
)
from evalharness.observability import setup_logging

dataset_app = typer.Typer(no_args_is_help=True)

_FACTORY_INSTALL_HINT = (
    "Dataset materialize requires the optional factory package. "
    "Install with `uv sync --extra datasets` (or `pip install 'evalanche[datasets]'`)."
)


def dataset_validate(
    dataset_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    final_eval: bool = typer.Option(
        False,
        "--i-am-doing-a-final-eval",
        help="Allow holdout split evaluation",
    ),
) -> None:
    """Validate a dataset manifest and cases."""
    setup_logging()
    try:
        bundle = load_dataset(dataset_dir)
    except (DatasetCaseError, DatasetManifestError, json.JSONDecodeError, yaml.YAMLError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    report = validate_dataset(bundle, allow_holdout=final_eval)
    if report.errors:
        for err in report.errors:
            console.print(f"[red]ERROR[/red] {err}")
    if report.warnings:
        for warn in report.warnings:
            console.print(f"[yellow]WARN[/yellow] {warn}")
    if report.valid:
        console.print(
            f"[green]Valid[/green] {bundle.manifest.name}@{bundle.manifest.version} "
            f"({len(bundle.cases)} cases, sha256={bundle.content_sha256[:12]}...)"
        )
        raise typer.Exit(0)
    raise typer.Exit(1)


@dataset_app.command("materialize")
def dataset_materialize(
    adapter: str = typer.Option(..., "--adapter", help="Registered offline adapter name"),
    source: Path = typer.Option(..., "--source", help="Pinned local source snapshot"),
    output: Path = typer.Option(..., "--out", help="New dataset bundle directory"),
    seed: int = typer.Option(..., "--seed"),
    size: int = typer.Option(..., "--size", min=1),
    tier: DatasetTier = typer.Option(..., "--tier"),
    check_deterministic: bool = typer.Option(False, "--check-deterministic"),
) -> None:
    """Materialize a pinned local snapshot without network access."""
    try:
        from evaldatasets import MaterializationError, materialize_dataset
    except (ModuleNotFoundError, ImportError) as exc:
        missing = getattr(exc, "name", None)
        if missing == "evaldatasets" or (
            isinstance(missing, str) and missing.startswith("evaldatasets.")
        ):
            console.print(f"[red]ERROR[/red] {escape(_FACTORY_INSTALL_HINT)}")
            raise typer.Exit(1) from exc
        raise
    try:
        materialize_dataset(
            adapter_name=adapter,
            source=source,
            output=output,
            seed=seed,
            size=size,
            tier=tier,
            check_deterministic=check_deterministic,
        )
    except MaterializationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    bundle = load_dataset(output)
    _emit_json(
        {
            "adapter": adapter,
            "dataset": bundle.manifest.name,
            "cases": len(bundle.cases),
            "content_sha256": bundle.content_sha256,
            "output": str(output),
        }
    )
