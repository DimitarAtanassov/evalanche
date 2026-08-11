"""The evalctl `matrix` and `baseline` command groups."""

from __future__ import annotations

from pathlib import Path

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console
from evalharness.matrix import MatrixValidationError

matrix_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)


@matrix_app.command("validate")
def matrix_validate(
    manifest: Path = typer.Argument(..., help="Path to matrix.yaml"),
) -> None:
    """Validate a matrix manifest and print name plus matrix_digest."""
    try:
        loaded = build_container().matrix.load_matrix(manifest)
    except MatrixValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "schema_version": loaded.manifest.schema_version,
            "name": loaded.manifest.name,
            "matrix_digest": loaded.matrix_digest,
            "cells": len(loaded.manifest.cells),
            "valid": True,
        }
    )


@matrix_app.command("digest")
def matrix_digest_cmd(
    manifest: Path = typer.Argument(..., help="Path to matrix.yaml"),
) -> None:
    """Print the content-addressed matrix digest as JSON."""
    try:
        loaded = build_container().matrix.load_matrix(manifest)
    except MatrixValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json({"name": loaded.manifest.name, "matrix_digest": loaded.matrix_digest})


@baseline_app.command("validate")
def baseline_validate(
    manifest: Path = typer.Argument(..., help="Path to baseline.yaml"),
    matrix: Path | None = typer.Option(None, "--matrix", help="Matrix.yaml to cross-check"),
) -> None:
    """Validate pinned baseline cells; optionally verify against a matrix."""
    try:
        loaded = build_container().matrix.load_baseline(manifest, matrix=matrix)
    except MatrixValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "schema_version": loaded.manifest.schema_version,
            "name": loaded.manifest.name,
            "matrix_name": loaded.manifest.matrix_name,
            "matrix_digest": loaded.manifest.matrix_digest,
            "pinned_cells": len(loaded.manifest.pinned_cells),
            "valid": True,
        }
    )


@baseline_app.command("promote")
def baseline_promote(
    matrix: Path = typer.Option(..., "--matrix", help="Path to matrix.yaml"),
    cell: str = typer.Option(..., "--cell", help="Matrix cell id to pin"),
    run_report: Path = typer.Option(..., "--run-report", help="Run report JSON to pin"),
    output: Path = typer.Option(..., "--output", help="baseline.yaml output path"),
    name: str | None = typer.Option(None, "--name", help="Baseline name (new or rename)"),
    allow_mismatch: bool = typer.Option(
        False,
        "--allow-mismatch",
        help="Skip matrix-cell identity checks (provider/model/digest pins)",
    ),
) -> None:
    """Write or merge an explicit digest-pinned baseline cell."""
    try:
        baseline = build_container().matrix.promote_baseline(
            matrix_path=matrix,
            cell_id=cell,
            run_report_path=run_report,
            output_path=output,
            name=name,
            allow_mismatch=allow_mismatch,
        )
    except MatrixValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    _emit_json(
        {
            "name": baseline.name,
            "matrix_name": baseline.matrix_name,
            "matrix_digest": baseline.matrix_digest,
            "pinned_cells": [cell.cell_id for cell in baseline.pinned_cells],
            "output": str(output),
        }
    )
