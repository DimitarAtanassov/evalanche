"""The evalctl `gates` command group: validate and check."""

from __future__ import annotations

from pathlib import Path

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console
from evalharness.gates import ArtifactOverrides, GatesValidationError

gates_app = typer.Typer(no_args_is_help=True)


@gates_app.command("validate")
def gates_validate(
    manifest: Path = typer.Argument(..., help="Path to gates.yaml"),
) -> None:
    """Validate a gates manifest and every bound artifact."""
    try:
        loaded = build_container().gates.load_gates(manifest)
    except GatesValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "schema_version": loaded.manifest.schema_version,
            "name": loaded.manifest.name,
            "gates": len(loaded.manifest.gates),
            "valid": True,
        }
    )


@gates_app.command("check")
def gates_check(
    gates: Path = typer.Option(..., "--gates", help="Path to gates.yaml"),
    run_report: Path | None = typer.Option(None, "--run-report", help="Override run report path"),
    compare: Path | None = typer.Option(None, "--compare", help="Override compare artifact path"),
    calibration: Path | None = typer.Option(
        None, "--calibration", help="Override calibration artifact path"
    ),
) -> None:
    """Evaluate gates; exit 1 only when a blocking gate fails."""
    overrides = ArtifactOverrides(
        run_report=str(run_report) if run_report is not None else None,
        compare=str(compare) if compare is not None else None,
        calibration=str(calibration) if calibration is not None else None,
    )
    try:
        gates_svc = build_container().gates
        loaded = gates_svc.load_gates(gates, overrides=overrides)
        result = gates_svc.evaluate_gates(loaded)
    except GatesValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(result.model_dump(mode="json"))
    if result.blocking_failed:
        raise typer.Exit(1)
