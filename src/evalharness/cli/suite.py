"""The evalctl `suite` command group: validate and build."""

from __future__ import annotations

from pathlib import Path

import typer

from evalharness.cli._common import _emit_json, console
from evalharness.suite import SuiteValidationError, load_suite, write_suite_artifacts

suite_app = typer.Typer(no_args_is_help=True)


@suite_app.command("validate")
def suite_validate(
    manifest: Path = typer.Argument(..., help="Path to suite.yaml"),
) -> None:
    """Validate a suite manifest and every declared artifact."""
    try:
        validated = load_suite(manifest)
    except SuiteValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "schema_version": validated.manifest.schema_version,
            "name": validated.manifest.name,
            "members": len(validated.members),
            "compares": len(validated.compares),
            "valid": True,
        }
    )


@suite_app.command("build")
def suite_build(
    manifest: Path = typer.Option(..., "--manifest", help="Path to suite.yaml"),
    output: Path = typer.Option(..., "--output", help="Suite artifact output directory"),
) -> None:
    """Build deterministic suite.json and offline suite.html."""
    try:
        report = write_suite_artifacts(manifest, output)
    except SuiteValidationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    _emit_json(
        {
            "suite_digest": report.suite_digest,
            "members": len(report.members),
            "output": str(output),
        }
    )
