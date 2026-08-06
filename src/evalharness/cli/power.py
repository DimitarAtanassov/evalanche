"""The evalctl `power` command."""

from __future__ import annotations

import typer

from evalharness.cli._common import _emit_json
from evalharness.statistics import required_sample_size


def power(
    baseline_rate: float = typer.Option(..., min=0.0, max=1.0),
    minimum_detectable_effect: float = typer.Option(..., "--mde"),
    desired_power: float = typer.Option(0.8, "--power", min=0.5, max=0.999),
    alpha: float = typer.Option(0.05, min=0.0001, max=0.5),
) -> None:
    """Calculate sample size for a two-sided rate comparison."""
    sample_size = required_sample_size(
        baseline_rate, minimum_detectable_effect, alpha=alpha, power=desired_power
    )
    _emit_json({"sample_size_per_arm": sample_size, "power": desired_power})
