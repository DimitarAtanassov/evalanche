"""Console, logger, and JSON emitter shared by every evalctl command module."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

from evalharness.observability import get_logger

console = Console()
logger = get_logger(__name__)


def _emit_json(payload: object) -> None:
    """Write machine JSON without Rich rendering or terminal wrapping."""
    typer.echo(json.dumps(payload, sort_keys=True, allow_nan=False), file=sys.stdout)
