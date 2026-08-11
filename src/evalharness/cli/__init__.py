"""evalctl — CLI for evalanche: assembles the Typer app from the command modules."""

from __future__ import annotations

import typer

from evalharness.cli.calibrate import calibrate
from evalharness.cli.dataset import dataset_app, dataset_validate
from evalharness.cli.gates import gates_app
from evalharness.cli.judge import judge_app
from evalharness.cli.matrix import baseline_app, matrix_app
from evalharness.cli.metrics import metrics_app
from evalharness.cli.power import power
from evalharness.cli.rag import rag_app
from evalharness.cli.run import run_eval
from evalharness.cli.runs import runs_app
from evalharness.cli.score import score_file
from evalharness.cli.suite import suite_app
from evalharness.observability import setup_logging

app = typer.Typer(no_args_is_help=True, help="evalanche — reproducible LLM evaluation harness")

app.command("power")(power)
app.command("score")(score_file)
app.command("calibrate")(calibrate)
app.command("dataset-validate")(dataset_validate)
app.command("run")(run_eval)

app.add_typer(runs_app, name="runs")
app.add_typer(metrics_app, name="metrics")
app.add_typer(dataset_app, name="dataset")
app.add_typer(suite_app, name="suite")
app.add_typer(judge_app, name="judge")
app.add_typer(rag_app, name="rag")
app.add_typer(matrix_app, name="matrix")
app.add_typer(baseline_app, name="baseline")
app.add_typer(gates_app, name="gates")

__all__ = ["app", "configure_cli_logging"]


@app.callback()
def configure_cli_logging() -> None:
    """Keep structured logs on stderr so command stdout remains machine-readable."""
    setup_logging()
