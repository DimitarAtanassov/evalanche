"""The evalctl `score` command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evalharness.cli._common import _emit_json, console
from evalharness.core.enums import FailureOutcome, TaskType
from evalharness.core.models import Case, Generation
from evalharness.wiring import build_app_context


def score_file(
    inputs: Path = typer.Argument(..., help="JSONL with output and reference fields"),
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Score supplied outputs without inference."""
    try:
        engine = build_app_context().scoring_engine()
        names = [name.strip() for name in metrics.split(",") if name.strip()]
        for index, line in enumerate(inputs.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            case = Case(
                external_id=str(row.get("id", index)),
                task_type=TaskType(row.get("task_type", "qa_short")),
                inputs=row.get("inputs", {}),
                reference_answer=row.get("reference"),
                references=row.get("references", []),
                expected_label=row.get("expected_label"),
                expected_json=row.get("expected_json"),
                qrels=row.get("qrels"),
            )
            generation = Generation(
                id=None,
                run_id="supplied",
                case_external_id=case.external_id,
                repeat_idx=0,
                output=row.get("output"),
                tool_calls=[],
                finish_reason=None,
                outcome=FailureOutcome.PASSED,
                prompt_tokens=None,
                completion_tokens=None,
                cost_usd=0.0,
                ttft_ms=None,
                total_ms=None,
                queue_wait_ms=None,
                attempts=0,
                attempt_log=[],
                cached=False,
                raw_response=None,
                trace_id=None,
            )
            scores = engine.score_one(generation, case, names)
            _emit_json(
                {
                    "id": case.external_id,
                    "scores": [
                        {
                            "metric": value.metric_name,
                            "value": value.value,
                            "passed": value.passed,
                            "detail": value.detail,
                        }
                        for value in scores
                    ],
                }
            )
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
