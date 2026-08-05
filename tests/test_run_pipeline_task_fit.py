"""The operator run path must headline the metric the pack declares, not `exact_match`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness.cli import _run_async
from evalharness.datasets import load_dataset
from evalharness.reporting.report import PRIMARY_METRIC

DATASET = Path("fixtures/datasets/synthetic-news-smoke")
TEMPLATE = Path("fixtures/templates/classification.jinja")


@pytest.mark.asyncio
async def test_run_publishes_report_scored_on_the_declared_task_metric(
    db_ready, tmp_path: Path
) -> None:
    """Drive the CLI pipeline body so `write_report` wiring, not just `assemble_run_report`, is
    gated: a classification pack scores no `exact_match`, so the default would publish an empty
    pass rate."""
    bundle = load_dataset(DATASET)
    assert bundle.manifest.task_metrics == ["classification"]

    await _run_async(
        dataset_dir=DATASET,
        template=TEMPLATE,
        model="mock-model",
        provider="mock",
        output_dir=tmp_path,
        repeats=1,
        concurrency=2,
        temperature=0.0,
        max_tokens=None,
        seed=None,
        resume=None,
        final_eval=False,
        coverage_floor=0.98,
        tenant_id="test",
    )

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))

    assert payload["primary_metric"] == "classification"
    assert payload["primary_metric"] != PRIMARY_METRIC
    assert payload["pass_rate_n"] == len(bundle.cases)
    # The registered mock provider answers "unknown", so every label is wrong but scored.
    assert payload["pass_rate"] == 0.0
    assert payload["publishable"] is True
    assert {row["metric"] for row in payload["metric_aggregates"]} == {"classification"}
