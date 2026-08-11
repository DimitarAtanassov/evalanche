#!/usr/bin/env python3
"""Run the offline PoC and write committed report artifacts.

Requires PostgreSQL (docker compose up -d postgres). Does not call Ollama.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from evalharness.app.settings import get_settings
from evalharness.datasets import dataset_upsert_fields, load_dataset, validate_dataset
from evalharness.datasets.loader import DatasetBundle
from evalharness.domain.generation import ModelVersion
from evalharness.execution.executor import Executor, render_prompt, response_cache_key
from evalharness.hashing import sha256_hex
from evalharness.observability import setup_logging, setup_otel
from evalharness.providers.mock import MOCK_DIGEST, MockProvider
from evalharness.reporting.report import report_to_html, report_to_json, write_report
from evalharness.scoring.engine import ScoringEngine
from evalharness.db.session import init_db, session_scope
from evalharness.db.models import GenerationRow, MetricAggregateRow, RunRow, ScoreRow
from evalharness.repositories import RunStoreUow

ROOT = Path(__file__).resolve().parents[1]
POC_DIR = ROOT / "fixtures" / "poc"
DATASET = ROOT / "fixtures" / "sample_dataset"
TEMPLATE = ROOT / "fixtures" / "templates" / "qa.jinja"
FIXED_RUN_ID = uuid.UUID("00000000-0000-4000-8000-0000000000c1")


async def _reset_poc_state(
    session: AsyncSession,
    *,
    bundle: DatasetBundle,
    model_version: ModelVersion,
    template_body: str,
    decode_params: dict[str, Any],
) -> None:
    """Drop prior PoC rows and their cached responses so every regeneration runs cold.

    Without the cache purge a second local run would report ``cache_hits == len(cases)``
    while a fresh CI database reports zero, making the committed artifacts irreproducible.
    """
    repo = RunStoreUow(session)
    await repo.delete_cache(
        [
            response_cache_key(
                provider=model_version.provider,
                resolved_version=model_version.resolved_version,
                rendered_prompt=render_prompt(template_body, case),
                decode_params=decode_params,
            )
            for case in bundle.cases
        ]
    )
    if await repo.get_run(FIXED_RUN_ID) is None:
        await session.flush()
        return
    gen_ids = [row.id for row in await repo.get_generations_for_run(FIXED_RUN_ID)]
    if gen_ids:
        await session.execute(delete(ScoreRow).where(ScoreRow.generation_id.in_(gen_ids)))
    await session.execute(
        delete(MetricAggregateRow).where(MetricAggregateRow.run_id == FIXED_RUN_ID)
    )
    await session.execute(delete(GenerationRow).where(GenerationRow.run_id == FIXED_RUN_ID))
    await session.execute(delete(RunRow).where(RunRow.id == FIXED_RUN_ID))
    await session.flush()


async def run_poc(*, output_dir: Path) -> Path:
    setup_logging()
    setup_otel()
    get_settings.cache_clear()
    await init_db()

    bundle = load_dataset(DATASET)
    validation = validate_dataset(bundle)
    if not validation.valid:
        raise SystemExit(f"Dataset invalid: {validation.errors}")

    template_body = TEMPLATE.read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)
    decode_params: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 32,
        "seed": 42,
        "top_p": None,
        "top_k": None,
        "stop": [],
    }

    provider = MockProvider()
    model_version = await provider.resolve_version("mock-qa")

    async with session_scope() as session:
        await _reset_poc_state(
            session,
            bundle=bundle,
            model_version=model_version,
            template_body=template_body,
            decode_params=decode_params,
        )
        repo = RunStoreUow(session)
        dataset_id = await repo.upsert_dataset(**dataset_upsert_fields(bundle))
        prompt_template_id = await repo.upsert_prompt_template(
            name="poc-qa-template",
            version="1.0.0",
            body=template_body,
            sha256=template_sha,
        )
        model_version_id = await repo.upsert_model_version(
            provider=model_version.provider,
            model=model_version.model,
            resolved_version=model_version.resolved_version,
            quantization=model_version.quantization,
            capabilities=dict(model_version.capabilities or {}),
        )

    executor = Executor(
        provider=provider,
        model="mock-qa",
        model_version=model_version,
        template_body=template_body,
    )
    run_id = await executor.create_run(
        bundle_dataset_id=dataset_id,
        prompt_template_id=prompt_template_id,
        model_version_id=model_version_id,
        dataset_sha256=bundle.content_sha256,
        prompt_template_sha256=template_sha,
        decode_params=decode_params,
        repeats=1,
        tenant_id="poc",
        run_id=FIXED_RUN_ID,
    )
    await executor.execute_run(run_id, concurrency=2)
    await ScoringEngine().rescore_run(run_id, ["exact_match"])

    output_dir.mkdir(parents=True, exist_ok=True)
    report = await write_report(run_id, output_dir, coverage_floor=0.98)

    # Stabilize committed artifacts (drop volatile OTel samples).
    report.trace_ids_sample = []
    payload = report_to_json(report)
    POC_DIR.mkdir(parents=True, exist_ok=True)
    json_dst = POC_DIR / "report.json"
    html_dst = POC_DIR / "report.html"
    json_dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    html_dst.write_text(report_to_html(report), encoding="utf-8")

    meta = {
        "run_id": str(run_id),
        "model_digest": MOCK_DIGEST,
        "dataset": "fixtures/sample_dataset",
        "provider": "mock",
        "pass_rate": report.pass_rate,
        "coverage": report.coverage,
        "publishable": report.publishable,
        "config_sha256": report.config_sha256,
    }
    (POC_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return json_dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "poc",
        help="Scratch directory for run-scoped reports",
    )
    args = parser.parse_args()
    path = asyncio.run(run_poc(output_dir=args.output))
    print(f"PoC report written to {path}")


if __name__ == "__main__":
    main()
