"""Create the real Ollama v0.2.0 release evidence bundle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import sys
from pathlib import Path

import yaml
from sqlalchemy import select

from evalharness.app import build_container
from evalharness.domain.scoring import ScoreValue
from evalharness.providers.ollama import OllamaProvider
from evalharness.reporting.report import write_report
from evalharness.scoring.embeddings import EmbeddingService
from evalharness.statistics import bca_bootstrap
from evalharness.db.session import session_scope
from evalharness.db.models import CaseRow, GenerationRow, RunRow
from evalharness.repositories import RunStoreUow

ROOT = Path("release/v0.2.0")
DATASET = ROOT / "work" / "qa-500"
REPORTS = ROOT / "reports"


def prepare_inputs() -> tuple[Path, Path]:
    DATASET.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "id": f"qa-{index:04d}",
                "task_type": "qa_short",
                "inputs": {"question": f"What is {index % 10} modulo 10?"},
                "reference_answer": str(index % 10),
                "slices": {"parity": "even" if index % 2 == 0 else "odd"},
            },
            separators=(",", ":"),
        )
        for index in range(500)
    ]
    content = "\n".join(rows)
    (DATASET / "cases.jsonl").write_text(content + "\n", encoding="utf-8")
    manifest = {
        "name": "evalanche-release-qa",
        "version": "0.2.0",
        "split": "dev",
        "license": "CC0-1.0",
        "pii_scrubbed": True,
        "created_at": "2026-08-05",
        "slices": ["parity"],
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    (DATASET / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    baseline = DATASET / "baseline.jinja"
    candidate = DATASET / "candidate.jinja"
    baseline.write_text("Answer with only the digit. {{question}}", encoding="utf-8")
    candidate.write_text(
        "Compute carefully, then output exactly one digit and nothing else: {{question}}",
        encoding="utf-8",
    )
    return baseline, candidate


async def latest_runs() -> list[RunRow]:
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(RunRow)
                    .where(
                        RunRow.tenant_id == "release-v0.2.0",
                        RunRow.status == "completed",
                    )
                    .order_by(RunRow.started_at.desc())
                    .limit(2)
                )
            )
            .scalars()
            .all()
        )
        return list(reversed(rows))


async def semantic_rescore(run: RunRow, provider: OllamaProvider) -> None:
    version = await provider.resolve_version("nomic-embed-text")
    service = EmbeddingService(
        provider,
        "nomic-embed-text",
        version.resolved_version,
        dimension=768,
        batch_size=64,
    )
    values: list[ScoreValue] = []
    async with session_scope() as session:
        repo = RunStoreUow(session)
        generations = await repo.get_generations_for_run(run.id)
        cases = {
            row.id: row
            for row in (
                await session.execute(select(CaseRow).where(CaseRow.dataset_id == run.dataset_id))
            ).scalars()
        }
        for start in range(0, len(generations), 64):
            batch = generations[start : start + 64]
            texts: list[str] = []
            valid: list[GenerationRow] = []
            for generation in batch:
                reference = (cases[generation.case_id].reference or {}).get("reference_answer")
                if generation.output is not None and reference is not None:
                    valid.append(generation)
                    texts.extend([generation.output, reference])
            vectors = await service.embed(texts)
            for index, generation in enumerate(valid):
                left, right = vectors[index * 2], vectors[index * 2 + 1]
                similarity = sum(a * b for a, b in zip(left, right, strict=True))
                score = ScoreValue(
                    "semantic_similarity",
                    "1.0.0",
                    hashlib.sha256(
                        f"nomic-embed-text:{version.resolved_version}:cosine".encode()
                    ).hexdigest(),
                    similarity,
                    similarity >= 0.8,
                    {
                        "model": "nomic-embed-text",
                        "revision": version.resolved_version,
                        "dimension": 768,
                        "variant": "max_reference",
                    },
                )
                values.append(score)
                await repo.save_score(
                    generation_id=generation.id,
                    metric_name=score.metric_name,
                    metric_version=score.metric_version,
                    metric_config_sha256=score.metric_config_sha256,
                    value=score.value,
                    passed=score.passed,
                    detail=score.detail,
                )
        numeric = [float(value.value) for value in values if value.value is not None]
        low, high = bca_bootstrap(numeric, seed=20260805)
        await repo.save_metric_aggregate(
            run_id=run.id,
            metric_name="semantic_similarity",
            metric_version="1.0.0",
            metric_config_sha256=values[0].metric_config_sha256,
            slice_key="__overall__",
            n=len(numeric),
            value=sum(numeric) / len(numeric),
            ci_low=low,
            ci_high=high,
            stddev=None,
            method="BCa-10000-seed-20260805",
        )


async def main() -> None:
    baseline_template, candidate_template = prepare_inputs()
    REPORTS.mkdir(parents=True, exist_ok=True)
    context = build_container()
    common = {
        "dataset_dir": DATASET,
        "model": "llama3.2:1b",
        "provider": "ollama",
        "output_dir": REPORTS,
        "repeats": 5,
        "concurrency": 2,
        "temperature": 0.2,
        "max_tokens": 4,
        "seed": None,
        "resume": None,
        "final_eval": False,
        "coverage_floor": 0.98,
        "tenant_id": "release-v0.2.0",
    }
    existing = await latest_runs()
    if len(existing) < 2:
        for template in (baseline_template, candidate_template):
            result = await context.evaluation.run(
                template=template,
                **common,
            )
            # The pipeline reports publishability; only the CLI aborts on it. Evidence
            # must never be assembled from a run that failed the coverage floor.
            if not result.report.publishable:
                raise SystemExit(f"Run {result.run_id} is not publishable")
        existing = await latest_runs()
    baseline_run, candidate_run = existing

    provider = OllamaProvider()
    await semantic_rescore(baseline_run, provider)
    await semantic_rescore(candidate_run, provider)
    await provider.aclose()
    await write_report(baseline_run.id, REPORTS)
    await write_report(candidate_run.id, REPORTS)

    comparison = await context.compare.compare_runs(
        baseline_run.id,
        candidate_run.id,
        "exact_match",
        True,
    )
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    manifest = {
        "baseline_run_id": str(baseline_run.id),
        "candidate_run_id": str(candidate_run.id),
        "dataset_sha256": baseline_run.config_sha256,
        "generation_model": "llama3.2:1b",
        "embedding_model": "nomic-embed-text",
        "repeats": 5,
        "temperature": 0.2,
        "python": sys.version,
        "platform": platform.platform(),
        "reproduce": "uv run python scripts/run_release_e2e.py",
    }
    (ROOT / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
