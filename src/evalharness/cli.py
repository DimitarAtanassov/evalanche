"""evalctl — CLI for evalanche."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from evalharness.config import get_settings
from evalharness.core.enums import FailureOutcome, TaskType
from evalharness.core.models import Case, Generation
from evalharness.datasets import load_dataset, validate_dataset
from evalharness.execution.executor import Executor
from evalharness.hashing import config_hash, sha256_hex
from evalharness.observability import setup_logging, setup_otel
from evalharness.providers.config import OllamaConfig, OpenAICompatibleConfig
from evalharness.providers.registry import create_provider, load_provider
from evalharness.reporting.report import write_report
from evalharness.scoring.calibration import calibrate_threshold
from evalharness.scoring.engine import ScoringEngine
from evalharness.statistics import apply_multiplicity, compare_binary, required_sample_size
from evalharness.store.db import init_db, session_scope
from evalharness.store.models import CaseRow, GenerationRow, ScoreRow
from evalharness.store.repository import RunRepository

app = typer.Typer(no_args_is_help=True, help="evalanche — reproducible LLM evaluation harness")
runs_app = typer.Typer(no_args_is_help=True)
app.add_typer(runs_app, name="runs")
console = Console()


@app.command("power")
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
    console.print(
        json.dumps({"sample_size_per_arm": sample_size, "power": desired_power}, indent=2)
    )


@app.command("score")
def score_file(
    inputs: Path = typer.Argument(..., help="JSONL with output and reference fields"),
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Score supplied outputs without inference."""
    engine = ScoringEngine()
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
        console.print(
            json.dumps(
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
        )


@runs_app.command("rescore")
def runs_rescore(
    run_id: str,
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Idempotently rescore stored generations with zero inference."""
    count = asyncio.run(
        ScoringEngine().rescore_run(
            uuid.UUID(run_id), [name.strip() for name in metrics.split(",") if name.strip()]
        )
    )
    console.print(json.dumps({"run_id": run_id, "scores_processed": count, "inference_calls": 0}))


@runs_app.command("compare")
def runs_compare(
    baseline_run_id: str,
    candidate_run_id: str,
    metric: str = typer.Option("exact_match", "--metric"),
    allow_compatible: bool = typer.Option(False, "--allow-compatible"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Compare aligned case/repeat outcomes with paired inference."""
    artifact = asyncio.run(
        _compare_runs_async(
            uuid.UUID(baseline_run_id),
            uuid.UUID(candidate_run_id),
            metric,
            allow_compatible,
        )
    )
    payload = json.dumps(artifact, indent=2)
    if output:
        output.write_text(payload, encoding="utf-8")
    console.print(payload)


async def _compare_runs_async(
    baseline_id: uuid.UUID,
    candidate_id: uuid.UUID,
    metric: str,
    allow_compatible: bool,
) -> dict[str, Any]:
    async with session_scope() as session:
        repo = RunRepository(session)
        baseline_run = await repo.get_run(baseline_id)
        candidate_run = await repo.get_run(candidate_id)
        if baseline_run is None or candidate_run is None:
            raise ValueError("Both runs must exist")
        identity = (
            baseline_run.dataset_id == candidate_run.dataset_id
            and baseline_run.prompt_template_id == candidate_run.prompt_template_id
            and baseline_run.config_sha256 == candidate_run.config_sha256
        )
        compatible = (
            baseline_run.dataset_id == candidate_run.dataset_id
            and baseline_run.repeats == candidate_run.repeats
        )
        if not identity and not (allow_compatible and compatible):
            raise ValueError(
                "Dataset/template/config mismatch; pass --allow-compatible only for a "
                "deliberate same-dataset comparison"
            )

        async def outcomes(run_id: uuid.UUID) -> dict[tuple[str, int], bool]:
            statement = (
                select(CaseRow.external_id, GenerationRow.repeat_idx, ScoreRow.passed)
                .join(GenerationRow, ScoreRow.generation_id == GenerationRow.id)
                .join(CaseRow, GenerationRow.case_id == CaseRow.id)
                .where(
                    GenerationRow.run_id == run_id,
                    ScoreRow.metric_name == metric,
                    ScoreRow.passed.is_not(None),
                )
            )
            return {
                (case_id, repeat): bool(passed)
                for case_id, repeat, passed in (await session.execute(statement)).all()
            }

        baseline = await outcomes(baseline_id)
        candidate = await outcomes(candidate_id)
        keys = sorted(set(baseline) & set(candidate))
        baseline_by_case: dict[str, set[bool]] = {}
        candidate_by_case: dict[str, set[bool]] = {}
        for case_id, repeat in keys:
            baseline_by_case.setdefault(case_id, set()).add(baseline[(case_id, repeat)])
            candidate_by_case.setdefault(case_id, set()).add(candidate[(case_id, repeat)])
        flaky = {
            case_id
            for case_id in baseline_by_case
            if len(baseline_by_case[case_id]) > 1 or len(candidate_by_case[case_id]) > 1
        }
        stable_keys = [key for key in keys if key[0] not in flaky]
        result = apply_multiplicity(
            [
                compare_binary(
                    metric,
                    [baseline[key] for key in stable_keys],
                    [candidate[key] for key in stable_keys],
                )
            ]
        )[0]
        return {
            "schema_version": "1.0",
            "baseline_run_id": str(baseline_id),
            "candidate_run_id": str(candidate_id),
            "excluded_flaky_cases": sorted(flaky),
            "result": result.to_dict(),
        }


@app.command("calibrate")
def calibrate(inputs: Path = typer.Argument(..., help="JSONL with label and score")) -> None:
    """Calibrate a threshold on development data."""
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
    result = calibrate_threshold(
        [bool(row["label"]) for row in rows],
        [float(row["score"]) for row in rows],
    )
    console.print(json.dumps(result, indent=2))


@app.command("dataset-validate")
def dataset_validate(
    dataset_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    final_eval: bool = typer.Option(
        False,
        "--i-am-doing-a-final-eval",
        help="Allow holdout split evaluation",
    ),
) -> None:
    """Validate a dataset manifest and cases."""
    setup_logging()
    bundle = load_dataset(dataset_dir)
    report = validate_dataset(bundle, allow_holdout=final_eval)
    if report.errors:
        for err in report.errors:
            console.print(f"[red]ERROR[/red] {err}")
    if report.warnings:
        for warn in report.warnings:
            console.print(f"[yellow]WARN[/yellow] {warn}")
    if report.valid:
        console.print(
            f"[green]Valid[/green] {bundle.manifest.name}@{bundle.manifest.version} "
            f"({len(bundle.cases)} cases, sha256={bundle.content_sha256[:12]}...)"
        )
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("run")
def run_eval(
    dataset_dir: Path = typer.Option(..., "--dataset", help="Dataset directory"),
    template: Path = typer.Option(..., "--template", help="Prompt template file"),
    model: str = typer.Option(..., "--model", help="Model name"),
    provider: str = typer.Option("ollama", "--provider", help="Provider name"),
    output_dir: Path = typer.Option(Path("reports"), "--output", help="Report output dir"),
    repeats: int = typer.Option(1, "--repeats", help="Number of repeats per case"),
    concurrency: int = typer.Option(2, "--concurrency", help="Max concurrent requests"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    seed: int | None = typer.Option(None, "--seed"),
    resume: str | None = typer.Option(None, "--resume", help="Resume existing run ID"),
    final_eval: bool = typer.Option(False, "--i-am-doing-a-final-eval"),
    coverage_floor: float = typer.Option(0.98, "--coverage-floor"),
    tenant_id: str = typer.Option("default", "--tenant"),
) -> None:
    """Run evaluation against a dataset."""
    asyncio.run(
        _run_async(
            dataset_dir=dataset_dir,
            template=template,
            model=model,
            provider=provider,
            output_dir=output_dir,
            repeats=repeats,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            resume=resume,
            final_eval=final_eval,
            coverage_floor=coverage_floor,
            tenant_id=tenant_id,
        )
    )


async def _run_async(
    *,
    dataset_dir: Path,
    template: Path,
    model: str,
    provider: str,
    output_dir: Path,
    repeats: int,
    concurrency: int,
    temperature: float,
    max_tokens: int | None,
    seed: int | None,
    resume: str | None,
    final_eval: bool,
    coverage_floor: float,
    tenant_id: str,
) -> None:
    setup_logging()
    setup_otel()
    settings = get_settings()
    await init_db()

    bundle = load_dataset(dataset_dir)
    validation = validate_dataset(bundle, allow_holdout=final_eval)
    if not validation.valid:
        for err in validation.errors:
            console.print(f"[red]ERROR[/red] {err}")
        raise typer.Exit(1)

    template_body = template.read_text(encoding="utf-8")
    template_sha = sha256_hex(template_body)
    decode_params: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "top_p": None,
        "top_k": None,
        "stop": [],
    }

    if provider == "ollama":
        prov = create_provider(
            OllamaConfig(base_url=settings.ollama_base_url, concurrency=concurrency)
        )
    elif provider == "openai_compatible":
        if (
            settings.openai_compatible_base_url is None
            or settings.openai_compatible_model_revision is None
        ):
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL_REVISION are required"
            )
        prov = create_provider(
            OpenAICompatibleConfig(
                base_url=settings.openai_compatible_base_url,
                api_key=settings.openai_compatible_api_key,
                model_revision=settings.openai_compatible_model_revision,
                concurrency=concurrency,
            )
        )
    else:
        prov = load_provider(provider)
    model_version = await prov.resolve_version(model)
    resumed_run_id = uuid.UUID(resume) if resume else None

    async with session_scope() as session:
        repo = RunRepository(session)
        dataset_id = await repo.upsert_dataset(bundle)
        prompt_template_id = await repo.upsert_prompt_template(
            name=f"{bundle.manifest.name}-{template.stem}",
            version=bundle.manifest.version,
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
        if resumed_run_id is not None:
            stored_run = await repo.get_run(resumed_run_id)
            if stored_run is None:
                raise typer.BadParameter(f"Run not found: {resumed_run_id}", param_hint="--resume")
            supplied_config_sha = config_hash(
                dataset_sha256=bundle.content_sha256,
                prompt_template_sha256=template_sha,
                provider=model_version.provider,
                model=model_version.model,
                resolved_version=model_version.resolved_version,
                decode_params=decode_params,
                harness_version=settings.harness_version,
            )
            mismatches = []
            if stored_run.dataset_id != dataset_id:
                mismatches.append("dataset")
            if stored_run.prompt_template_id != prompt_template_id:
                mismatches.append("prompt template")
            if stored_run.model_version_id != model_version_id:
                mismatches.append("model version")
            if stored_run.config_sha256 != supplied_config_sha:
                mismatches.append("configuration hash")
            if mismatches:
                raise typer.BadParameter(
                    "Resume inputs do not match stored run: " + ", ".join(mismatches),
                    param_hint="--resume",
                )

    executor = Executor(
        provider=prov,
        model=model,
        model_version=model_version,
        template_body=template_body,
    )

    if resume:
        assert resumed_run_id is not None
        run_id = resumed_run_id
        await executor.validate_resume(
            run_id,
            dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            decode_params=decode_params,
            repeats=repeats,
            tenant_id=tenant_id,
        )
        console.print(f"[cyan]Resuming run[/cyan] {run_id}")
    else:
        run_id = await executor.create_run(
            bundle_dataset_id=dataset_id,
            prompt_template_id=prompt_template_id,
            model_version_id=model_version_id,
            dataset_sha256=bundle.content_sha256,
            prompt_template_sha256=template_sha,
            decode_params=decode_params,
            repeats=repeats,
            tenant_id=tenant_id,
        )
        console.print(f"[cyan]Created run[/cyan] {run_id}")

    await executor.execute_run(run_id, concurrency=concurrency)
    await ScoringEngine().rescore_run(run_id, ["exact_match"])
    report = await write_report(run_id, output_dir, coverage_floor=coverage_floor)

    table = Table(title="Run Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", str(run_id))
    table.add_row("Config SHA256", report.config_sha256[:16] + "...")
    table.add_row("Model digest", report.model_digest[:16] + "...")
    table.add_row("Coverage", f"{report.coverage:.2%}")
    table.add_row(
        "Pass rate",
        f"{report.pass_rate:.2%} [{report.pass_rate_ci[0]:.2%}, {report.pass_rate_ci[1]:.2%}]",
    )
    table.add_row("Publishable", str(report.publishable))
    console.print(table)

    if hasattr(prov, "aclose"):
        await prov.aclose()

    if not report.publishable:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
