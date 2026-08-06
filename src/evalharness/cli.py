"""evalctl — CLI for evalanche."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from evalharness.cli_progress import PipelineProgress
from evalharness.config import get_settings
from evalharness.core.enums import FailureOutcome, TaskType
from evalharness.core.models import Case, Generation
from evalharness.core.protocols import Provider
from evalharness.datasets import (
    DatasetCaseError,
    DatasetManifestError,
    DatasetTier,
    load_dataset,
    validate_dataset,
)
from evalharness.execution.executor import Executor
from evalharness.hashing import config_hash, sha256_hex
from evalharness.judge import (
    JudgeError,
    attach_calibration,
    run_judgment,
    run_live_judgment,
    validate_calibration,
)
from evalharness.judge.models import JudgeMode, JudgmentArtifact
from evalharness.observability import (
    StageTimer,
    exception_summary,
    get_logger,
    setup_logging,
    setup_otel,
)
from evalharness.providers.call_policy import ProviderCallPolicy
from evalharness.providers.config import OllamaConfig, OpenAICompatibleConfig
from evalharness.providers.registry import create_provider, load_provider
from evalharness.rag import RagError, build_live_rag_evidence, build_rag_evidence
from evalharness.reporting.report import PRIMARY_METRIC, write_report
from evalharness.scoring.calibration import calibrate_threshold
from evalharness.scoring.engine import ScoringEngine
from evalharness.statistics import apply_multiplicity, compare_binary, required_sample_size
from evalharness.store.db import init_db, session_scope
from evalharness.store.models import CaseRow, GenerationRow, ScoreRow
from evalharness.store.repository import RunRepository
from evalharness.suite import SuiteValidationError, load_suite, write_suite_artifacts
from tools.datasets import MaterializationError, materialize_dataset

app = typer.Typer(no_args_is_help=True, help="evalanche — reproducible LLM evaluation harness")
runs_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
suite_app = typer.Typer(no_args_is_help=True)
judge_app = typer.Typer(no_args_is_help=True)
rag_app = typer.Typer(no_args_is_help=True)
app.add_typer(runs_app, name="runs")
app.add_typer(dataset_app, name="dataset")
app.add_typer(suite_app, name="suite")
app.add_typer(judge_app, name="judge")
app.add_typer(rag_app, name="rag")
console = Console()
logger = get_logger(__name__)


def _emit_json(payload: object) -> None:
    """Write machine JSON without Rich rendering or terminal wrapping."""
    typer.echo(json.dumps(payload, sort_keys=True, allow_nan=False), file=sys.stdout)


@app.callback()
def configure_cli_logging() -> None:
    """Keep structured logs on stderr so command stdout remains machine-readable."""
    setup_logging()


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
    _emit_json({"sample_size_per_arm": sample_size, "power": desired_power})


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


@runs_app.command("rescore")
def runs_rescore(
    run_id: str,
    metrics: str = typer.Option("exact_match", "--metrics"),
) -> None:
    """Idempotently rescore stored generations with zero inference."""
    setup_logging()
    with PipelineProgress(console) as pipeline_progress:
        count = asyncio.run(
            ScoringEngine().rescore_run(
                uuid.UUID(run_id),
                [name.strip() for name in metrics.split(",") if name.strip()],
                progress=pipeline_progress,
            )
        )
    _emit_json({"run_id": run_id, "scores_processed": count, "inference_calls": 0})


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
    payload = json.dumps(artifact, indent=2, allow_nan=False)
    if output:
        output.write_text(payload, encoding="utf-8")
    _emit_json(artifact)


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
    _emit_json(result)


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
    try:
        bundle = load_dataset(dataset_dir)
    except (DatasetCaseError, DatasetManifestError, json.JSONDecodeError, yaml.YAMLError) as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
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


@dataset_app.command("materialize")
def dataset_materialize(
    adapter: str = typer.Option(..., "--adapter", help="Registered offline adapter name"),
    source: Path = typer.Option(..., "--source", help="Pinned local source snapshot"),
    output: Path = typer.Option(..., "--out", help="New dataset bundle directory"),
    seed: int = typer.Option(..., "--seed"),
    size: int = typer.Option(..., "--size", min=1),
    tier: DatasetTier = typer.Option(..., "--tier"),
    check_deterministic: bool = typer.Option(False, "--check-deterministic"),
) -> None:
    """Materialize a pinned local snapshot without network access."""
    try:
        materialize_dataset(
            adapter_name=adapter,
            source=source,
            output=output,
            seed=seed,
            size=size,
            tier=tier,
            check_deterministic=check_deterministic,
        )
    except MaterializationError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[red]IO_ERROR[/red] {exc}")
        raise typer.Exit(2) from exc
    bundle = load_dataset(output)
    _emit_json(
        {
            "adapter": adapter,
            "dataset": bundle.manifest.name,
            "cases": len(bundle.cases),
            "content_sha256": bundle.content_sha256,
            "output": str(output),
        }
    )


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


def _create_scoring_provider(
    provider_name: str,
    *,
    concurrency: int,
    rpm: int,
    tpm: int,
) -> Provider:
    settings = get_settings()
    if provider_name == "ollama":
        return create_provider(
            OllamaConfig(
                base_url=settings.ollama_base_url,
                concurrency=concurrency,
                rpm=rpm,
                tpm=tpm,
            )
        )
    if provider_name == "openai_compatible":
        if (
            settings.openai_compatible_base_url is None
            or settings.openai_compatible_model_revision is None
        ):
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL_REVISION are required"
            )
        return create_provider(
            OpenAICompatibleConfig(
                base_url=settings.openai_compatible_base_url,
                api_key=settings.openai_compatible_api_key,
                model_revision=settings.openai_compatible_model_revision,
                concurrency=concurrency,
                rpm=rpm,
                tpm=tpm,
            )
        )
    raise ValueError(
        f"live scoring provider must be ollama or openai_compatible, got {provider_name!r}"
    )


def _provider_call_policy(request_timeout_s: float | None) -> ProviderCallPolicy:
    settings = get_settings()
    return ProviderCallPolicy(
        request_timeout_s=request_timeout_s or settings.default_request_timeout_s,
        max_retries=settings.default_max_retries,
        retry_base_s=settings.default_retry_base_s,
        retry_cap_s=settings.default_retry_cap_s,
    )


async def _close_provider(provider: Provider) -> None:
    await provider.aclose()


@judge_app.command("run")
def judge_run(
    mode: JudgeMode = typer.Option(..., "--mode"),
    rubric: Path = typer.Option(..., "--rubric"),
    candidates: Path | None = typer.Option(None, "--candidates"),
    pairs: Path | None = typer.Option(None, "--pairs"),
    provider: str = typer.Option(..., "--provider"),
    model: str = typer.Option(..., "--model"),
    judge_family: str = typer.Option(..., "--judge-family"),
    candidate_family: str = typer.Option(..., "--candidate-family"),
    responses: Path | None = typer.Option(None, "--responses"),
    seed: int = typer.Option(..., "--seed"),
    output: Path = typer.Option(..., "--output"),
    concurrency: int = typer.Option(2, "--concurrency", min=1),
    request_timeout_s: float | None = typer.Option(
        None,
        "--request-timeout",
        min=0.1,
        help="Per-provider-call timeout in seconds",
    ),
) -> None:
    """Run pointwise or pairwise judging into judgment.json (always informational)."""
    setup_logging()
    try:
        if provider == "mock":
            artifact = run_judgment(
                mode=mode,
                rubric_path=rubric,
                candidates_path=candidates,
                pairs_path=pairs,
                provider=provider,
                model=model,
                judge_family=judge_family,
                candidate_family=candidate_family,
                responses_path=responses,
                seed=seed,
                output_path=output,
            )
        else:
            if responses is not None:
                raise JudgeError(
                    "INVALID_PROVIDER_CONFIG",
                    "--responses is only valid with --provider mock",
                )
            artifact = asyncio.run(
                _judge_run_live_async(
                    mode=mode,
                    rubric=rubric,
                    candidates=candidates,
                    pairs=pairs,
                    provider_name=provider,
                    model=model,
                    judge_family=judge_family,
                    candidate_family=candidate_family,
                    seed=seed,
                    output=output,
                    concurrency=concurrency,
                    request_timeout_s=request_timeout_s,
                )
            )
    except (JudgeError, ValueError) as exc:
        code = exc.code if isinstance(exc, JudgeError) else "INVALID_PROVIDER_CONFIG"
        logger.error(
            "judge_run_failed",
            code=code,
            mode=mode.value,
            provider=provider,
            model=model,
            **exception_summary(exc),
        )
        console.print(f"[red]{code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "mode": artifact.mode.value,
            "items": len(artifact.items),
            "gating_allowed": artifact.gating_allowed,
            "output": str(output),
        }
    )


async def _judge_run_live_async(
    *,
    mode: JudgeMode,
    rubric: Path,
    candidates: Path | None,
    pairs: Path | None,
    provider_name: str,
    model: str,
    judge_family: str,
    candidate_family: str,
    seed: int,
    output: Path,
    concurrency: int,
    request_timeout_s: float | None,
) -> JudgmentArtifact:
    settings = get_settings()
    provider = _create_scoring_provider(
        provider_name,
        concurrency=concurrency,
        rpm=settings.judge_provider_rpm,
        tpm=settings.judge_provider_tpm,
    )
    try:
        return await run_live_judgment(
            mode=mode,
            rubric_path=rubric,
            candidates_path=candidates,
            pairs_path=pairs,
            provider=provider,
            model=model,
            judge_family=judge_family,
            candidate_family=candidate_family,
            seed=seed,
            output_path=output,
            concurrency=concurrency,
            policy=_provider_call_policy(request_timeout_s),
        )
    finally:
        await _close_provider(provider)


@judge_app.command("validate")
def judge_validate(
    judgments: Path = typer.Option(..., "--judgments"),
    labels_dev: Path | None = typer.Option(None, "--labels-dev"),
    labels_holdout: Path = typer.Option(..., "--labels-holdout"),
    rubric: Path = typer.Option(..., "--rubric", help="Rubric that produced the judgment"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Compute holdout calibration into calibration.json (source of the gate bit)."""
    setup_logging()
    try:
        artifact = validate_calibration(
            judgment_path=judgments,
            labels_dev_path=labels_dev,
            labels_holdout_path=labels_holdout,
            rubric_path=rubric,
            output_path=output,
        )
    except JudgeError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "gating_allowed": artifact.gating_allowed,
            "family_separation_ok": artifact.family_separation_ok,
            "n_holdout": artifact.holdout.n,
            "agreement_holdout": artifact.holdout.agreement,
            "output": str(output),
        }
    )


@judge_app.command("attach-calibration")
def judge_attach_calibration(
    judgment: Path = typer.Option(..., "--judgment"),
    calibration: Path = typer.Option(..., "--calibration"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Copy a passing calibration digest onto a new judgment artifact."""
    setup_logging()
    try:
        artifact = attach_calibration(
            judgment_path=judgment,
            calibration_path=calibration,
            output_path=output,
        )
    except JudgeError as exc:
        console.print(f"[red]{exc.code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "gating_allowed": artifact.gating_allowed,
            "calibration_digest": artifact.calibration_digest,
            "output": str(output),
        }
    )


@rag_app.command("evidence")
def rag_evidence(
    report: Path = typer.Option(..., "--report"),
    evidence: Path = typer.Option(..., "--evidence"),
    output: Path = typer.Option(..., "--output"),
    nli_provider: str | None = typer.Option(None, "--nli-provider"),
    nli_model: str | None = typer.Option(None, "--nli-model"),
    nli_responses: Path | None = typer.Option(None, "--nli-responses"),
    concurrency: int = typer.Option(2, "--concurrency", min=1),
    request_timeout_s: float | None = typer.Option(
        None,
        "--request-timeout",
        min=0.1,
        help="Per-provider-call timeout in seconds",
    ),
) -> None:
    """Build rag_evidence.json from a run report and local evidence JSONL."""
    setup_logging()
    try:
        if nli_provider in {None, "mock"}:
            artifact = build_rag_evidence(
                report_path=report,
                evidence_path=evidence,
                output_path=output,
                nli_provider=nli_provider,
                nli_model=nli_model,
                nli_responses_path=nli_responses,
            )
        else:
            if nli_model is None:
                raise RagError("NLI_UNAVAILABLE", "--nli-model is required with a live provider")
            if nli_responses is not None:
                raise RagError(
                    "INVALID_PROVIDER_CONFIG",
                    "--nli-responses is only valid with --nli-provider mock",
                )
            artifact = asyncio.run(
                _rag_evidence_live_async(
                    report=report,
                    evidence=evidence,
                    output=output,
                    provider_name=nli_provider,
                    nli_model=nli_model,
                    concurrency=concurrency,
                    request_timeout_s=request_timeout_s,
                )
            )
    except (RagError, ValueError) as exc:
        code = exc.code if isinstance(exc, RagError) else "INVALID_PROVIDER_CONFIG"
        logger.error(
            "rag_evidence_failed",
            code=code,
            provider=nli_provider,
            model=nli_model,
            **exception_summary(exc),
        )
        console.print(f"[red]{code}[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit_json(
        {
            "run_id": artifact.get("run_id"),
            "faithfulness_status": (artifact.get("faithfulness") or {}).get("status"),
            "retrieval_status": (artifact.get("retrieval") or {}).get("status"),
            "gating_allowed": artifact.get("gating_allowed"),
            "output": str(output),
        }
    )


async def _rag_evidence_live_async(
    *,
    report: Path,
    evidence: Path,
    output: Path,
    provider_name: str,
    nli_model: str,
    concurrency: int,
    request_timeout_s: float | None,
) -> dict[str, Any]:
    settings = get_settings()
    provider = _create_scoring_provider(
        provider_name,
        concurrency=concurrency,
        rpm=settings.nli_provider_rpm,
        tpm=settings.nli_provider_tpm,
    )
    try:
        return await build_live_rag_evidence(
            report_path=report,
            evidence_path=evidence,
            output_path=output,
            provider=provider,
            nli_model=nli_model,
            concurrency=concurrency,
            policy=_provider_call_policy(request_timeout_s),
        )
    finally:
        await _close_provider(provider)


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
    pipeline_timer = StageTimer()
    settings = get_settings()
    await init_db()

    logger.info("dataset_validation_started", dataset_path=str(dataset_dir))
    bundle = load_dataset(dataset_dir)
    validation = validate_dataset(bundle, allow_holdout=final_eval)
    logger.info(
        "dataset_validated",
        dataset=bundle.manifest.name,
        version=bundle.manifest.version,
        split=bundle.manifest.split,
        cases=len(bundle.cases),
        content_sha256=bundle.content_sha256,
        valid=validation.valid,
        warnings=len(validation.warnings),
        errors=len(validation.errors),
    )
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
    logger.info("provider_resolution_started", provider=provider, model=model)
    model_version = await prov.resolve_version(model)
    logger.info(
        "provider_resolved",
        provider=model_version.provider,
        model=model_version.model,
        model_digest=model_version.resolved_version,
        capabilities=dict(model_version.capabilities or {}),
    )
    resumed_run_id = uuid.UUID(resume) if resume else None

    logger.info(
        "pipeline_planning_started",
        resume=bool(resume),
        repeats=repeats,
        concurrency=concurrency,
    )
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

    logger.info(
        "pipeline_planning_finished",
        dataset_id=dataset_id,
        prompt_template_id=prompt_template_id,
        model_version_id=model_version_id,
        resume=bool(resume),
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

    # The headline pass rate must be computed from a metric this run actually scored.
    metric_names = list(bundle.manifest.task_metrics or [PRIMARY_METRIC])
    try:
        with PipelineProgress(console) as pipeline_progress:
            await executor.execute_run(run_id, concurrency=concurrency, progress=pipeline_progress)
            await ScoringEngine().rescore_run(
                run_id,
                metric_names,
                progress=pipeline_progress,
            )
            report = await write_report(
                run_id,
                output_dir,
                coverage_floor=coverage_floor,
                progress=pipeline_progress,
                primary_metric=metric_names[0],
            )
    except Exception as exc:
        logger.exception("pipeline_failed", **exception_summary(exc))
        raise
    finally:
        if hasattr(prov, "aclose"):
            await prov.aclose()
    logger.info(
        "pipeline_finished",
        run_id=str(run_id),
        publishable=report.publishable,
        coverage=report.coverage,
        primary_metric=report.primary_metric,
        pass_rate=report.pass_rate,
        pass_rate_n=report.pass_rate_n,
        duration_ms=pipeline_timer.elapsed_ms,
    )

    table = Table(title="Run Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", str(run_id))
    table.add_row("Config SHA256", report.config_sha256[:16] + "...")
    table.add_row("Model digest", report.model_digest[:16] + "...")
    table.add_row("Coverage", f"{report.coverage:.2%}")
    table.add_row(
        (
            f"Mean ({report.primary_metric}, n={report.pass_rate_n})"
            if report.headline_kind == "mean"
            else f"Pass rate ({report.primary_metric}, n={report.pass_rate_n})"
        ),
        (
            "n/a"
            if report.pass_rate is None
            else (
                f"{report.pass_rate:.2%}"
                if report.pass_rate_ci[0] is None or report.pass_rate_ci[1] is None
                else (
                    f"{report.pass_rate:.2%} "
                    f"[{report.pass_rate_ci[0]:.2%}, {report.pass_rate_ci[1]:.2%}]"
                )
            )
        ),
    )
    table.add_row("Publishable", str(report.publishable))
    console.print(table)

    if not report.publishable:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
