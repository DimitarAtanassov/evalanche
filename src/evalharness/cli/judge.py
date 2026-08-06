"""The evalctl `judge` command group: run, validate, and attach-calibration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from evalharness.cli._common import _emit_json, console, logger
from evalharness.cli._provider import (
    _close_provider,
    _provider_call_policy,
    _require_live_scoring_provider,
)
from evalharness.judge import (
    JudgeError,
    attach_calibration,
    run_judgment,
    run_live_judgment,
    validate_calibration,
)
from evalharness.judge.models import JudgeMode, JudgmentArtifact
from evalharness.observability import exception_summary, setup_logging
from evalharness.wiring import AppContext, build_app_context

judge_app = typer.Typer(no_args_is_help=True)


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
                    context=build_app_context(),
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
    context: AppContext,
) -> JudgmentArtifact:
    _require_live_scoring_provider(provider_name)
    settings = context.settings
    provider = context.build_provider(
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
            policy=_provider_call_policy(settings, request_timeout_s),
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
