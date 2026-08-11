"""The evalctl `rag` command group: evidence."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from evalharness.app import build_container
from evalharness.cli._common import _emit_json, console, logger
from evalharness.observability import exception_summary, setup_logging
from evalharness.rag import RagError

rag_app = typer.Typer(no_args_is_help=True)


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
        context = build_container()
        if nli_provider in {None, "mock"}:
            artifact = context.rag.build_rag_evidence(
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
                context.rag.build_live_rag_evidence(
                    report_path=report,
                    evidence_path=evidence,
                    output_path=output,
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
