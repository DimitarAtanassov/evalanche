"""evalctl — CLI for evalanche."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from evalharness.config import get_settings
from evalharness.datasets import load_dataset, validate_dataset
from evalharness.execution.executor import Executor
from evalharness.hashing import sha256_hex
from evalharness.observability import setup_logging, setup_otel
from evalharness.providers.registry import load_provider
from evalharness.reporting.report import write_report
from evalharness.store.db import init_db, session_scope
from evalharness.store.repository import RunRepository

app = typer.Typer(no_args_is_help=True, help="evalanche — reproducible LLM evaluation harness")
console = Console()


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

    prov = load_provider(provider, base_url=settings.ollama_base_url)
    model_version = await prov.resolve_version(model)

    async with session_scope() as session:
        repo = RunRepository(session)
        dataset_id = await repo.upsert_dataset(bundle)
        prompt_template_id = await repo.upsert_prompt_template(
            name=f"{bundle.manifest.name}-template",
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

    executor = Executor(
        provider=prov,
        model=model,
        model_version=model_version,
        template_body=template_body,
    )

    if resume:
        run_id = uuid.UUID(resume)
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
