"""Opt-in live Ollama smoke for judge and NLI execution."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evalharness.judge.live import run_live_judgment
from evalharness.judge.models import JudgeMode
from evalharness.providers.call_policy import ProviderCallPolicy
from evalharness.providers.config import OllamaConfig
from evalharness.providers.registry import create_provider
from evalharness.rag.live import build_live_rag_evidence

ROOT = Path(__file__).parents[1]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/evalanche-judge-rag-smoke"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = ProviderCallPolicy(
        request_timeout_s=60.0,
        max_retries=2,
        retry_base_s=0.5,
        retry_cap_s=5.0,
    )

    judge_provider = create_provider(OllamaConfig(concurrency=2, rpm=60, tpm=60_000))
    try:
        await run_live_judgment(
            mode=JudgeMode.POINTWISE,
            rubric_path=ROOT / "fixtures/judge/rubric-pointwise.yaml",
            candidates_path=ROOT / "fixtures/judge/candidates-pointwise.jsonl",
            pairs_path=None,
            provider=judge_provider,
            model=args.judge_model,
            judge_family=args.judge_model.split(":", 1)[0],
            candidate_family="smoke-candidate",
            seed=42,
            output_path=args.output_dir / "judgment.json",
            concurrency=2,
            policy=policy,
        )
    finally:
        await judge_provider.aclose()

    nli_provider = create_provider(OllamaConfig(concurrency=2, rpm=60, tpm=60_000))
    try:
        await build_live_rag_evidence(
            report_path=ROOT / "fixtures/rag/report.json",
            evidence_path=ROOT / "fixtures/rag/evidence.jsonl",
            output_path=args.output_dir / "rag_evidence.json",
            provider=nli_provider,
            nli_model=args.nli_model,
            concurrency=2,
            policy=policy,
        )
    finally:
        await nli_provider.aclose()

    print(args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
