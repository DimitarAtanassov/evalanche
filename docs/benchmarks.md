# Performance gates

## Purpose

evalanche is meant to scale to large datasets without blowing up memory. This document
records the **performance gates** and how to reproduce them. The design goal is that
throughput and memory are bounded by configuration, not by dataset size.

## Design guarantees

- **Bounded execution.** The executor drives a worker‑queue pool whose queue is bounded
  to twice the worker concurrency, so a run does not create one live task per case (see
  [dataplane.md](dataplane.md#concurrency-backpressure-and-shutdown)).
- **Streaming scoring.** `ScoringEngine` processes rows in batches of 500.
- **Deduplicated embeddings.** The embedding service batches in groups of 64 and
  deduplicates by content hash, so repeated text is embedded once.
- **Lean reports.** Reporting queries a single run and never loads raw provider payloads
  into leadership artifacts.

## The 100k benchmark

`scripts/benchmark_100k.py` scores 100,000 synthetic `exact_match` cases through the
real `ScoringEngine` while tracking peak Python heap with `tracemalloc`:

```bash
uv run python scripts/benchmark_100k.py
```

It prints a JSON result and **fails (`exit 1`) if peak memory exceeds 256 MiB**:

```json
{
  "cases": 100000,
  "passed": 100000,
  "elapsed_s": 0.0,
  "peak_mib": 0.0
}
```

## The gates

| Gate | Threshold | How it's checked |
|------|-----------|------------------|
| Streaming scoring memory | < 256 MiB peak Python heap | `benchmark_100k.py` raises `SystemExit` above the limit |
| Planning task count | < 100 live asyncio tasks | Bounded worker‑queue pool by construction |

When recording release evidence, capture the **machine, Python version, elapsed time,
and peak memory** alongside the run.

## Related

- [dataplane.md](dataplane.md) — the bounded executor and streaming scoring
- [operations.md](operations.md) — dev quality gates (ruff/mypy/pytest)
