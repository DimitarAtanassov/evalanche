# Performance gates

The executor uses a queue bounded to twice worker concurrency. Scoring processes
rows in batches of 500; embedding defaults to batches of 64 and deduplicates by
content hash. Reports query only one run and do not load provider payloads into
leadership artifacts.

Run the 100,000-case synthetic planner/scorer benchmark with:

```bash
uv run python scripts/benchmark_100k.py
```

The gate requires planning to retain fewer than 100 live asyncio tasks and the
streaming scoring benchmark to stay below 256 MiB peak Python heap. Record the
machine, Python version, elapsed time, and peak memory with release evidence.
