# evalharness (evalctl)

Production-grade LLM evaluation harness — Phase 1 core loop.

## Phase 1 scope

- `evalctl dataset-validate` — validate JSONL datasets with manifest sidecars
- `evalctl run` — resumable async evaluation against Ollama with digest-pinned models
- Immutable generations in PostgreSQL + pgvector, blobs in MinIO/filesystem
- Exact-match scoring with versioned normalizer
- JSON/HTML reports with Wilson CIs, latency percentiles, outcome histograms

## Non-goals (Phase 1)

- Not a training or fine-tuning pipeline
- Not a prompt optimizer
- Not a real-time production guardrail
- Not a replacement for human review on high-stakes decisions
- Hosted providers, judge subsystem, full metric catalog, and `evald` API are Phase 2+

## Quick start

```bash
# Start infrastructure
docker compose up -d

# Install
uv sync --all-extras

# Copy env
cp .env.example .env

# Initialize database
uv run python -c "import asyncio; from evalharness.store.db import init_db; asyncio.run(init_db())"

# Validate sample dataset
uv run evalctl dataset-validate fixtures/sample_dataset

# Run evaluation (requires Ollama model)
ollama pull llama3.2:1b
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --concurrency 2

# Resume interrupted run
uv run evalctl run --resume <run_id> --dataset fixtures/sample_dataset --template fixtures/templates/qa.jinja --model llama3.2:1b
```

## Generate 500-case fixture

```bash
uv run python scripts/generate_fixture.py --cases 500 --output fixtures/large_dataset
```

## Architecture

Generation and scoring are separate stages joined by a durable store. Generations are immutable;
scores reference generations and can be recomputed without re-running inference.

## Tests

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src/evalharness
```
