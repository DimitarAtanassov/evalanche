# evalanche

Reproducible, resumable LLM evaluation harness. The **`evalctl`** CLI validates datasets, runs digest-pinned evaluations, stores immutable generations in PostgreSQL, rescoring them through a versioned metric catalog, and emits statistically honest JSON/HTML/JUnit reports.

> Generation and scoring are separate stages. You generate once; you score many times.

## Status

| Capability | Available |
|------------|-----------|
| Dataset validate + holdout guard | Yes |
| Ollama + OpenAI-compatible providers (revision-pinned) | Yes |
| Mock provider (CI / PoC) | Yes |
| Bounded resumable executor, limiter, breaker, deadlines | Yes |
| Lexical, structured, classification, calibration, retrieval, overlap metrics | Yes |
| Zero-generation score/rescore | Yes |
| Paired comparison, BCa/Wilson/McNemar/BH, power, pass@k | Yes |
| JSON + self-contained HTML + JUnit analytical reports | Yes |
| Optional BERTScore ML extra | Yes |

## Docs

| Doc | Description |
|-----|-------------|
| [docs/guide.md](docs/guide.md) | **Engineer onboarding & operations** (start here) |
| [docs/architecture.md](docs/architecture.md) | Components and seams |
| [docs/dataplane.md](docs/dataplane.md) | Case → generate → score → report |
| [docs/schema.md](docs/schema.md) | PostgreSQL model |
| [docs/operations.md](docs/operations.md) | Local ops and PoC |
| [docs/principles.md](docs/principles.md) | Non-negotiables |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker (PostgreSQL via Compose; Ollama optional)

## Quick start

```bash
docker compose up -d postgres
uv sync --all-extras
cp .env.example .env

uv run alembic upgrade head

# Validate sample dataset
uv run evalctl dataset-validate fixtures/sample_dataset

# Offline PoC (no GPU / Ollama) — regenerates fixtures/poc/
uv run python scripts/run_poc.py
uv run pytest tests/test_poc.py -q
```

### Live Ollama run

```bash
docker compose up -d ollama
ollama pull llama3.2:1b
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama \
  --concurrency 2
```

### Resume

```bash
uv run evalctl run --resume <run_id> \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b
```

## Proof of concept

Committed artifacts in [`fixtures/poc/`](fixtures/poc/):

- `report.json` / `report.html` — full report from a fixed mock run
- `meta.json` — run id, digest, pass rate, coverage

These prove the data plane in CI without pulling models. See [docs/operations.md](docs/operations.md#proof-of-concept).

## Non-goals

- Not a training or fine-tuning pipeline
- Not a prompt optimizer
- Not a real-time production guardrail
- Not a replacement for human review on high-stakes decisions

## Development

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run pytest -q
```

The installable package name is **evalanche**; the Python import path remains `evalharness` and the CLI remains `evalctl`.

## License

MIT
