# evalanche

Reproducible, resumable LLM evaluation harness. The **`evalctl`** CLI validates
datasets, runs digest‑pinned evaluations, stores **immutable** generations in
PostgreSQL, rescoring them through a **versioned metric catalog**, and emits
statistically honest **JSON / HTML / JUnit** reports.

> **The load‑bearing idea:** generation and scoring are separate, independently
> versioned stages joined by a durable store. *You generate once; you score many
> times.* Re‑scoring historical outputs costs zero inference dollars.

The installable package is **evalanche**; the Python import path is `evalharness`;
the CLI is `evalctl`.

## Capability matrix

| Capability | Status | Where |
|------------|--------|-------|
| Dataset validate + holdout guard | Shipped | `datasets/`, `evalctl dataset-validate` |
| Providers: Ollama, OpenAI‑compatible (revision‑pinned), Mock | Shipped | `providers/` |
| Managed runtime: RPM/TPM token buckets, concurrency, circuit breaker | Shipped | `providers/runtime.py` |
| Bounded resumable executor: retries, cache, three‑layer timeouts | Shipped | `execution/executor.py` |
| Metric catalog: lexical, structured, classification, retrieval, overlap | Shipped | `scoring/catalog.py` |
| Calibration + embedding similarity helpers | Shipped | `scoring/{calibration,embeddings}.py` |
| Zero‑inference score / rescore | Shipped | `evalctl score`, `evalctl runs rescore` |
| Statistics: Wilson, BCa, paired bootstrap, McNemar, BH, Cohen's h, pass@k, power | Shipped | `statistics/` |
| Paired run comparison | Shipped | `evalctl runs compare` |
| Reports: JSON + self‑contained HTML run dashboard + JUnit | Shipped | `reporting/report.py` |
| Per‑slice metric rollups (`dimension=value` beside `__overall__`) | Shipped | `scoring/engine.py` |
| Pipeline observability: Rich progress, structured logs, privacy-safe payload lineage, OTLP tracing | Shipped | `observability.py`, `cli_progress.py` |
| Optional BERTScore | Shipped (extra) | `metrics-ml` extra, `scoring/ml.py` |
| Object storage for raw payloads, LLM‑as‑judge, `evald` HTTP API | Deferred | [`DEFERRED.md`](DEFERRED.md), `docs/` |

## Documentation

New here? Read [**`docs/guide.md`**](docs/guide.md) — the end‑to‑end engineer
onboarding & operations guide. For a role‑based reading order and the full index,
see [**`docs/README.md`**](docs/README.md).

| Doc | Purpose |
|-----|---------|
| [docs/guide.md](docs/guide.md) | Deep onboarding: mental model, CLI, schema, metrics, logs, runbook |
| [docs/architecture.md](docs/architecture.md) | Components, seams, and where‑to‑find‑what module map |
| [docs/dataplane.md](docs/dataplane.md) | Case → generate → score → report; timeouts, retries, coverage |
| [docs/schema.md](docs/schema.md) | PostgreSQL model + Alembic `0003` |
| [docs/metrics.md](docs/metrics.md) | Metric catalog narrative: what each metric is for and how they compose |
| [docs/providers.md](docs/providers.md) | Provider protocol, adapters, limiter/breaker, adding a backend |
| [docs/reports.md](docs/reports.md) | Report artifacts and audience views |
| [docs/operations.md](docs/operations.md) | Local ops, CLI recipes, failure modes, PoC |
| [docs/principles.md](docs/principles.md) | Non‑negotiables that shape every change |
| [docs/benchmarks.md](docs/benchmarks.md) | Performance gates |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker (PostgreSQL via Compose; Ollama optional for live runs)

## Quick start

```bash
docker compose up -d postgres
uv sync --all-extras
cp .env.example .env

uv run alembic upgrade head          # Alembic owns the schema (head: 0003)

# Validate the sample dataset
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

### Rescore & compare (zero inference)

```bash
uv run evalctl runs rescore <run_id> --metrics exact_match,squad_f1,rouge_l
uv run evalctl runs compare <baseline_run_id> <candidate_run_id> --metric exact_match --allow-compatible
```

See [docs/guide.md §4](docs/guide.md#4-cli-command-reference--end-to-end-workflows)
for the full CLI reference and an end‑to‑end baseline‑vs‑candidate workflow.

## Proof of concept

Committed artifacts in [`fixtures/poc/`](fixtures/poc/) prove the full data plane in
CI without pulling models:

- `report.json` / `report.html` — full report from a fixed mock run
- `meta.json` — run id, digest, pass rate, coverage

See [docs/operations.md](docs/operations.md#proof-of-concept).

## Non‑goals

- Not a training or fine‑tuning pipeline
- Not a prompt optimizer
- Not a real‑time production guardrail
- Not a replacement for human review on high‑stakes decisions

## Development

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run pytest -q
```

## License

MIT
