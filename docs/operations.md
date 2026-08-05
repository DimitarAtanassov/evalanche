# Operations

## Local stack

```bash
docker compose up -d postgres   # required
docker compose up -d ollama     # only for live Ollama runs
cp .env.example .env
uv sync --all-extras
uv run python -c "import asyncio; from evalharness.store.db import init_db; asyncio.run(init_db())"
```

Services ([`compose.yaml`](../compose.yaml)):

| Service | Port | Required for |
|---------|------|----------------|
| postgres (pgvector) | 5432 | All runs, tests with DB, PoC |
| ollama | 11434 | Live inference only |

## CLI

`evalctl` is the evalanche command-line interface.

### Validate dataset

```bash
uv run evalctl dataset-validate fixtures/sample_dataset
uv run evalctl dataset-validate path/to/holdout --i-am-doing-a-final-eval
```

### Run evaluation

```bash
# Live Ollama
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama

# Offline mock (CI / PoC)
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa \
  --provider mock \
  --seed 42
```

### Resume

```bash
uv run evalctl run --resume <run_uuid> \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa \
  --provider mock
```

Exit codes: `0` success, `1` validation/config error, `2` coverage below publish floor.

## Proof of concept

Committed artifacts live under [`fixtures/poc/`](../fixtures/poc/). They prove generate → score → report without GPU/Ollama.

```bash
# Regenerate PoC artifacts (requires Postgres)
uv run python scripts/run_poc.py

# Verify against committed golden report
uv run pytest tests/test_poc.py -q
```

CI runs the same path: Postgres service + mock provider + golden report assertions.

## Observability

- Structured JSON logs via structlog (`run_id`, `case_id`, attempt fields when bound)
- OpenTelemetry spans (`case`, `provider.call`) when `OTEL_ENABLED=true`
- `trace_id` persisted on generation rows

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Tests skip `db_ready` | No Postgres | `docker compose up -d postgres` |
| Ollama digest error | Model not pulled / no digest | `ollama pull <model>`; inspect `/api/show` |
| Exit code 2 | Coverage < floor | Inspect harness outcomes; fix infra before publishing |
| Duplicate key on resume | Concurrent writers | Single executor per run_id |

## Quality gates (dev)

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run pytest -q
```
