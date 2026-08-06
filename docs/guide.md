# evalanche — Onboarding & Operations Guide

> Audience: a new engineer (or on‑call) who needs **full** context on the `evalanche`
> harness — what it does, why it is built this way, how to run it, how to query the
> database, what every metric means, and how to read the logs (including the
> Ollama/llama.cpp container logs).
>
> This guide is written against **`main` after the merge of PR #1 / release `v0.2.0`**
> (`git` commit `9de4d03` and its parents through `19e462a`). The installable package
> is **evalanche**; the Python import path is **`evalharness`**; the CLI is **`evalctl`**.

---

## Scope & ground truth (read this first)

Everything below is grounded in the source on current `main`. After PR #1 the harness
ships the full Phase‑2/3 surface:

| Area | On `main` today |
|------|-----------------|
| CLI | `dataset-validate`, `run`, `score`, `runs rescore`, `runs compare`, `power`, `calibrate`, `judge run|validate|attach-calibration`, `rag evidence` |
| Providers | `ollama`, `mock`, `openai_compatible`, wrapped by `ManagedProvider` (token‑bucket RPM/TPM + concurrency + circuit breaker) |
| Metrics | Full catalog via `MetricRegistry` (lexical, structured, classification, retrieval, overlap) + optional `bertscore_f1` (`metrics-ml` extra) + `EmbeddingService` / calibration helpers |
| Statistics | `statistics/` — Wilson, BCa, paired bootstrap, McNemar, BH, Cohen's h, pass@k, power, flaky‑case detection |
| Schema | Alembic through **`0003_foundation_correctness`** (`metric_config_sha256`, aggregate uniqueness, indexes, NULL‑safe model identity) |
| Reports | JSON + self‑contained HTML run dashboard (Vega‑Lite) + JUnit XML |

There is **no** standalone `evalctl report` command — reports are written by
`evalctl run` (and by library callers such as `scripts/run_release_e2e.py`) via
`reporting.write_report`. The calibrate command is **`evalctl calibrate`**, not
`calibrate-threshold`.

Truly deferred items (object storage, LLM‑as‑judge, `evald` HTTP API, native
Anthropic/Google adapters) live in [§8.4](#84-known-gaps--deferred) and
[`DEFERRED.md`](../DEFERRED.md).

This guide is the deep, example‑driven reference. For the shorter, topic‑focused
design docs — and a role‑based reading order — see the [documentation index](README.md):
[architecture](architecture.md), [data plane](dataplane.md), [schema](schema.md),
[metrics](metrics.md), [providers](providers.md), [reports](reports.md),
[operations](operations.md), and [principles](principles.md).

---

## Table of contents

1. [System overview & mental model](#1-system-overview--mental-model)
2. [Architecture & data plane](#2-architecture--data-plane)
3. [Local setup & running](#3-local-setup--running)
4. [CLI command reference & end‑to‑end workflows](#4-cli-command-reference--end-to-end-workflows)
5. [Database deep‑dive](#5-database-deep-dive)
6. [Metric catalog & statistics](#6-metric-catalog--statistics)
7. [Reading the logs (harness + Ollama/llama.cpp)](#7-reading-the-logs-harness--ollamallamacpp)
8. [Operational runbook, troubleshooting & FAQ](#8-operational-runbook-troubleshooting--faq)

---

## 1. System overview & mental model

### 1.1 What evalanche is

evalanche answers one question honestly and reproducibly: **"How good is model *M*
on dataset *D* under prompt *P* and decode params *θ*?"** — with an interval, a
coverage number, and a paper trail you can re‑run months later without spending a
single inference dollar again.

It is **not** a training pipeline, a prompt optimizer, a production guardrail, or a
substitute for human review on high‑stakes decisions (see the README non‑goals).

### 1.2 The one load‑bearing idea

> **Generation and scoring are separate, independently versioned stages joined by a
> durable store. You generate once; you score many times.**

This single decision shapes the schema, the module boundaries, and the CLI. Raw
provider outputs land in immutable `generations` rows. Metric opinions land in
`scores` rows keyed by `(metric_name, metric_version, metric_config_sha256)`. Changing
a normalizer, adding ROUGE, or recalibrating a threshold never re‑calls the model.

`evalctl run` generates, then calls `ScoringEngine.rescore_run` (default
`exact_match`), then writes reports. `evalctl runs rescore` and `evalctl score`
never accept a provider.

### 1.3 Core invariants (the non‑negotiables)

These come straight from [`docs/principles.md`](principles.md) and are enforced by
code and schema, not by convention:

| # | Invariant | Why it exists | Where it lives in code |
|---|-----------|---------------|------------------------|
| 1 | **Immutability.** Generation rows are written once with provider‑time outcomes. Scores are separate rows. Scoring never updates `generations.outcome`. | A frozen output is the only thing you can honestly re‑score. | `Executor` writes generations; `ScoringEngine` only inserts `scores` / `metric_aggregates`. |
| 2 | **Content addressing.** Datasets, templates, and run configs are SHA‑256 hashed. Different hash ⇒ different run. Upserting `(name, version)` with a different hash is a hard error. | Prevents "we improved 3%" when the dataset silently changed. | `hashing.py`, dataset loader, resume FK/`config_sha256` checks. |
| 3 | **Harness errors ≠ model errors.** Report them separately; exclude harness failures from coverage denominators. | A flaky network must never make a model look worse. | `FailureOutcome`, coverage math in `reporting/report.py`. |
| 4 | **No point estimate without an interval.** Rates ship with a 95% CI (Wilson for binomials; BCa for continuous means / deltas). | A bare "92%" hides whether *n=25* or *n=25,000*. | `statistics.wilson_interval`, `bca_bootstrap`, report CI fields. |
| 5 | **Generation ≠ scoring.** Module and schema boundaries enforce it; rescore must never call a provider. | See §1.2. | `ScoringEngine.rescore_run` docstring + implementation; CLI `runs rescore`. |
| 6 | **Everything versioned.** Dataset, template, model digest, decode params, metric impl, normalizer / metric config. | Reproducibility and honest diffs. | `runs.config_sha256`, `scores.(metric_version, metric_config_sha256)`. |
| 7 | **Deterministic where possible, honest where not.** Record seed/temperature/top_p/top_k and whether the provider honors seeding. | Local models don't guarantee bit‑exact repeats; say so. | `runs.decode_params`, `Capabilities.supports_seed`, flaky‑case detection. |
| 8 | **Resume over restart.** Checkpoint after each generation; resume skips completed `(case_id, repeat_idx)`. | A long run that dies at 90% must not restart from 0. | `UNIQUE (run_id, case_id, repeat_idx)`, idempotent inserts, `--resume`. |
| 9 | **One provider file to extend.** New backend = implement `Provider` + one entry‑point line. | Keeps the blast radius of a new backend tiny. | `providers/registry.py`, `pyproject.toml` entry points. |
| 10 | **Docs match code.** Behavior change ⇒ update `docs/` in the same PR. | This guide is part of that contract. | — |

### 1.4 Coverage and publishability

- **Planned cardinality** = `cases × repeats`
- **Coverage** = `(written_generations − harness_failures) / planned`
- **Publishable** only when `run.status == completed`, `written == planned`, and
  coverage ≥ floor (CLI `--coverage-floor`, default `0.98`)

Pass/fail *quality* lives on `scores.passed` (e.g. `exact_match`), not on mutating
`generations.outcome` after the fact. Generation‑time outcomes encode provider/harness
taxonomy (`passed`, `truncated`, `harness_error`, …).

---

## 2. Architecture & data plane

### 2.1 Component map

```mermaid
flowchart LR
  CLI[evalctl] --> Loader[DatasetLoader + Validator]
  CLI --> Exec[Executor]
  CLI --> ScoreEng[ScoringEngine]
  CLI --> Stats[statistics]
  Loader --> Store[(PostgreSQL 16 + pgvector)]
  Exec --> Reg[ProviderRegistry]
  Reg --> Managed[ManagedProvider]
  Managed --> Limiter[TokenBucket RPM/TPM]
  Managed --> Breaker[CircuitBreaker]
  Managed --> Ollama[Ollama]
  Managed --> OpenAI[OpenAICompatible]
  Managed --> Mock[Mock]
  Exec --> Cache[(response_cache)]
  Exec --> Store
  ScoreEng --> Registry[MetricRegistry]
  Registry --> Catalog[catalog + exact_match + ml]
  ScoreEng --> Store
  CLI --> Reporter[Reporter]
  Reporter --> Store
  Reporter --> Artifacts["JSON / HTML / JUnit"]
```

| Component | Package | Responsibility |
|-----------|---------|----------------|
| Dataset loader + validator | `evalharness.datasets` | `manifest.yaml` + `cases.jsonl` → `Case`; holdout guard |
| Provider registry + runtime | `evalharness.providers` | Entry‑point discovery; `ManagedProvider` rate limit / concurrency / breaker |
| Executor | `evalharness.execution` | Plan, concurrency, retries, cache, checkpoint, resume, outcome taxonomy |
| Scoring engine + registry | `evalharness.scoring` | Versioned metrics, zero‑inference rescore, aggregates |
| Statistics | `evalharness.statistics` | Wilson, BCa, McNemar, BH, pass@k, power, comparisons |
| Store | `evalharness.store` | Async SQLAlchemy; Alembic‑owned schema |
| Reporting | `evalharness.reporting` | Multi‑view JSON/HTML + JUnit; publishability gate |

Seams you must not violate: the **`Provider` protocol**, the **`Metric` protocol**
(aggregation is metric‑specific — never assume `mean()`), **content addressing**, and
**harness‑vs‑model** separation.

### 2.2 Data plane: Case → Generate → Score → Report

```mermaid
sequenceDiagram
  participant Case
  participant Executor
  participant Cache as response_cache
  participant Runtime as ManagedProvider
  participant Provider
  participant Store as generations
  participant Engine as ScoringEngine
  participant Scores as scores/aggregates
  participant Report as Reporter

  Case->>Executor: render(template, inputs)
  Executor->>Cache: lookup(cache_key)
  alt cache hit (temperature == 0 only)
    Cache-->>Executor: GenerationResponse
  else miss
    Executor->>Runtime: generate(request)
    Runtime->>Runtime: acquire RPM/TPM + semaphore; breaker.before_call
    Runtime->>Provider: generate
    Provider-->>Runtime: text + timings + raw
    Runtime-->>Executor: response (+ raw.runtime.queue_wait_ms)
    Executor->>Cache: put(cache_key) when temperature == 0
  end
  Executor->>Store: INSERT generation (immutable)
  Note over Engine,Scores: Separate stage — zero inference
  Engine->>Store: load generations
  Engine->>Scores: INSERT scores + metric_aggregates
  Report->>Scores: read-only aggregates
  Report-->>Report: JSON + HTML + JUnit
```

**Cache key** = SHA‑256 of canonical JSON
`{provider, model_version, rendered_prompt, decode_params, adapter_version}`.
Cache is used only when `temperature == 0.0`. Hits set `generations.cached = true`.
Puts are race‑safe (`ON CONFLICT DO NOTHING`).

### 2.3 Retries, timeouts, limiter, breaker

**Retries (executor‑owned — adapters must not retry):**

- Retry only `RETRYABLE_TRANSIENT` and `RETRYABLE_RATE_LIMIT`
- Full jitter backoff: base `0.5s`, cap `30s`, max 5 retries (settings)
- Honor HTTP `Retry-After` when present (max of jitter and server hint)
- Every attempt appended to `attempt_log`

**Timeouts (three layers):**

1. **Per‑request** — `asyncio.wait_for` around provider `generate` (`default_request_timeout_s`, 60s)
2. **Per‑case** — wall clock including retries (`default_case_timeout_s`, 120s)
3. **Per‑run** — executor wall budget (`default_run_timeout_s`, 4h); drain then `failed` / `cancelled`

**ManagedProvider runtime** (`providers/runtime.py`):

- `TokenBucket` for RPM and TPM (estimated tokens ≈ `len(chars)/3 + max_tokens`)
- `asyncio.Semaphore` for concurrency
- `CircuitBreaker` CLOSED → OPEN after `failure_threshold` (default 5) → HALF_OPEN after
  `recovery_timeout_s` (default 30s)
- Queue wait is recorded on `generations.queue_wait_ms` from `raw.runtime.queue_wait_ms`

### 2.4 Outcome taxonomy

| Outcome | Meaning | In coverage denominator? |
|---------|---------|--------------------------|
| `passed` | Provider returned usable output | Yes |
| `refused` / `truncated` / `empty_output` / `content_filtered` / `model_error` | Model‑side | Yes |
| `harness_timeout` / `harness_error` | Infrastructure | **No** — coverage loss |
| `skipped` | Explicitly skipped | No |

`failed_score` exists as an enum value for historical/forward compatibility; current
`main` does **not** mutate generation rows after scoring. Quality is read from `scores`.

---

## 3. Local setup & running

### 3.1 Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker (Compose services below)

### 3.2 Services (`compose.yaml`)

| Service | Image | Port | Required for |
|---------|-------|------|--------------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | All runs, DB tests, PoC |
| `ollama` | `ollama/ollama:latest` | 11434 | Live inference / embeddings |

Credentials (Compose defaults): user/password/db = `evalharness` / `evalharness` /
`evalharness`.

### 3.3 One‑time bootstrap

```bash
# 1. Start Postgres (add Ollama when you want live runs)
docker compose up -d postgres
# docker compose up -d ollama

# 2. Environment + dependencies
cp .env.example .env
uv sync --all-extras

# 3. Schema — Alembic is the sole schema owner
uv run alembic upgrade head
# equivalent (also called by evalctl run):
# uv run python -c "import asyncio; from evalharness.store.db import init_db; asyncio.run(init_db())"
```

`.env.example` keys: `DATABASE_URL`, `OLLAMA_BASE_URL`, `HARNESS_VERSION`, `GIT_SHA`,
`LOG_LEVEL`, `LOG_FORMAT`, `LOG_PAYLOADS`, `LOG_PAYLOAD_HASHES`,
`LOG_PROGRESS_EVERY`, `OTEL_ENABLED`, and optional
`OTEL_EXPORTER_OTLP_ENDPOINT`. OpenAI‑compatible runs also need
`OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_MODEL_REVISION` (and optionally
an API key).
Phase 6 live scoring budgets are independently configurable with
`JUDGE_PROVIDER_RPM`, `JUDGE_PROVIDER_TPM`, `NLI_PROVIDER_RPM`, and
`NLI_PROVIDER_TPM`.

Head revision is **`0003_foundation_correctness`**. Confirm:

```bash
uv run alembic current
# or
PGPASSWORD=evalharness psql -h localhost -U evalharness -d evalharness \
  -c "SELECT version_num FROM alembic_version;"
```

### 3.4 Smoke test: offline PoC (no GPU / Ollama)

```bash
uv run evalctl dataset-validate fixtures/sample_dataset
uv run python scripts/run_poc.py
uv run pytest tests/test_poc.py -q
```

Committed artifacts under `fixtures/poc/` prove generate → score → report without
pulling models.

### 3.5 Dev quality gates

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run pytest -q
```

Optional 100k planner/scorer memory gate: `uv run python scripts/benchmark_100k.py`
(see [`docs/benchmarks.md`](benchmarks.md)).

---

## 4. CLI command reference & end‑to‑end workflows

Verified with `uv run evalctl --help` and each subcommand's `--help` on this tree.

```text
evalctl
├── dataset-validate
├── run
├── score
├── judge
│   ├── run
│   ├── validate
│   └── attach-calibration
├── rag
│   └── evidence
├── runs
│   ├── rescore
│   └── compare
├── power
└── calibrate
```

Exit codes for `run`: `0` success + publishable, `1` validation/config error, `2`
coverage below floor / not publishable.

### 4.1 `evalctl dataset-validate`

```text
Usage: evalctl dataset-validate [OPTIONS] DATASET_DIR
  --i-am-doing-a-final-eval   Allow holdout split evaluation
```

```bash
uv run evalctl dataset-validate fixtures/sample_dataset
# Holdout sets are blocked unless you mean it:
# uv run evalctl dataset-validate path/to/holdout --i-am-doing-a-final-eval
```

### 4.2 `evalctl run`

```text
Usage: evalctl run [OPTIONS]
  --dataset PATH          Dataset directory          [required]
  --template PATH         Prompt template file       [required]
  --model TEXT            Model name                 [required]
  --provider TEXT         Provider name              [default: ollama]
  --output PATH           Report output dir          [default: reports]
  --repeats INT           Repeats per case           [default: 1]
  --concurrency INT       Max concurrent requests    [default: 2]
  --temperature FLOAT                                [default: 0.0]
  --max-tokens INT
  --seed INT
  --resume TEXT           Resume existing run ID
  --i-am-doing-a-final-eval
  --coverage-floor FLOAT                             [default: 0.98]
  --tenant TEXT                                      [default: default]
```

After generations complete, `run` rescores with the pack's declared
`task_metrics` (falling back to `exact_match` for a legacy manifest), headlines
the report pass rate on the first of them, and writes `{run_id}.json`,
`{run_id}.html`, and `{run_id}.xml` under `--output`.

### 4.3 `evalctl score`

Score supplied JSONL **without inference**.

```text
Usage: evalctl score [OPTIONS] INPUTS
  --metrics TEXT   Comma-separated metric names [default: exact_match]
```

Each JSONL row may include `id`, `task_type`, `inputs`, `reference` / `references`,
`expected_label`, `expected_json`, `qrels`, and `output`.

```bash
uv run evalctl score /tmp/outputs.jsonl --metrics exact_match,squad_f1,rouge_l
```

### 4.4 `evalctl runs rescore`

Idempotently rescore **stored** generations — zero inference.

```text
Usage: evalctl runs rescore [OPTIONS] RUN_ID
  --metrics TEXT   [default: exact_match]
```

```bash
uv run evalctl runs rescore "$RUN_ID" --metrics exact_match,squad_f1,assertions
```

Prints `{"run_id": "...", "scores_processed": N, "inference_calls": 0}`. Same
metric+config is upsert‑safe; a changed normalizer / metric config hash writes a
**new** score row beside the old one.

### 4.5 `evalctl runs compare`

Paired comparison of aligned `(case, repeat)` outcomes.

```text
Usage: evalctl runs compare [OPTIONS] BASELINE_RUN_ID CANDIDATE_RUN_ID
  --metric TEXT
  --allow-compatible          Permit same-dataset compare when template/config differ
  --output PATH
```

Identity requires matching `dataset_id`, `prompt_template_id`, and `config_sha256`.
`--allow-compatible` relaxes to same dataset + same `repeats` (deliberate A/B of
different templates/models). Flaky cases (disagreement across repeats within either
run) are excluded from the claim and listed in `excluded_flaky_cases`. Output includes
McNemar p‑value, paired BCa CI on the delta, Cohen's h, and BH significance.

### 4.6 `evalctl power`

```text
Usage: evalctl power [OPTIONS]
  --baseline-rate FLOAT   [0..1] required
  --mde FLOAT             minimum detectable effect (required)
  --power FLOAT           [default: 0.8]
  --alpha FLOAT           [default: 0.05]
```

```bash
uv run evalctl power --baseline-rate 0.70 --mde 0.05 --power 0.8 --alpha 0.05
# -> {"sample_size_per_arm": ..., "power": 0.8}
```

### 4.7 `evalctl calibrate`

Development‑only threshold selection from JSONL `{label, score}` rows. **Never** fit
thresholds on holdout.

```text
Usage: evalctl calibrate [OPTIONS] INPUTS
```

```bash
uv run evalctl calibrate dev-similarities.jsonl
# -> roc_auc, pr_auc, threshold, dev_f1
```

### 4.8 Live judge and RAG NLI

Live Phase 6 scoring is file-primary and requires no database. Ollama resolves the
installed model digest. OpenAI-compatible endpoints require
`OPENAI_COMPATIBLE_BASE_URL` and an immutable
`OPENAI_COMPATIBLE_MODEL_REVISION`.

```bash
uv run evalctl judge run \
  --mode pointwise \
  --rubric fixtures/judge/rubric-pointwise.yaml \
  --candidates fixtures/judge/candidates-pointwise.jsonl \
  --provider ollama \
  --model llama3.2:1b \
  --judge-family llama \
  --candidate-family qwen \
  --seed 42 \
  --concurrency 2 \
  --request-timeout 60 \
  --output /tmp/judgment.json

uv run evalctl rag evidence \
  --report fixtures/rag/report.json \
  --evidence fixtures/rag/evidence.jsonl \
  --nli-provider ollama \
  --nli-model llama3.2:1b \
  --concurrency 2 \
  --request-timeout 60 \
  --output /tmp/rag_evidence.json
```

`--responses` and `--nli-responses` remain required only for the hermetic `mock`
paths. Live outputs must satisfy versioned JSON schemas. A malformed or
out-of-range pointwise result, either failed pairwise ordering, or an invalid NLI
label aborts without writing a partial artifact. All new judgment and RAG
artifacts retain `gating_allowed: false`.

The opt-in smoke command calls both live paths and is not part of default pytest:

```bash
uv run python scripts/live_judge_rag_smoke.py \
  --judge-model llama3.2:1b \
  --nli-model llama3.2:1b
```

### 4.9 Workflow A — offline mock (works anywhere Postgres is up)

```bash
docker compose up -d postgres
uv run alembic upgrade head

uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa \
  --provider mock \
  --seed 42 \
  --output reports
```

### 4.10 Workflow B — live Ollama baseline + candidate (release‑style)

Matches the pattern in `scripts/run_release_e2e.py` (smaller dataset for a laptop):

```bash
docker compose up -d postgres ollama
ollama pull llama3.2:1b
# optional for semantic rescoring demos:
# ollama pull nomic-embed-text

uv run evalctl dataset-validate fixtures/sample_dataset

# Baseline
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama \
  --repeats 3 \
  --concurrency 2 \
  --temperature 0.2 \
  --max-tokens 32 \
  --tenant demo-ab \
  --output reports
# note the printed Run ID → BASELINE_RUN_ID

# Candidate (different template and/or model; same dataset)
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama \
  --repeats 3 \
  --concurrency 2 \
  --temperature 0.2 \
  --max-tokens 32 \
  --tenant demo-ab \
  --output reports
# note CANDIDATE_RUN_ID

# Zero-inference multi-metric rescore on both runs
uv run evalctl runs rescore "$BASELINE_RUN_ID" \
  --metrics exact_match,squad_f1,normalized_levenshtein,assertions
uv run evalctl runs rescore "$CANDIDATE_RUN_ID" \
  --metrics exact_match,squad_f1,normalized_levenshtein,assertions

# Paired compare (allow-compatible because templates/config may differ)
uv run evalctl runs compare "$BASELINE_RUN_ID" "$CANDIDATE_RUN_ID" \
  --metric exact_match \
  --allow-compatible \
  --output reports/comparison.json

# Resume a partial run (same inputs required)
# uv run evalctl run --resume "$BASELINE_RUN_ID" \
#   --dataset fixtures/sample_dataset \
#   --template fixtures/templates/qa.jinja \
#   --model llama3.2:1b --provider ollama ...
```

For the full published `v0.2.0` evidence bundle (500 cases × 5 repeats, semantic
similarity via `EmbeddingService`, comparison artifact):

```bash
uv run python scripts/run_release_e2e.py
```

Ollama version pinning resolves digests from **`GET /api/tags`** (fallback to
`/api/show`). Pull the model before running or resolve will fail with "has no digest".

---

## 5. Database deep‑dive

PostgreSQL 16 + `pgvector`. This is the **only** persistence service — raw provider
payloads live in `generations.raw_response` (JSONB), not object storage (see
[`DEFERRED.md`](../DEFERRED.md)). **Alembic is the sole schema owner**; `init_db()`
runs `alembic upgrade head` and never `create_all`.

### 5.1 Connecting

App URL (async):

```text
postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness
```

Interactive `psql`:

```bash
# A) exec into the container
docker exec -it evalv1-postgres-1 psql -U evalharness -d evalharness

# B) local client against the mapped port
PGPASSWORD=evalharness psql -h localhost -p 5432 -U evalharness -d evalharness
```

Live head on a healthy local stack: `0003_foundation_correctness`.

### 5.2 Entity‑relationship model

```mermaid
erDiagram
  datasets ||--o{ cases : contains
  datasets ||--o{ runs : evaluated_by
  prompt_templates ||--o{ runs : uses
  model_versions ||--o{ runs : pins
  runs ||--o{ generations : produces
  cases ||--o{ generations : for
  generations ||--o{ scores : scored_as
  runs ||--o{ metric_aggregates : rolls_up
  runs ||--o{ runs : baseline
  model_versions ||--o{ embeddings : embedded_by
  model_versions ||--o{ judgments : judges
  generations ||--o{ judgments : judged
  cases ||--o{ annotations : labeled
  generations ||--o{ annotations : labeled
```

Three groups:

- **Definitions** (append‑once, content‑addressed): `datasets`, `cases`,
  `prompt_templates`, `model_versions`
- **Run + immutable outputs**: `runs`, `generations`, `scores`, `metric_aggregates`
- **Forward‑compatible / cache**: `judgments`, `annotations`, `embeddings`,
  `response_cache`

### 5.3 Table‑by‑table reference

All types/constraints below match `src/evalharness/store/models.py` and the live
`\d+` output after migration `0003`.

#### `datasets`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | surrogate |
| `name` | text | no | e.g. `synthetic-qa` |
| `version` | text | no | e.g. `1.0.0` |
| `content_sha256` | text | no | hash of JSONL body — true identity |
| `split` | text | no | `dev` / `holdout` / … |
| `manifest` | jsonb | no | license, pii_scrubbed, slices, … |
| `created_at` | timestamptz | no | `now()` |

**Constraint:** `UNIQUE (name, version)`.

#### `cases`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `dataset_id` | bigint FK→datasets | no | |
| `external_id` | text | no | stable join key across runs |
| `task_type` | text | no | `TaskType` value |
| `inputs` | jsonb | no | template variables |
| `reference` | jsonb | yes | packed answer fields |
| `qrels` | jsonb | yes | graded relevance for retrieval |
| `slices` | jsonb | no, default `{}` | analysis dimensions |
| `weight` | float | no, default `1.0` | reserved |

**Constraints / indexes:** `UNIQUE (dataset_id, external_id)`; GIN
`ix_cases_slices` on `slices jsonb_path_ops`.

Repository packs `reference_answer`, `references`, `expected_label`,
`expected_json`, `must_contain`, `must_not_contain` into the `reference` JSONB.

#### `prompt_templates`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `name` / `version` | text | no | |
| `body` | text | no | raw template |
| `content_sha256` | text | no | feeds `config_sha256` |

**Constraint:** `UNIQUE (name, version)`.

#### `model_versions`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `provider` | text | no | `ollama` / `mock` / `openai_compatible` |
| `model` | text | no | requested name |
| `resolved_version` | text | no | **digest / pinned revision** |
| `quantization` | text | yes | e.g. `Q8_0` |
| `params_b` | float | yes | |
| `context_window` | int | yes | |
| `capabilities` | jsonb | no | seed/tools/streaming flags |

**Index (0003):** unique `uq_model_versions_identity` on
`(provider, model, resolved_version, COALESCE(quantization, ''))` — NULL‑safe so a
missing quant does not collide incorrectly.

#### `runs`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | uuid PK | no | |
| `dataset_id` / `prompt_template_id` / `model_version_id` | FK | no | content‑addressed inputs |
| `decode_params` | jsonb | no | temperature/max_tokens/seed/top_p/top_k/stop |
| `config_sha256` | text | no | run identity hash |
| `harness_version` / `git_sha` | text | no | from settings |
| `repeats` | int | no, default 1 | |
| `status` | text | no | `queued`→`running`→`completed`\|`failed`\|`cancelled` |
| `tenant_id` | text | no | |
| `started_at` / `finished_at` | timestamptz | yes | |
| `baseline_run_id` | uuid FK→runs | yes | optional A/B link |

**Index (0003):** `ix_runs_status_started_at` on `(status, started_at)`.

#### `generations` — immutable outputs

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `run_id` | uuid FK→runs | no | |
| `case_id` | bigint FK→cases | no | |
| `repeat_idx` | int | no, default 0 | |
| `output` | text | yes | model text |
| `tool_calls` | jsonb | yes | |
| `finish_reason` | text | yes | `stop` / `length` / … |
| `outcome` | text | no | generation‑time taxonomy |
| `prompt_tokens` / `completion_tokens` | int | yes | |
| `cost_usd` | numeric(12,6) | yes | |
| `ttft_ms` / `total_ms` | float | yes | latency |
| `queue_wait_ms` | float | yes | from ManagedProvider runtime |
| `attempts` | int | no, default 1 | |
| `attempt_log` | jsonb | yes | retry forensics |
| `cached` | bool | no, default false | |
| `raw_response` | jsonb | yes | full provider payload |
| `trace_id` | text | yes | OTel linkage |
| `created_at` | timestamptz | no | |

**Constraints / indexes:** `UNIQUE (run_id, case_id, repeat_idx)`;
`ix_generations_run_id`; `ix_generations_run_case_repeat` (0003).

#### `scores`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `generation_id` | bigint FK→generations | no | |
| `metric_name` / `metric_version` | text | no | |
| `metric_config_sha256` | text | no | normalizer / metric config identity |
| `value` | float | yes | |
| `passed` | bool | yes | |
| `detail` | jsonb | yes | |
| `scored_at` | timestamptz | no | |

**Constraints / indexes:**
`UNIQUE (generation_id, metric_name, metric_version, metric_config_sha256)`;
`ix_scores_generation_id` (0003).

#### `metric_aggregates`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `run_id` | uuid FK→runs | no | |
| `metric_name` / `metric_version` | text | | |
| `metric_config_sha256` | text | no | **added/enforced by 0003** |
| `slice_key` | text | no, default `__overall__` | `__overall__`, or `dimension=value` from `cases.slices` |
| `n` | int | no | |
| `value` | float | no | |
| `ci_low` / `ci_high` | float | yes | |
| `stddev` | float | yes | |
| `method` | text | yes | e.g. `wilson`, `mean+wilson`, `BCa-…` |

**Constraint (0003):** `uq_metric_aggregates_identity`
`UNIQUE (run_id, metric_name, metric_version, slice_key, metric_config_sha256)`.

`ScoringEngine.rescore_run` writes one `__overall__` row per metric plus one row per
`dimension=value` found in `cases.slices`. A dimension with more than
`max_slice_cardinality` (default 50) distinct values is skipped rather than emitting one
aggregate row per case.

#### `response_cache`

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `cache_key` | text PK | no | |
| `response` | jsonb | no | serialized generation payload |
| `created_at` | timestamptz | no | |

#### Forward‑compatible tables

- **`judgments`** — LLM‑as‑judge / pairwise (rubric, score, preference, swap_position,
  reasoning, evidence, cost). Schema present; judge subsystem not shipped.
- **`annotations`** — human labels (`annotator_id`, `label` JSONB, `adjudicated`).
- **`embeddings`** — `content_sha256` + `embedding_model_version_id` +
  `vec vector(1024)`, `UNIQUE (content_sha256, embedding_model_version_id)`.
  `EmbeddingService` today dedupes in‑process; durable pgvector writes / HNSW are
  not yet wired on the hot path. **Dim caveat:** column is fixed `vector(1024)`;
  `nomic-embed-text` is 768‑d — release e2e scores cosine in application memory.

### 5.4 What migration `0003` added

From `alembic/versions/0003_foundation_correctness.py`:

1. `metric_aggregates.metric_config_sha256 NOT NULL` (+ backfill `legacy-unversioned`)
2. Unique identity on aggregates including that hash
3. Btree indexes: `ix_generations_run_id`, `ix_scores_generation_id`,
   `ix_runs_status_started_at`, `ix_generations_run_case_repeat`
4. NULL‑safe unique index `uq_model_versions_identity`

### 5.5 Query library

Replace `:run` with a UUID. Grab recent runs first:

```sql
SELECT id, status, tenant_id, started_at, finished_at, config_sha256
FROM runs
ORDER BY started_at DESC
LIMIT 5;
```

**1. Pass rate from scores + coverage from generations**

```sql
-- Coverage (harness failures excluded from numerator)
SELECT
  count(*) AS written,
  count(*) FILTER (WHERE outcome IN ('harness_error','harness_timeout')) AS harness_failures,
  round(
    (count(*) - count(*) FILTER (WHERE outcome IN ('harness_error','harness_timeout')))::numeric
    / nullif(count(*), 0), 4
  ) AS coverage_vs_written
FROM generations
WHERE run_id = :run;

-- Quality pass rate (exact_match scores — the report's primary rate)
SELECT
  count(*) AS n,
  count(*) FILTER (WHERE passed) AS passed,
  round(avg(value)::numeric, 4) AS mean_value
FROM scores s
JOIN generations g ON g.id = s.generation_id
WHERE g.run_id = :run AND s.metric_name = 'exact_match' AND s.passed IS NOT NULL;
```

*How to read it:* Coverage uses planned cardinality in the app (see query 3). Quality
comes from `scores`, not from treating `generations.outcome = 'passed'` as a grade.

**2. Pass rate by slice**

```sql
SELECT
  c.slices ->> 'parity' AS parity,   -- or 'difficulty', etc.
  count(*) FILTER (WHERE s.passed) AS passed,
  count(*) AS n,
  round(avg(s.value)::numeric, 4) AS mean_value
FROM scores s
JOIN generations g ON g.id = s.generation_id
JOIN cases c ON c.id = g.case_id
WHERE g.run_id = :run AND s.metric_name = 'exact_match' AND s.passed IS NOT NULL
GROUP BY 1
ORDER BY 1;
```

*How to read it:* Tiny `n` ⇒ wide Wilson interval even if the point estimate looks
great. Prefer slice keys declared in the dataset manifest.

**3. Coverage vs planned**

```sql
SELECT
  (SELECT count(*) FROM cases c
     JOIN runs r ON r.dataset_id = c.dataset_id WHERE r.id = :run)
  * (SELECT repeats FROM runs WHERE id = :run) AS planned,
  count(*) AS produced,
  count(*) FILTER (WHERE outcome IN ('harness_error','harness_timeout')) AS harness_failures
FROM generations
WHERE run_id = :run;
```

*How to read it:* `produced < planned` ⇒ incomplete / resumable. `produced = planned`
with high harness failures ⇒ finished but may fail the publish floor.

**4. Harness vs model failure taxonomy**

```sql
SELECT outcome, count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM generations
WHERE run_id = :run
GROUP BY outcome
ORDER BY n DESC;
```

*How to read it:* `harness_*` is infra; everything else is model‑side. Fix infra and
`--resume` before drawing quality conclusions.

**5. Latency percentiles**

```sql
SELECT
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p50,
  round(percentile_cont(0.90) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p90,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p95,
  round(percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p99,
  round(max(total_ms)::numeric, 2) AS max,
  round(avg(total_ms)::numeric, 2) AS mean
FROM generations
WHERE run_id = :run AND total_ms IS NOT NULL;
```

*How to read it:* Matches the report's latency block. Prefer p95/p99 over mean for
skewed decode latency.

**6. TTFT vs decode**

```sql
SELECT
  round(avg(ttft_ms)::numeric, 1) AS avg_ttft,
  round(avg(total_ms - ttft_ms)::numeric, 1) AS avg_after_first_token,
  round(avg(queue_wait_ms)::numeric, 1) AS avg_queue_wait,
  round(avg(total_ms)::numeric, 1) AS avg_total
FROM generations
WHERE run_id = :run AND ttft_ms IS NOT NULL AND total_ms IS NOT NULL;
```

*How to read it:* High TTFT ↔ prompt‑eval / cold load; high after‑first‑token ↔ decode.
Maps to llama.cpp `print_timing` in §7.

**7. Cache hit rate**

```sql
SELECT
  count(*) FILTER (WHERE cached) AS hits,
  count(*) AS total,
  round(100.0 * count(*) FILTER (WHERE cached) / nullif(count(*), 0), 1) AS hit_pct
FROM generations
WHERE run_id = :run;
```

*How to read it:* Cache only engages at `temperature == 0`. Unexpected misses usually
mean digest, template bytes, or decode params changed.

**8. Per‑metric aggregates from score rows**

```sql
SELECT
  s.metric_name, s.metric_version, left(s.metric_config_sha256, 12) AS config12,
  count(*) AS n,
  count(*) FILTER (WHERE s.passed) AS passed,
  round(avg(s.value)::numeric, 4) AS mean_value
FROM scores s
JOIN generations g ON g.id = s.generation_id
WHERE g.run_id = :run
GROUP BY 1, 2, 3
ORDER BY 1, 3;
```

*How to read it:* Same `metric_name`, different `config12` ⇒ different scoring opinion
over identical generations (e.g. normalizer change).

**9. Persisted rollups**

```sql
SELECT metric_name, metric_version, slice_key, n, value,
       ci_low, ci_high, method, left(metric_config_sha256, 12) AS config12
FROM metric_aggregates
WHERE run_id = :run
ORDER BY metric_name, slice_key;
```

**10. Flaky cases across repeats**

```sql
SELECT
  c.external_id,
  count(*) AS n_samples,
  count(*) FILTER (WHERE s.passed) AS n_passed,
  count(DISTINCT g.output) AS distinct_outputs
FROM generations g
JOIN cases c ON c.id = g.case_id
LEFT JOIN scores s ON s.generation_id = g.id AND s.metric_name = 'exact_match'
WHERE g.run_id = :run
GROUP BY c.external_id
HAVING count(*) FILTER (WHERE s.passed) NOT IN (0, count(*))
    OR count(DISTINCT g.output) > 1
ORDER BY distinct_outputs DESC, c.external_id;
```

*How to read it:* `runs compare` excludes these from claims. `distinct_outputs > 1` at
`temperature=0` means the backend is not bit‑deterministic.

**11. Compare two runs (disagreements)**

```sql
WITH a AS (
  SELECT c.external_id, bool_or(s.passed) AS passed
  FROM scores s
  JOIN generations g ON g.id = s.generation_id
  JOIN cases c ON c.id = g.case_id
  WHERE g.run_id = :run_a AND s.metric_name = 'exact_match' AND s.passed IS NOT NULL
  GROUP BY c.external_id
),
b AS (
  SELECT c.external_id, bool_or(s.passed) AS passed
  FROM scores s
  JOIN generations g ON g.id = s.generation_id
  JOIN cases c ON c.id = g.case_id
  WHERE g.run_id = :run_b AND s.metric_name = 'exact_match' AND s.passed IS NOT NULL
  GROUP BY c.external_id
)
SELECT a.external_id, a.passed AS run_a_passed, b.passed AS run_b_passed
FROM a JOIN b USING (external_id)
WHERE a.passed IS DISTINCT FROM b.passed
ORDER BY a.external_id;
```

*How to read it:* Wins vs regressions. Prefer `evalctl runs compare` for McNemar /
BCa / BH rather than eyeballing only this list.

**12. Inspect raw_response / attempt_log**

```sql
SELECT
  g.id, c.external_id, g.outcome, g.finish_reason,
  g.ttft_ms, g.total_ms, g.queue_wait_ms, g.attempts, g.cached,
  jsonb_pretty(g.attempt_log) AS attempt_log,
  jsonb_pretty(g.raw_response #> '{chunks,-1}') AS last_raw_chunk,
  left(g.output, 300) AS output_preview
FROM generations g
JOIN cases c ON c.id = g.case_id
WHERE g.id = :generation_id;
```

*How to read it:* Ollama `raw_response` is `{"chunks":[…]}` plus optional
`runtime` from ManagedProvider. Final chunk carries `done_reason`,
`prompt_eval_count`, `eval_count`.

**13. Embedding table lookups**

```sql
SELECT e.id, e.content_sha256, mv.model, mv.resolved_version,
       vector_dims(e.vec) AS dims
FROM embeddings e
JOIN model_versions mv ON mv.id = e.embedding_model_version_id
LIMIT 20;
-- Once populated at matching dimension:
-- SELECT content_sha256, vec <=> :query_vec AS cosine_distance
-- FROM embeddings ORDER BY vec <=> :query_vec LIMIT 10;
```

*How to read it:* Often empty on current paths (in‑memory `EmbeddingService` cache).
`<=>` is pgvector cosine distance.

---

## 6. Metric catalog & statistics

### 6.0 Metric contract & registry

Every metric implements `Metric` (`core/protocols.py`): `name`, `version`,
`task_types`, `requires`, `score()`, `aggregate()`. Aggregation is **metric‑specific**.

`MetricRegistry.defaults()` registers `exact_match` plus all `builtin_metrics()` from
`scoring/catalog.py`. Entry points under `evalharness.metrics` add more (today:
`bertscore` → `BERTScoreMetric`). Discover names:

```bash
uv run python -c "from evalharness.scoring.registry import MetricRegistry; \
print(MetricRegistry.defaults().names())"
```

Current built‑ins:
`assertions`, `chrf_pp`, `classification`, `exact_match`, `json_field_f1`,
`json_validity`, `meteor`, `normalized_levenshtein`, `numeric_assertion`,
`retrieval_ndcg_10`, `rouge_l`, `sacrebleu`, `squad_f1`.

`evalctl run` scores the manifest's `task_metrics`, defaulting to `exact_match`
only when the manifest declares none; use `runs rescore --metrics …` for the
rest.

### 6.1 Deterministic / lexical

#### `exact_match` v1.0.0

- **Definition:** normalized prediction equals normalized reference.
- **Normalizer** (`NormalizerConfig` v1.0.0): unicode NFKC, lowercase, strip
  punctuation, strip articles `{a,an,the}`, collapse whitespace; optional
  `numeric_tol`. Config hash = `metric_config_sha256` on each score.
- **Range:** value ∈ `{0.0, 1.0}` (or `NULL` if missing ref/output); `passed` mirrors value.
- **Aggregate:** pass rate + **Wilson 95% CI** (`method=wilson`).
- **When to use:** short‑form QA with a single canonical answer.
- **Gotcha:** over‑normalizing can hide real errors; under‑normalizing inflates fails.
  Never change the normalizer silently — the config hash is the audit trail.

#### `squad_f1` v1.0.0

- **Formula:** token bags (casefold `\w+`); precision/recall from multiset overlap;
  \(F_1 = 2PR/(P+R)\).
- **Range:** `[0, 1]`; default pass threshold `0.5`.
- **Aggregate:** mean + Wilson over thresholded successes (`mean+wilson`).
- **When to use:** extractive / short answers with paraphrase tolerance.
- **Gotcha:** still lexical — synonyms without shared tokens score 0.

#### `normalized_levenshtein` v1.0.0

- **Formula:** RapidFuzz `normalized_similarity` (1 − distance / max length).
- **Default threshold:** `0.8` (configurable).
- **When to use:** near‑copy answers, OCR‑ish noise.
- **Gotcha:** calibrate the threshold on **dev** (`evalctl calibrate`); do not reuse a
  magic 0.8 on holdout without recording ROC‑AUC.

#### `assertions` v1.0.0

- **Definition:** all `must_contain` terms present (casefold) and all
  `must_not_contain` absent. Value `1.0` / `0.0`; threshold `1.0`.
- **Requires:** nothing beyond generation output (uses case assertion lists).
- **When to use:** content constraints / safety denylist smoke checks.

#### `numeric_assertion` v1.0.0

- **Definition:** extract numbers from prediction and reference; require equal counts
  and `math.isclose` with abs/rel tol `1e-6`.
- **When to use:** arithmetic / unit answers embedded in text.

#### `json_validity` / `json_field_f1`

- **`json_validity`:** `json.loads` (+ optional `inputs.json_schema` via jsonschema).
  Value 1/0. Task types: extraction, generation, tool_use.
- **`json_field_f1`:** flatten nested objects/arrays to dotted / indexed keys; field
  precision/recall/F1 vs `expected_json`.
- **When to use:** structured extraction. **Gotcha:** invalid JSON ⇒ field F1 = 0, not
  NULL — distinguish via detail/`json_validity`.

### 6.2 Classification

#### `classification` v1.0.0

- **Per‑case:** strip equality to `expected_label` (value 1/0).
- **Aggregate value:** accuracy with Wilson CI; **method** JSON encodes
  balanced accuracy, macro/micro/weighted F1, weighted P/R, **MCC**, Cohen's κ.
- **When to use:** label tasks. Prefer MCC under class imbalance.
- **Gotcha:** free‑form model chatter fails exact label match — constrain decode or
  post‑parse before scoring.

### 6.3 Calibration (helpers, not registry metrics)

`scoring/calibration.py` (invoked by `evalctl calibrate` and callable from research
code):

| Metric | Definition | Notes |
|--------|------------|-------|
| Adaptive ECE | Equal‑mass bins (default 15); weighted \|acc − conf\| | Needs aligned `correct` + `confidence` |
| Brier | mean `(p − y)²` | |
| NLL | binary cross‑entropy | clips probs away from {0,1} |
| Risk–coverage / AURC | sort by confidence; trapezoid of risk vs coverage | |
| Accuracy @ 80% coverage | selective prediction operating point | |
| ROC‑AUC / PR‑AUC | when both classes present | |
| `calibrate_threshold` | argmax F1 on PR curve | **dev only**; returns threshold + AUC + dev_f1 |

**Gotcha:** confidence must be real (logprobs or elicited). Without it, skip this
section rather than inventing scores.

### 6.4 Ranking / retrieval

#### `retrieval_ndcg_10` v1.0.0

Requires `qrels`. Ranking from JSON list or comma‑separated ids (stable unique order).

For cutoffs \(k \in \{1,3,5,10,20\}\) detail includes:

- **P@k** = \|relevant ∩ top‑k\| / k
- **R@k** = \|relevant ∩ top‑k\| / \|relevant\|
- **Hit@k** = 1 if any relevant in top‑k
- **MRR** = 1 / rank of first relevant
- **MAP** = mean AP over relevant items
- **Primary value:** **NDCG@10** with exponential gain
  \(\mathrm{DCG}=\sum_i (2^{rel_i}-1)/\log_2(i+1)\), normalized by ideal DCG
- **recall_ceiling** = min(1, \|ranking\| / \|relevant\|)

Zero‑relevance queries return `NULL` with `excluded: zero_relevance`. Ties break by
original order (`dict.fromkeys`).

**When to use:** retrieval / RAG ranking eval. **Gotcha:** linear‑gain NDCG ≠ this
implementation — always state exponential gain.

### 6.5 Summarization / generation overlap

These measure **surface overlap** and correlate weakly with human judgment on
abstractive tasks. Use as regression tripwires, not as quality gospel.

| Metric | Definition | Range | Aggregate notes |
|--------|------------|-------|-----------------|
| `rouge_l` | rouge‑score ROUGE‑1/2/L/Lsum when importable; else deterministic LCS/ngram fallback | F ∈ [0,1] | mean+wilson on thresholded values |
| `sacrebleu` | sentence BLEU stored per case; **corpus BLEU** on aggregate with SacreBLEU signature | score/100 | Do **not** average sentence BLEUs for the headline number |
| `chrf_pp` | sacrebleu `sentence_chrf` word_order=2 (chrF2++) | [0,1] | Better than BLEU for morphologically rich languages |
| `meteor` | NLTK METEOR with wordnet; returns NULL if resources missing | [0,1] or NULL | Declare language/resources; install NLTK data in CI |
| `bertscore_f1` (optional extra) | pinned `microsoft/deberta-xlarge-mnli` @ revision `7d9f5b4`, layer 40, baseline rescale | F1 | Heavy; isolate behind `metrics-ml` |

### 6.6 Semantic similarity (embeddings)

`scoring/embeddings.py` — `EmbeddingService`:

- Pins `model` + `revision`; default dimension **1024** (release e2e overrides to 768
  for `nomic-embed-text`)
- Dedupes by content SHA‑256 in memory; **L2‑normalizes** before cosine
- `cosine_max_reference` — max cosine vs each reference
- `asymmetric_similarity` — cosine vs reference centroid with explicit variant label
  for threshold provenance

There is **no** `semantic_similarity` entry in `MetricRegistry` today. Release evidence
writes those scores explicitly (see `scripts/run_release_e2e.py`) with BCa CIs on the
mean. Thresholds belong on **dev** via `evalctl calibrate` (report ROC‑AUC / PR‑AUC /
operating point) — never a silent magic `0.8` on holdout.

**Gotcha:** `embeddings.vec` is `vector(1024)` while some local embedders are 768‑d;
keep dimension in the metric config hash and avoid forcing mismatched inserts.

### 6.7 Statistics package

Implemented in `evalharness.statistics` (also re‑exported helpers in
`scoring/stats.py` for Wilson/percentiles used by exact match / reporting).

| Tool | Formula / behavior | Default | When to use |
|------|--------------------|---------|-------------|
| **Wilson interval** | Score interval for binomial \(p\) | 95% (z from Normal) | Pass rates, any Bernoulli aggregate |
| **BCa bootstrap** | Bias‑corrected accelerated CI via `scipy.stats.bootstrap` | 10 000 resamples, seed 0 | Means of continuous metrics (ROUGE, cosine, …) |
| **Paired bootstrap** | BCa on paired deltas \(c_i - b_i\) | same | Run‑vs‑run continuous or binary‑as‑float deltas |
| **Exact McNemar** | \(b\)=baseline‑only, \(c\)=candidate‑only; `binomtest` two‑sided | — | Paired binary disagreements |
| **Benjamini–Hochberg** | FDR control at q | q=0.05 | Multiplicity across metrics/slices |
| **Cohen's h** | \(2(\arcsin\sqrt{p_1}-\arcsin\sqrt{p_0})\) plus absolute/relative delta | — | Effect size beside p‑values |
| **pass@k** | Unbiased HumanEval‑style estimator via log‑gamma | — | Pass if any of k samples succeed given c/n |
| **Power / n** | Two‑sided rate comparison via arcsine (Cohen's h) sample size | α=0.05, power=0.8 | `evalctl power` before collecting data |
| **Flaky cases** | Cases with >1 distinct boolean outcome across repeats | — | Excluded from compare claims |
| **Between‑repeat variance** | Mean of per‑case sample variances | — | Stability diagnostics |

**Gotchas:**

- Small \(n\) ⇒ honest wide Wilson intervals — widen the dataset, don't shrink the CI.
- Always seed bootstraps when publishing (`seed=`).
- BH is applied across the comparison result list (`apply_multiplicity`); a single
  metric compare still returns `significant_bh`.
- `pass@k` is **not** "best of k then score once" without recording \(n,c,k\).

---

## 7. Reading the logs (harness + Ollama/llama.cpp)

Two streams: **harness** structlog JSON on the Python process stderr, and **Ollama**
container logs (`docker compose logs -f ollama`) with llama.cpp slot lines + Gin HTTP
access lines.

### 7.1 Harness (structlog) logs

`observability.setup_logging` emits ISO timestamps. `LOG_FORMAT=auto` selects readable
console logs on a TTY and JSON in CI/redirection; force `json` or `console` when needed.
Level comes from `LOG_LEVEL` (default `INFO`). Stable lifecycle events cover dataset
validation, provider resolution, generation, attempts/retries/cache, scoring batches,
slice aggregation, and report artifacts. Interactive CLI runs additionally show Rich
stage progress with counts and ETA.

Prompts and outputs are represented by `{chars}` by default. `LOG_PAYLOAD_HASHES=true`
adds SHA‑256 for correlation, but is not anonymization for low‑entropy outputs. Set
`LOG_PAYLOADS=true` only for scrubbed development data to add a bounded, redacted
preview. Full payloads remain in the database, not the log stream.

With `OTEL_ENABLED=true`, spans `run.generate → case → provider.call` and `run.score`
are opened and `trace_id` is persisted on generation rows. Without an endpoint they use
the in-memory exporter. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to an OTLP/HTTP `/v1/traces`
endpoint for batched production export. See [operations.md](operations.md#observability)
for the event vocabulary and privacy contract.

### 7.2 Decoding Ollama / llama.cpp logs

The harness calls Ollama `POST /api/chat` with `stream: true` for generation and
`POST /api/embed` for embeddings. Example annotated lines:

```text
srv  update_slots: all slots are idle
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 41
slot update_slots: id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)
slot print_timing: id  0 | task 0 | prompt eval time =   57.59 ms /   41 tokens ( 1.40 ms per token, 711.87 tokens per second)
slot print_timing: id  0 | task 0 |        eval time =  157.41 ms /    8 tokens (19.68 ms per token,  50.82 tokens per second)
slot print_timing: id  0 | task 0 |       total time =  215.00 ms /   49 tokens
slot print_timing: id  0 | task 0 |    graphs reused =        7
slot      release: id  0 | task 0 | stop processing: n_tokens = 48, truncated = 0
srv  update_slots: all slots are idle
[GIN] 2026/08/05 - 09:52:09 | 200 |  1.929801501s | 192.168.65.1 | POST "/api/chat"
[GIN] 2026/08/05 - 09:52:09 | 200 |   356.027125ms | 127.0.0.1     | POST "/api/embed"
```

Line‑by‑line:

- **`srv update_slots: all slots are idle`** — healthy idle, not stuck.
- **`slot get_availabl: … LRU` / `LCP similarity, sim_best, f_keep`** — scheduler picked
  a slot. Newer builds prefer **longest common prefix** overlap so the KV cache can be
  reused (`sim_best` = best overlap fraction; `f_keep` = fraction of cache kept).
- **`slot launch_slot_`** — work started (`is_child = 0` ⇒ top‑level request).
- **`new prompt, n_ctx_slot, n_keep, task.n_tokens`** — context budget, pinned prefix
  tokens, prompt length (aligns with `prompt_tokens`).
- **`cached n_tokens` / `memory_seq_rm [k, end)`** — KV reuse. `cached n_tokens = 0`
  means full prompt‑eval; `k>0` keeps the first k tokens.
- **`print_timing`** — **prompt eval** (prefill → drives TTFT) vs **eval** (decode →
  drives `total_ms − ttft_ms`); **graphs reused** is a warm‑path optimization.
  Gin wall time can exceed `print_timing` on cold model load.
- **`slot release: … truncated = 0|1`** — `truncated = 1` means context overflow
  (usually harness `truncated` / `finish_reason=length`).
- **`[GIN] … POST "/api/chat"` / `"/api/embed"`** — HTTP status + duration. Health
  polls show as `GET /api/tags` / `HEAD /`.

### 7.3 Mapping Ollama signals → harness

| Ollama signal | Harness effect |
|---------------|----------------|
| `/api/chat` 200 + streamed chunks | `raw_response.chunks`; usually `finish_reason=stop` |
| final chunk `done_reason == "length"` | `finish_reason=length` → outcome `truncated` |
| `truncated = 1` in slot release | context overflow → likely `truncated` |
| `prompt eval time` | drives `ttft_ms` |
| `eval time` | drives remaining latency |
| HTTP 429 | `RETRYABLE_RATE_LIMIT` → jittered retry |
| HTTP 5xx / timeouts / conn errors | `RETRYABLE_TRANSIENT` → retry; else `harness_error` |
| HTTP 401/403 | `NON_RETRYABLE_AUTH` → `harness_error` |
| other 4xx | `NON_RETRYABLE_REQUEST` → `harness_error` |
| circuit open | `CircuitOpenError` → retryable transient |

### 7.4 Troubleshooting log noise

- **CPU pegged during a run** — expected for CPU‑side decode; concurrency defaults to 2.
- **All slots idle forever** — healthy rest. If the harness thinks it sent work, check
  Gin non‑200s and `attempt_log`.
- **`truncated = 1`** — raise `--max-tokens`, shorten prompt, or use a larger context.
- **429 / 5xx storms** — retries fill `attempt_log`; exhaustion ⇒ `harness_error` and
  coverage loss → exit code 2. Fix backend, then `--resume`.

---

## 8. Operational runbook, troubleshooting & FAQ

### 8.1 Failure modes (fast reference)

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Tests skip `db_ready` | Postgres down | `docker compose up -d postgres` |
| `evalctl run` exits `1` | Invalid dataset / config | Read red `ERROR` lines |
| `evalctl run` exits `2` | Not publishable / coverage < floor | Inspect harness outcomes; `--resume` |
| "has no digest" | Model not pulled / missing digest in tags | `ollama pull <model>`; check `/api/tags` |
| Resume rejected | Dataset/template/model/config mismatch | Reuse exact original inputs |
| Duplicate key on resume | Two executors on one `run_id` | Single writer per run |
| OpenAI‑compatible ValueError | Missing base URL / revision env | Set `OPENAI_COMPATIBLE_*` |
| `meteor` NULL scores | NLTK wordnet missing | Install NLTK resources or omit metric |
| Embedding dim errors | 768‑d model vs `vector(1024)` column | Keep scoring in‑process or align dim |

### 8.2 Observability

- TTY-readable or structured JSON logs (structlog)
- Rich stage progress driven by transport-neutral callbacks
- Privacy-safe payload hashes/lengths; explicit opt-in redacted previews
- Optional OTel spans with in-memory or batched OTLP/HTTP export
- `trace_id` on generations for join across logs / DB / traces

### 8.3 FAQ

- **Q: I changed the normalizer. Do I regenerate?** No. `runs rescore` writes new
  `scores` rows with a new `metric_config_sha256` over the same generations.
- **Q: Why is my CI so wide?** Small \(n\). Wilson is telling the truth — add cases or
  repeats (`evalctl power` to plan).
- **Q: Are `temperature=0` runs bit‑reproducible?** Not guaranteed for local models.
  Record seed support; use the flaky‑case query.
- **Q: Where is the raw model output?** `generations.output` and
  `generations.raw_response`.
- **Q: How do I compare two prompt variants?** Two `evalctl run`s (different
  templates ⇒ different `config_sha256`), then
  `evalctl runs compare … --allow-compatible`.
- **Q: How do I add a provider?** Implement `Provider`, register one
  `evalharness.providers` entry point. Prefer wrapping via `create_provider` so
  limiter/breaker apply.
- **Q: How do I add a metric?** Implement `Metric`, register in
  `MetricRegistry.defaults()` or an `evalharness.metrics` entry point; rescore
  historical runs with zero inference.

### 8.4 Known gaps / deferred

Only items that are **truly not on current `main`** (or intentionally unfinished):

| Item | Reality | Source |
|------|---------|--------|
| Object storage for raw payloads | JSONB in Postgres by design | [`DEFERRED.md`](../DEFERRED.md) |
| Blocking release policy for judge/RAG signals | Phase 6 artifacts remain informational until calibrated and attached | Phase 6 contracts |
| `evald` HTTP API, OIDC, webhooks, `gates.yaml` service | Not shipped | Phase‑5 notes |
| Native Anthropic / Google adapters | Only `openai_compatible` + Ollama + mock | Provider package |
| Durable pgvector write path + HNSW for embeddings | Table exists; hot path is in‑memory | `EmbeddingService` / models |
| Registered `semantic_similarity` metric | Release script writes scores ad hoc | `scripts/run_release_e2e.py` |
| Standalone `evalctl report` | Reports emitted by `run` / library `write_report` | CLI surface |
| Comparison block inside the HTML report | `evalctl runs compare` emits JSON only | CLI surface |

---

### Appendix: file map (where to look)

| Concern | Path |
|---------|------|
| CLI | `src/evalharness/cli.py` |
| Config | `src/evalharness/config.py` |
| Core types | `src/evalharness/core/{models,enums,protocols}.py` |
| Executor | `src/evalharness/execution/executor.py` |
| Providers | `src/evalharness/providers/{ollama,openai_compatible,mock,runtime,registry,config}.py` |
| Scoring | `src/evalharness/scoring/{engine,registry,catalog,exact_match,normalizer,calibration,embeddings,ml,stats}.py` |
| Statistics | `src/evalharness/statistics/{core,comparison}.py` |
| Store | `src/evalharness/store/{models,repository,db}.py` |
| Reporting | `src/evalharness/reporting/report.py` + `templates/` |
| Migrations | `alembic/versions/000{1,2,3}_*.py` |
| Release E2E | `scripts/run_release_e2e.py` |
| PoC | `fixtures/poc/`, `scripts/run_poc.py` |
| Design docs | `docs/{architecture,dataplane,schema,operations,metrics,reports,providers,principles}.md` |
