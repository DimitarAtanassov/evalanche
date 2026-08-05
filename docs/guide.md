# evalanche — Onboarding & Operations Guide

> Audience: a new engineer (or on‑call) who needs **full** context on the `evalanche`
> harness — what it does, why it is built this way, how to run it, how to query the
> database, what every metric means, and how to read the logs (including the
> Ollama/llama.cpp container logs).
>
> This guide is written against the code on **`main`** (`git` commit `2c75262` —
> *"Ship evalanche core loop with Postgres-only persistence and offline PoC"*). The
> installable package is **evalanche**; the Python import path is **`evalharness`**;
> the CLI is **`evalctl`**.

---

## Scope & ground truth (read this first)

Everything below is grounded in the actual source on `main`. `main` ships a
deliberately small, correct **Phase‑1 core loop**:

- One CLI with **two** commands: `dataset-validate` and `run`.
- **Two** providers: `ollama` (live, digest‑pinned) and `mock` (deterministic, offline).
- **One** metric: `exact_match` v1.0.0, aggregated as a pass rate with a **Wilson** 95% CI.
- Latency percentiles + a mutually‑exclusive **outcome taxonomy**.
- Immutable generations + separate scores in **PostgreSQL 16 + pgvector**.
- A committed **offline PoC** (`fixtures/poc/`) that proves the whole plane without a GPU.

A larger "abstraction & catalog" effort (a full metric catalog, a `statistics`
package, an OpenAI‑compatible provider, a provider runtime with rate‑limiter /
circuit‑breaker, `runs rescore` / `runs compare` / `power` / `calibrate-threshold`
CLI verbs, and leadership/research/engineering report views) exists on an
**unmerged** branch (`feat/abstraction-and-catalog`, PR #1, tagged evidence
`v0.2.0`). **None of that is on `main`.** Where this guide describes those things
it labels them clearly as **Roadmap (not on `main`)** and gives you the plain‑language
math anyway so the concepts are useful. See [§9 Roadmap & known gaps](#9-roadmap--known-gaps-not-on-main).

Two environment facts matter and are called out again later:

1. The **local Postgres has already been migrated past `main`** (Alembic
   `0003_foundation_correctness`, from the unmerged branch). It therefore has extra
   columns/indexes `main`'s code does not create. For a clean `main` experience,
   bootstrap a **fresh** database. See [§5.1](#51-connecting-to-the-database).
2. `main`'s `OllamaProvider.resolve_version` expects `/api/show` to return a
   `digest`. The Ollama image currently running does **not** return one there
   (the digest lives in `/api/tags`), so a live `evalctl run --provider ollama` will
   fail at version resolution on this box. See [§7.4](#74-troubleshooting).

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
9. [Roadmap & known gaps (not on `main`)](#9-roadmap--known-gaps-not-on-main)

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
model outputs are expensive to produce (GPU time, latency, rate limits) and are the
*ground truth of what the model said*. Scores are cheap opinions *about* that text
and change every time you fix a normalizer bug, add a metric, or change a threshold.
Coupling them would mean re‑running the model every time you change your mind about
scoring — which is both wasteful and a subtle correctness trap (you'd be comparing a
new metric against *freshly sampled* outputs, not the frozen ones).

### 1.3 Core invariants (the non‑negotiables)

These come straight from [`docs/principles.md`](principles.md) and are enforced by
code and schema, not by convention:

| # | Invariant | Why it exists | Where it lives in code |
|---|-----------|---------------|------------------------|
| 1 | **Immutability.** Generation rows are written once; corrections are new rows, not mutations. | A frozen output is the only thing you can honestly re‑score. | `generations` has no update path except one narrow, documented `failed_score` flip (§1.4). |
| 2 | **Content addressing.** Datasets, templates, and run configs are SHA‑256 hashed. Different hash ⇒ different run. No silent cross‑hash comparison. | Prevents "we improved 3%" when the dataset silently changed. | `hashing.py` (`sha256_canonical`, `config_hash`), `datasets/loader.py` content hash. |
| 3 | **Harness errors ≠ model errors.** Report them separately; exclude harness failures from the model‑quality denominator. | A flaky network must never make a model look worse. | `enums.FailureOutcome`, coverage math in `reporting/report.py`. |
| 4 | **No point estimate without an interval.** Rates ship with a 95% CI (Wilson today). | A bare "92%" hides whether *n=25* or *n=25,000*. | `scoring/stats.wilson_interval`, surfaced in the report and CLI table. |
| 5 | **Generation ≠ scoring.** Module and schema boundaries enforce it; rescore must never call a provider. | See §1.2. | `scores` reference `generations` by id; scoring is pure over stored rows. |
| 6 | **Everything versioned.** Dataset, template, model digest, decode params, metric impl, normalizer ruleset. | Reproducibility and honest diffs. | `runs.config_sha256`, `scores.(metric_version, metric_config_sha256)`. |
| 7 | **Deterministic where possible, honest where not.** Record seed/temperature/top_p/top_k and whether the provider honors seeding. | Local models don't guarantee bit‑exact repeats; say so. | `runs.decode_params`, `Capabilities.supports_seed`. |
| 8 | **Resume over restart.** Checkpoint after each generation; resume skips completed `(case_id, repeat_idx)`. | A 4‑hour run that dies at 90% must not restart from 0. | `UNIQUE (run_id, case_id, repeat_idx)`, `Executor.plan`. |
| 9 | **One provider file to extend.** New backend = implement the `Provider` protocol + one entry‑point line. No runner/scorer/store edits. | Keeps the blast radius of a new backend tiny. | `core/protocols.Provider`, `providers/registry.py`, `pyproject.toml` entry points. |
| 10 | **Docs match code.** Behavior change ⇒ update `docs/` in the same PR. | This guide is part of that contract. | — |

### 1.4 The one honest wart

Principle 1 has a documented exception. After a generation is scored, if the metric
fails, the executor performs a *single narrow* update flipping the row's `outcome`
from `passed` to `failed_score`:

```379:397:src/evalharness/execution/executor.py
            if response and outcome not in (
                FailureOutcome.HARNESS_ERROR,
                FailureOutcome.HARNESS_TIMEOUT,
            ):
                ...
                for score in scores:
                    if outcome == FailureOutcome.PASSED and score.passed is False:
                        await session.execute(
                            update(GenerationRow)
                            .where(GenerationRow.id == gen_id)
                            .values(outcome=FailureOutcome.FAILED_SCORE.value)
                        )
```

This is tracked technical debt: the "right" version derives `passed`/`failed_score`
purely from the `scores` table at read time and never mutates the generation. Know it
exists; don't build new logic that depends on the mutation.

---

## 2. Architecture & data plane

### 2.1 Component map

```mermaid
flowchart LR
  CLI[evalctl] --> Loader[DatasetLoader + Validator]
  CLI --> Exec[Executor]
  Loader --> Store[(PostgreSQL 16 + pgvector)]
  Exec --> Reg[ProviderRegistry]
  Reg --> Ollama[OllamaProvider]
  Reg --> Mock[MockProvider]
  Exec --> Cache[(response_cache)]
  Exec --> Store
  Exec --> Scorer[ExactMatchMetric + Normalizer]
  Scorer --> Store
  CLI --> Reporter[Reporter]
  Reporter --> Store
  Reporter --> Artifacts[run_id.json / run_id.html]
```

| Component | Package | Responsibility |
|-----------|---------|----------------|
| Dataset loader + validator | `evalharness.datasets` | `manifest.yaml` + `cases.jsonl` → `Case`; fail‑fast validation, holdout guard, duplicate detection |
| Provider registry | `evalharness.providers` | Entry‑point discovery; one file to add a backend |
| Executor | `evalharness.execution` | Plan, concurrency, retries, cache, checkpoint, resume, outcome classification |
| Scoring | `evalharness.scoring` | Versioned normalizer + exact match + Wilson/percentiles |
| Store | `evalharness.store` | Async SQLAlchemy repository over PostgreSQL |
| Reporting | `evalharness.reporting` | Coverage, histograms, latency, pass‑rate CI; JSON + HTML |
| Config / hashing / observability | `evalharness.{config,hashing,observability}` | Settings, canonical hashing, structlog + OTel |

The seams you must not violate: the **`Provider` protocol**
(`core/protocols.py`), the **`Metric` protocol** (aggregation is metric‑specific —
never assume `mean()`), **content addressing**, and **harness‑vs‑model** separation.

### 2.2 Data plane: Case → Generate → Score → Report

```mermaid
sequenceDiagram
  participant Case
  participant Executor
  participant Cache as response_cache
  participant Provider
  participant Store as generations/scores
  participant Scorer

  Case->>Executor: render_prompt(template, inputs)
  Executor->>Cache: get(cache_key)
  alt cache hit
    Cache-->>Executor: GenerationResponse (cached=true)
  else miss
    Executor->>Provider: generate(model, request)
    Provider-->>Executor: text + finish_reason + timings + raw
    Executor->>Cache: put(cache_key, payload)
  end
  Executor->>Store: INSERT generation (immutable, with outcome)
  alt not a harness failure
    Executor->>Scorer: score(generation, case)
    Scorer-->>Executor: ScoreValue(passed, value, detail)
    Executor->>Store: INSERT score row (+ narrow failed_score flip)
  end
  Note over Executor,Store: after all cases → Reporter aggregates + persists metric_aggregates
```

**Render.** `render_prompt` is a deliberately dumb `{{key}}` substitution over
`case.inputs` — *not* Jinja — so the rendered prompt is a pure function of template
bytes + inputs (important for the cache key and `config_sha256`). Jinja is only used
to render the HTML report.

```87:91:src/evalharness/execution/executor.py
def render_prompt(template: str, case: Case) -> str:
    rendered = template
    for key, value in case.inputs.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered
```

**Cache key.** SHA‑256 of canonical JSON over
`{provider, model_version, prompt, decode, adapter}`. A hit sets
`generations.cached = true` and skips inference entirely. The cache lives in
Postgres (`response_cache`), so it survives process restarts and is shared across
runs with identical inputs.

```245:253:src/evalharness/execution/executor.py
        cache_key = sha256_canonical(
            {
                "provider": self.model_version.provider,
                "model_version": self.model_version.resolved_version,
                "prompt": rendered,
                "decode": config.decode_params,
                "adapter": f"{self.model_version.provider}-v1",
            }
        )
```

**Outcome taxonomy.** Every generation terminates in exactly one mutually‑exclusive
`FailureOutcome` (`core/enums.py`). This is the backbone of "harness ≠ model":

| Outcome | Meaning | In model‑quality denominator? |
|---------|---------|-------------------------------|
| `passed` | Generated and metric passed | ✅ yes |
| `failed_score` | Generated fine, metric failed | ✅ yes |
| `refused` | Output begins "I can't"/"I cannot" | ✅ yes (quality) |
| `truncated` | `finish_reason == length` | ✅ yes (quality) |
| `empty_output` | No/blank text | ✅ yes (quality) |
| `content_filtered` | `finish_reason == content_filter` | ✅ yes (quality) |
| `model_error` | Model‑side error | ✅ yes (quality) |
| `schema_invalid` | Structured‑output validation failed | ✅ yes (quality) |
| `harness_timeout` | Our timeout tripped | ❌ **no** — coverage loss |
| `harness_error` | Non‑retryable/exhausted infra error | ❌ **no** — coverage loss |
| `skipped` | Explicitly skipped | ❌ no |

Classification logic (note the order — timeouts/errors win, then empties, then
finish‑reason, then the refusal heuristic):

```94:115:src/evalharness/execution/executor.py
def classify_outcome(
    *,
    output: str | None,
    finish_reason: FinishReason | None,
    harness_error: bool,
    harness_timeout: bool,
) -> FailureOutcome:
    if harness_timeout:
        return FailureOutcome.HARNESS_TIMEOUT
    if harness_error:
        return FailureOutcome.HARNESS_ERROR
    if not output or not output.strip():
        return FailureOutcome.EMPTY_OUTPUT
    if finish_reason == FinishReason.LENGTH:
        return FailureOutcome.TRUNCATED
    if finish_reason == FinishReason.CONTENT_FILTER:
        return FailureOutcome.CONTENT_FILTERED
    ...
```

> Opinion: the `refused` heuristic (prefix "i can't"/"i cannot") is intentionally
> crude and English‑only. It is a *pragmatic placeholder*, not a safety classifier.
> Treat `refused` counts as a smell to investigate, not a metric to optimize.

**Coverage & the publishability gate.** `coverage = 1 − harness_failures / total`.
The default floor is **0.98**. Below the floor the report is marked non‑publishable
and `evalctl run` exits `2`. This is the mechanism that stops a leadership deck from
quoting a pass rate that was really "the GPU fell over for 20% of the run."

```114:118:src/evalharness/reporting/report.py
        total = len(generations)
        harness_failures = sum(
            1 for g in generations if g.outcome in {o.value for o in HARNESS_OUTCOMES}
        )
        coverage = 1.0 - (harness_failures / total if total else 0.0)
```

### 2.3 Retries, timeouts, and resume

**Retries.** Only `RETRYABLE_TRANSIENT` and `RETRYABLE_RATE_LIMIT` errors retry.
Backoff is **full‑jitter** exponential (`random.uniform(0, min(cap, base·2^attempt))`),
`base=0.5s`, `cap=30s`, up to `default_max_retries=5`. Every attempt is appended to
`attempt_log` — *retries are data*, not noise.

```118:120:src/evalharness/execution/executor.py
async def _retry_delay(attempt: int, base: float, cap: float) -> float:
    exp = min(cap, base * (2**attempt))
    return random.uniform(0, exp)
```

**Timeouts (layered).** `main` enforces two of the three planned layers:

1. **Per‑request** — `httpx` timeout (`default_request_timeout_s = 60s`), passed as
   `GenerationRequest.timeout_s`.
2. **Per‑case** — `asyncio.wait_for(..., timeout=case_timeout_s)`
   (`default_case_timeout_s = 120s`), covering all retries for a case.
3. **Per‑run wall budget** — *configured but not enforced on `main`* (it lands with
   the HTTP service). Don't rely on it yet.

**Resume.** `UNIQUE (run_id, case_id, repeat_idx)` makes each checkpoint idempotent.
`Executor.plan` loads the set of completed keys and only schedules the missing ones,
so `--resume <run_id>` re‑plans exactly the gap. Graceful `SIGTERM`/`SIGINT` handling
stops scheduling new cases (`GracefulShutdown`).

```184:192:src/evalharness/execution/executor.py
            completed = await repo.get_completed_keys(run_id)
            cases = await repo.get_cases_for_dataset(run.dataset_id)
            items: list[RunPlanItem] = []
            for case_db_id, case in cases:
                for repeat_idx in range(run.repeats):
                    if (case_db_id, repeat_idx) not in completed:
                        items.append(
                            RunPlanItem(case_db_id=case_db_id, case=case, repeat_idx=repeat_idx)
                        )
```

---

## 3. Local setup & running

### 3.1 Prerequisites

- Python **3.12+**
- [`uv`](https://github.com/astral-sh/uv)
- Docker (PostgreSQL via Compose; Ollama optional for live runs)

### 3.2 Services (`compose.yaml`)

| Service | Image | Port | Required for |
|---------|-------|------|--------------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | all runs, DB tests, PoC |
| `ollama` | `ollama/ollama:latest` | 11434 | live inference only |

Postgres credentials are baked into compose for local dev:
`user=evalharness`, `password=evalharness`, `db=evalharness`. Data persists in the
`postgres_data` / `ollama_data` named volumes.

### 3.3 One‑time bootstrap

```bash
# 1. Start Postgres (Ollama only if you want live runs)
docker compose up -d postgres
docker compose up -d ollama          # optional

# 2. Environment + dependencies
cp .env.example .env
uv sync --all-extras

# 3. Create the schema (SQLAlchemy metadata is the primary bootstrap on main)
uv run python -c "import asyncio; from evalharness.store.db import init_db; asyncio.run(init_db())"
```

`.env.example` (all values are read by `evalharness/config.py`):

```dotenv
DATABASE_URL=postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness
OLLAMA_BASE_URL=http://localhost:11434
HARNESS_VERSION=0.1.0
GIT_SHA=local
LOG_LEVEL=INFO
OTEL_ENABLED=false
```

Additional settings exist in `config.py` with defaults and can be overridden via env
(the field name upper‑cased): `DEFAULT_COVERAGE_FLOOR=0.98`, `DEFAULT_MAX_RETRIES=5`,
`DEFAULT_RETRY_BASE_S=0.5`, `DEFAULT_RETRY_CAP_S=30.0`, `DEFAULT_CONCURRENCY=2`,
`DEFAULT_CASE_TIMEOUT_S=120.0`, `DEFAULT_REQUEST_TIMEOUT_S=60.0`,
`OTEL_SERVICE_NAME=evalanche`.

### 3.4 Schema bootstrap: `init_db` vs Alembic

On `main`, tables are created by **SQLAlchemy metadata** (`init_db`), which also
ensures the `vector` extension exists. Alembic revisions exist for migration‑workflow
continuity:

| Revision | Change |
|----------|--------|
| `0001_initial` | `CREATE EXTENSION IF NOT EXISTS vector` (metadata create is the real bootstrap) |
| `0002_raw_response_jsonb` | add `generations.raw_response JSONB`; drop legacy `raw_uri` |

```39:44:src/evalharness/store/db.py
async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        with contextlib.suppress(Exception):
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
```

`init_db` is **create‑only** (`create_all` never alters existing tables). If your DB
already contains tables from a *newer* schema, `init_db` leaves them as‑is — which is
exactly the situation on this box (see [§5.1](#51-connecting-to-the-database)). For a
clean `main` environment, point `DATABASE_URL` at a fresh database/volume.

### 3.5 Smoke test: the offline PoC (no GPU, no Ollama)

```bash
uv run python scripts/run_poc.py          # regenerates fixtures/poc/*
uv run pytest tests/test_poc.py -q         # asserts against the committed golden report
```

The PoC runs the mock provider against `fixtures/sample_dataset` (5 QA cases) with a
fixed run id `00000000-0000-4000-8000-0000000000c1`, proving generate → score →
report end‑to‑end. The committed `fixtures/poc/report.json` shows `pass_rate=1.0`,
`coverage=1.0`, `publishable=true`, and a Wilson CI of `[0.5655, 1.0]` for 5/5 — a
great teaching example of *why* you never ship a bare "100%": with n=5 the lower
bound is only ~57%.

### 3.6 Dev quality gates

```bash
uv run ruff check .
uv run mypy src/evalharness      # strict mode
uv run pytest -q
```

---

## 4. CLI command reference & end‑to‑end workflows

`evalctl` (Typer app) exposes exactly two commands on `main`:

```text
Commands:
  dataset-validate   Validate a dataset manifest and cases.
  run                Run evaluation against a dataset.
```

> Roadmap verbs (`score`, `runs rescore`, `runs compare`, `power`,
> `calibrate-threshold`, `report`) do **not** exist on `main`; see
> [§9](#9-roadmap--known-gaps-not-on-main). The re‑score‑only workflow is currently a
> *design property* (generations are immutable and independent of scores) rather than
> a shipped command.

### 4.1 `evalctl dataset-validate`

```text
Usage: evalctl dataset-validate [OPTIONS] DATASET_DIR

Arguments:
  DATASET_DIR   Path to dataset directory   [required]

Options:
  --i-am-doing-a-final-eval   Allow holdout split evaluation
```

What it checks (`datasets/validator.py`): duplicate case ids; per‑`task_type`
required fields (e.g. `qa_short`/`summarization` require `reference_answer`,
`classification` requires `expected_label`, `retrieval` requires `qrels`, `rag`
requires both); every manifest‑declared slice key is present on every case;
`manifest.content_sha256` matches the recomputed hash; and duplicate **normalized
prompts** (a warning — likely train/test leakage or copy‑paste). The **holdout
guard** refuses to evaluate a `split: holdout` dataset unless you explicitly pass
`--i-am-doing-a-final-eval`, so you can't overfit to your final test set by accident.

```bash
# Dev dataset — should pass
uv run evalctl dataset-validate fixtures/sample_dataset
# -> Valid synthetic-qa@1.0.0 (5 cases, sha256=c3dbdeaf2ef6...)

# A holdout set — blocked unless you mean it
uv run evalctl dataset-validate path/to/holdout --i-am-doing-a-final-eval
```

Exit codes: `0` valid, `1` invalid (errors printed in red).

### 4.2 `evalctl run`

```text
Usage: evalctl run [OPTIONS]

Options:
  --dataset PATH                 Dataset directory              [required]
  --template PATH                Prompt template file           [required]
  --model TEXT                   Model name                     [required]
  --provider TEXT                Provider name                  [default: ollama]
  --output PATH                  Report output dir              [default: reports]
  --repeats INT                  Number of repeats per case     [default: 1]
  --concurrency INT              Max concurrent requests        [default: 2]
  --temperature FLOAT            [default: 0.0]
  --max-tokens INT
  --seed INT
  --resume TEXT                  Resume existing run ID
  --i-am-doing-a-final-eval
  --coverage-floor FLOAT         [default: 0.98]
  --tenant TEXT                  [default: default]
```

Lifecycle (`cli._run_async`): set up logging/OTel → `init_db` → load + validate
dataset (abort on invalid) → hash template → resolve model version via the provider →
upsert `datasets`/`cases`/`prompt_templates`/`model_versions` → create or resume the
`runs` row → `execute_run` (concurrent, resumable) → `write_report` → print a summary
table → exit `2` if not publishable.

On success you get a Rich summary table:

| Field | Source |
|-------|--------|
| Run ID | the `runs.id` UUID |
| Config SHA256 | `runs.config_sha256` (dataset+template+model+decode+harness) |
| Model digest | `model_versions.resolved_version` |
| Coverage | `1 − harness_failures/total` |
| Pass rate | pass rate + Wilson 95% CI `[low, high]` |
| Publishable | `coverage >= coverage_floor` |

### 4.3 Workflow A — offline mock run (works anywhere Postgres is up)

```bash
docker compose up -d postgres
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa \
  --provider mock \
  --seed 42
# reports/<run_id>.json and reports/<run_id>.html are written
```

This is the fastest way to exercise the full plane. The mock provider answers the
synthetic "What is N plus one?" cases deterministically, so you get `pass_rate=1.0`,
`coverage=1.0`.

### 4.4 Workflow B — live Ollama run (the "v0.2.0‑style" run)

> ⚠️ On this machine, `main`'s `resolve_version` fails against the running Ollama
> because `/api/show` returns no `digest` (see [§7.4](#74-troubleshooting)). The
> commands below are the intended workflow and are exactly what the docs prescribe;
> if you hit the digest error, that is the known incompatibility, not a mistake in
> your invocation.

```bash
docker compose up -d ollama
ollama pull llama3.2:1b          # pull once; version is pinned by digest

# Single-model run, low concurrency (a 1B model on CPU is happy at 2)
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama \
  --concurrency 2 \
  --repeats 5 \
  --temperature 0.0 \
  --max-tokens 32
```

`--repeats 5` samples each case five times — the basis for measuring flakiness /
self‑consistency (see the flaky‑case SQL in [§5.4](#54-query-library)). Because
`temperature=0.0`, repeats should be near‑identical for a deterministic backend;
divergence is itself a signal.

**Resume** a run that was interrupted (same dataset/template/model, add `--resume`):

```bash
uv run evalctl run --resume <run_id> \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama
```

### 4.5 Re‑scoring & comparing runs (today vs roadmap)

- **Re‑score** (change a metric/normalizer and re‑evaluate *stored* outputs without
  inference): on `main` there is no `runs rescore` command. The building blocks are
  all present — scoring is a pure function over `generations` — but you would drive it
  from a small script using `RunRepository.get_generations_for_run` +
  `ExactMatchMetric`. The immutability + `scores` uniqueness on
  `(generation_id, metric_name, metric_version, metric_config_sha256)` guarantee a new
  normalizer config produces *new* score rows without clobbering the old ones.
- **Compare two runs**: no `runs compare` command on `main` either — use the
  cross‑run SQL in [§5.4](#54-query-library), which joins two runs on the case's
  stable `external_id`.

---

## 5. Database deep‑dive

PostgreSQL 16 + `pgvector`. This is the **only** persistence service — raw provider
payloads live in `generations.raw_response` (JSONB), *not* object storage. That is a
deliberate, documented deferral (`DEFERRED.md`): MinIO/S3 adds a service, credentials,
backups, and failure modes the current payload volume doesn't justify. Revisit when
payloads bloat backups/vacuum, exceed practical JSONB sizes, or need a separate
retention policy.

### 5.1 Connecting to the database

The app uses the async URL from `config.py`:

```text
postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness
```

For interactive `psql`, use the sync form. Two easy options:

```bash
# A) exec into the container (no local psql needed)
docker exec -it evalv1-postgres-1 psql -U evalharness -d evalharness

# B) local psql against the mapped port
PGPASSWORD=evalharness psql -h localhost -p 5432 -U evalharness -d evalharness
```

> **Important environment caveat.** The database on this machine has already been
> migrated to Alembic `0003_foundation_correctness` (from the unmerged
> `feat/abstraction-and-catalog` branch). Its live schema therefore has objects that
> `main`'s `store/models.py` does **not** define, e.g. `metric_aggregates.metric_config_sha256`
> (`NOT NULL`), a unique `uq_metric_aggregates_identity`, `ix_runs_status_started_at`,
> `ix_scores_generation_id`, and a `COALESCE(quantization,'')`‑based uniqueness on
> `model_versions`. Consequences for a `main` engineer:
>
> - A `main` `evalctl run` against *this* DB would fail at report time, because
>   `metric_aggregates.metric_config_sha256` is `NOT NULL` and `main`'s reporter never
>   populates it. To run `main` cleanly, use a **fresh** database.
> - Every table/column named below is defined by `main`'s `store/models.py`; it is a
>   **subset** of the live DB, so all read queries in [§5.4](#54-query-library) run
>   fine against either schema.

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
  generations ||--o{ judgments : judged
  cases ||--o{ annotations : labeled
```

Three groups of tables:

- **Definitions** (append‑once, content‑addressed): `datasets`, `cases`,
  `prompt_templates`, `model_versions`.
- **Run + immutable outputs**: `runs`, `generations`, `scores`, `metric_aggregates`.
- **Forward‑compatible / cache**: `judgments`, `annotations`, `embeddings`,
  `response_cache`.

### 5.3 Table‑by‑table reference

All types/constraints below are from `src/evalharness/store/models.py`.

#### `datasets` — a named, versioned, content‑hashed dataset

| Column | Type | Null | Notes / why |
|--------|------|------|-------------|
| `id` | bigint PK | no | surrogate key |
| `name` | text | no | e.g. `synthetic-qa` |
| `version` | text | no | e.g. `1.0.0` |
| `content_sha256` | text | no | hash of the JSONL body — the identity that matters |
| `split` | text | no | `dev` / `holdout` / … (drives the holdout guard) |
| `manifest` | jsonb | no | denormalized manifest (license, pii_scrubbed, slices, …) |
| `created_at` | timestamptz | no | `now()` |

Uniqueness: `(name, version)`. Referenced by `cases`, `runs`.
**Why hash the content and not trust the version string?** Because humans forget to
bump versions. `validate_dataset` refuses a run if `manifest.content_sha256` disagrees
with the recomputed hash.

#### `cases` — one evaluation case

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | surrogate; FK target for generations |
| `dataset_id` | bigint FK→datasets | no | owner |
| `external_id` | text | no | *stable* id from the JSONL (`case-00000`) — use this for cross‑run joins |
| `task_type` | text | no | one of `TaskType` (qa_short, classification, …) |
| `inputs` | jsonb | no | template variables (e.g. `{"question": "…"}`) |
| `reference` | jsonb | yes | packed answer fields (see below) |
| `qrels` | jsonb | yes | relevance judgments for retrieval/RAG |
| `slices` | jsonb | no, default `{}` | analysis dimensions (`{"difficulty":"hard","lang":"en"}`) |
| `weight` | float | no, default `1.0` | reserved for weighted aggregation |

Uniqueness: `(dataset_id, external_id)`. Index: **GIN** on `slices`
(`jsonb_path_ops`) — makes slice filters (`slices @> '{"difficulty":"hard"}'`) fast.
The repository packs answer‑shaped fields into the `reference` JSONB:
`reference_answer`, `references`, `expected_label`, `expected_json`, `must_contain`,
`must_not_contain`. **Why pack them?** Task types need different answer shapes; a
single JSONB column avoids a wide, mostly‑null table while keeping everything on the
case row.

#### `prompt_templates` — template body + hash

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `name` | text | no | e.g. `synthetic-qa-template` |
| `version` | text | no | |
| `body` | text | no | raw template (the `{{var}}` string) |
| `content_sha256` | text | no | feeds `config_sha256` |

Uniqueness: `(name, version)`.

#### `model_versions` — the *resolved* model identity

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `provider` | text | no | `ollama` / `mock` |
| `model` | text | no | requested name (`llama3.2:1b`) |
| `resolved_version` | text | no | **digest** — the pinned identity, not the tag |
| `quantization` | text | yes | e.g. `Q8_0` |
| `params_b` | float | yes | parameter count in billions |
| `context_window` | int | yes | |
| `capabilities` | jsonb | no | seed/logprobs/tools/json/streaming/system flags + max ctx |

Uniqueness (per `main`): `(provider, model, resolved_version, quantization)`.
**Why store a digest and not the tag?** `llama3.2:1b` is a moving target; the digest
is immutable. Pinning the digest is what makes "same run, six months later" mean
something. (The live DB's `0003` variant makes the uniqueness `COALESCE(quantization,'')`
to treat NULL quant as a distinct value — a correctness fix that is not yet on `main`.)

#### `runs` — the job

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | uuid PK | no | `default uuid4` |
| `dataset_id` / `prompt_template_id` / `model_version_id` | FK | no | the three content‑addressed inputs |
| `decode_params` | jsonb | no | temperature/max_tokens/seed/top_p/top_k/stop |
| `config_sha256` | text | no | hash of all inputs + harness version → the run's identity |
| `harness_version` | text | no | from settings |
| `git_sha` | text | no | from settings |
| `repeats` | int | no, default 1 | samples per case |
| `status` | text | no | `queued`→`running`→`completed`\|`failed`\|`cancelled` |
| `tenant_id` | text | no | multi‑tenant scoping |
| `started_at` / `finished_at` | timestamptz | yes | |
| `baseline_run_id` | uuid FK→runs | yes | self‑reference for A/B comparisons |

#### `generations` — immutable model outputs (the crown jewels)

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `run_id` | uuid FK→runs | no | |
| `case_id` | bigint FK→cases | no | |
| `repeat_idx` | int | no, default 0 | which sample |
| `output` | text | yes | the model's text |
| `tool_calls` | jsonb | yes | reserved (empty on `main`) |
| `finish_reason` | text | yes | `stop`/`length`/`content_filter`/… |
| `outcome` | text | no | the `FailureOutcome` |
| `prompt_tokens` / `completion_tokens` | int | yes | provider usage |
| `cost_usd` | numeric(12,6) | yes | `0.0` for local models |
| `ttft_ms` | float | yes | time‑to‑first‑token |
| `total_ms` | float | yes | end‑to‑end wall time |
| `queue_wait_ms` | float | yes | reserved (NULL on `main`) |
| `attempts` | int | no, default 1 | length of attempt_log |
| `attempt_log` | jsonb | yes | per‑attempt `{attempt, error_class, duration_ms, at, message?}` |
| `cached` | bool | no, default false | cache hit? |
| `raw_response` | jsonb | yes | full provider payload (Ollama: `{"chunks":[…]}`) |
| `trace_id` | text | yes | OTel linkage |
| `created_at` | timestamptz | no | `now()` |

**Uniqueness: `(run_id, case_id, repeat_idx)`** — this one constraint *is* the resume
mechanism and the idempotency guard. A second executor that re‑inserts a completed key
hits a duplicate‑key error rather than silently double‑counting.

#### `scores` — metric results *about* a generation

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `generation_id` | bigint FK→generations | no | what was scored |
| `metric_name` | text | no | `exact_match` |
| `metric_version` | text | no | `1.0.0` |
| `metric_config_sha256` | text | no | hash of the **normalizer config** |
| `value` | float | yes | `1.0`/`0.0` for exact match; NULL if unscoreable |
| `passed` | bool | yes | pass/fail; NULL if unscoreable |
| `detail` | jsonb | yes | e.g. `{normalized_prediction, normalized_reference}` |
| `scored_at` | timestamptz | no | `now()` |

**Uniqueness: `(generation_id, metric_name, metric_version, metric_config_sha256)`.**
This is *the* re‑score‑safety constraint: re‑scoring with the *same* metric+config is
idempotent, and re‑scoring with a *changed* normalizer (different
`metric_config_sha256`) writes a **new** row alongside the old one. History is
additive.

#### `metric_aggregates` — per‑run rollups

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `id` | bigint PK | no | |
| `run_id` | uuid FK→runs | no | |
| `metric_name` / `metric_version` | text | | |
| `slice_key` | text | no, default `__overall__` | overall today; per‑slice later |
| `n` | int | no | denominator |
| `value` | float | no | the rate |
| `ci_low` / `ci_high` | float | yes | Wilson bounds |
| `stddev` | float | yes | NULL for binomial pass rate |
| `method` | text | yes | `wilson` |

> On the live DB this table also has `metric_config_sha256 NOT NULL` and a unique
> `(run_id, metric_name, metric_version, slice_key, metric_config_sha256)` from
> `0003`; `main`'s model has neither. This is the column that makes `main`'s reporter
> incompatible with the migrated DB.

#### `response_cache` — the inference cache

| Column | Type | Null | Notes |
|--------|------|------|-------|
| `cache_key` | text PK | no | SHA‑256 over `{provider, model_version, prompt, decode, adapter}` |
| `response` | jsonb | no | the serialized `GenerationResponse` payload |
| `created_at` | timestamptz | no | `now()` |

Writes are insert‑if‑absent (`put_cache` no‑ops on an existing key), so the first
result for a key wins and is never overwritten.

#### Forward‑compatible tables (schema present, unused by `main`'s paths)

- **`judgments`** — LLM‑as‑judge / pairwise preference results (rubric, score,
  preference, swap_position for position‑bias control, reasoning, evidence, cost).
- **`annotations`** — human labels (annotator_id, label JSONB, adjudicated flag).
- **`embeddings`** — `content_sha256` + `embedding_model_version_id` + `vec vector(1024)`,
  unique on `(content_sha256, embedding_model_version_id)`. **Caveat:** the column is
  `vector(1024)`, but a common local embedder (`nomic-embed-text`) returns **768** dims;
  wiring embedding metrics will require a per‑model dimension (or a wider/again‑versioned
  column). Unused on `main`.

### 5.4 Query library

Copy‑paste queries a researcher/engineer actually runs. Replace `:run` with a UUID.
Grab the most recent run id first:

```sql
SELECT id, status, started_at, finished_at
FROM runs
ORDER BY started_at DESC
LIMIT 5;
```

**1. Overall pass rate + coverage (the numbers on the CLI summary).**

```sql
SELECT
  count(*)                                                        AS total,
  count(*) FILTER (WHERE outcome IN ('harness_error','harness_timeout')) AS harness_failures,
  round(1 - count(*) FILTER (WHERE outcome IN ('harness_error','harness_timeout'))::numeric
          / nullif(count(*),0), 4)                                AS coverage,
  count(*) FILTER (WHERE outcome = 'passed')                       AS passed,
  count(*) FILTER (WHERE outcome NOT IN ('harness_error','harness_timeout')) AS eligible,
  round(count(*) FILTER (WHERE outcome = 'passed')::numeric
          / nullif(count(*) FILTER (WHERE outcome NOT IN ('harness_error','harness_timeout')),0), 4) AS pass_rate
FROM generations
WHERE run_id = '00000000-0000-4000-8000-0000000000c1';
```

*Reading it:* `coverage` must be ≥ your floor (0.98) for the run to be publishable.
`pass_rate` uses the **eligible** denominator (harness failures excluded) — that is
principle #3 in SQL form. Postgres gives you the point estimate; the Wilson interval
around it is computed app‑side (see [§6.3](#63-statistics-what-shipped)).

**2. Pass rate by slice (e.g. difficulty).** Joins to `cases` for the slice value.

```sql
SELECT
  c.slices ->> 'difficulty'                                       AS difficulty,
  count(*) FILTER (WHERE g.outcome NOT IN ('harness_error','harness_timeout')) AS eligible,
  count(*) FILTER (WHERE g.outcome = 'passed')                    AS passed,
  round(count(*) FILTER (WHERE g.outcome = 'passed')::numeric
          / nullif(count(*) FILTER (WHERE g.outcome NOT IN ('harness_error','harness_timeout')),0), 4) AS pass_rate
FROM generations g
JOIN cases c ON c.id = g.case_id
WHERE g.run_id = :run
GROUP BY 1
ORDER BY 1;
```

*Reading it:* a big gap between `easy` and `hard` is the interesting signal. Watch the
`eligible` column — a slice with tiny n has a wide (untrustworthy) interval even if the
point estimate looks great.

**3. Coverage vs planned (did we actually evaluate every case?).**

```sql
SELECT
  (SELECT count(*) FROM cases c JOIN runs r ON r.dataset_id = c.dataset_id
     WHERE r.id = :run) * (SELECT repeats FROM runs WHERE id = :run) AS planned,
  count(*)                                                            AS produced
FROM generations WHERE run_id = :run;
```

*Reading it:* `produced < planned` means the run is incomplete (interrupted, or still
resumable). `produced = planned` with low coverage means it finished but with
harness failures.

**4. Harness failures vs model failures (the taxonomy histogram).**

```sql
SELECT outcome, count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM generations
WHERE run_id = :run
GROUP BY outcome
ORDER BY n DESC;
```

*Reading it:* `harness_*` rows are *your* problem (infra); everything else is the
model's. If `harness_error`/`harness_timeout` dominate, fix infra and re‑run before
drawing any quality conclusion.

**5. Latency percentiles (server‑side, matches the report).**

```sql
SELECT
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p50,
  round(percentile_cont(0.90) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p90,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p95,
  round(percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms)::numeric, 2) AS p99,
  round(max(total_ms)::numeric, 2)  AS max,
  round(avg(total_ms)::numeric, 2)  AS mean
FROM generations
WHERE run_id = :run AND total_ms IS NOT NULL;
```

*Reading it:* the report uses the same percentiles (`scoring/stats.percentile`, linear
interpolation) and labels `mean` separately — because for skewed latency the mean is a
vanity number; p95/p99 are what your users feel. `percentile_cont` here interpolates
just like the harness.

**6. TTFT vs total (where did the time go?).**

```sql
SELECT
  round(avg(ttft_ms)::numeric,1)                 AS avg_ttft,
  round(avg(total_ms - ttft_ms)::numeric,1)      AS avg_gen_after_first_token,
  round(avg(total_ms)::numeric,1)                AS avg_total
FROM generations
WHERE run_id = :run AND ttft_ms IS NOT NULL AND total_ms IS NOT NULL;
```

*Reading it:* a high `avg_ttft` relative to total means the model spent its time on
**prompt eval** / queueing / cold‑start; a high after‑first‑token time means
**decoding** dominated. This maps directly onto the llama.cpp `print_timing` split in
[§7.2](#72-decoding-the-ollamallamacpp-logs).

**7. Cache hit rate.**

```sql
SELECT
  count(*) FILTER (WHERE cached) AS hits,
  count(*)                       AS total,
  round(100.0 * count(*) FILTER (WHERE cached) / nullif(count(*),0), 1) AS hit_pct
FROM generations
WHERE run_id = :run;
```

*Reading it:* on a first run this is ~0%. On a `--resume` or a re‑run with identical
inputs it climbs — proving inference was skipped. If you *expected* hits and got none,
your cache key changed (different decode params, template bytes, or resolved digest).

**8. Per‑metric aggregates (from the `scores` table directly).**

```sql
SELECT
  s.metric_name, s.metric_version, s.metric_config_sha256,
  count(*)                              AS n,
  count(*) FILTER (WHERE s.passed)      AS passed,
  round(avg(s.value)::numeric, 4)       AS mean_value
FROM scores s
JOIN generations g ON g.id = s.generation_id
WHERE g.run_id = :run
GROUP BY 1,2,3
ORDER BY 1;
```

*Reading it:* the `metric_config_sha256` column lets you see *multiple normalizer
configs* side by side if you've re‑scored. Same metric_name, different config hash =
different scoring opinion over the same outputs.

**9. The persisted rollup (what the reporter wrote).**

```sql
SELECT metric_name, slice_key, n, value, ci_low, ci_high, method
FROM metric_aggregates
WHERE run_id = :run;
```

**10. Find flaky cases across repeats (needs `--repeats > 1`).**

```sql
SELECT
  c.external_id,
  count(*)                                         AS n_samples,
  count(*) FILTER (WHERE g.outcome = 'passed')     AS n_passed,
  count(DISTINCT g.output)                         AS distinct_outputs
FROM generations g
JOIN cases c ON c.id = g.case_id
WHERE g.run_id = :run
GROUP BY c.external_id
HAVING count(*) FILTER (WHERE g.outcome = 'passed') NOT IN (0, count(*))
    OR count(DISTINCT g.output) > 1
ORDER BY distinct_outputs DESC, c.external_id;
```

*Reading it:* a case that passes on some repeats and fails on others (`n_passed`
strictly between 0 and `n_samples`) is **flaky** — the model is not self‑consistent
there, or your normalizer is borderline. `distinct_outputs > 1` at `temperature=0.0`
means the backend isn't truly deterministic (expected for many local models — that's
why principle #7 records whether seeding is honored).

**11. Compare two runs on the same cases (A/B).**

```sql
WITH a AS (
  SELECT g.case_id, bool_or(g.outcome = 'passed') AS passed
  FROM generations g WHERE g.run_id = :run_a GROUP BY g.case_id
),
b AS (
  SELECT g.case_id, bool_or(g.outcome = 'passed') AS passed
  FROM generations g WHERE g.run_id = :run_b GROUP BY g.case_id
)
SELECT c.external_id,
       a.passed AS run_a_passed,
       b.passed AS run_b_passed
FROM a JOIN b USING (case_id)
JOIN cases c ON c.id = a.case_id
WHERE a.passed IS DISTINCT FROM b.passed
ORDER BY c.external_id;
```

*Reading it:* the rows are exactly the cases where the two runs **disagree** — the
"wins" and "regressions." Count `run_b_passed AND NOT run_a_passed` (wins) vs
`run_a_passed AND NOT run_b_passed` (regressions); that 2×2 is the input a McNemar
test would consume (see [§6.4](#64-statistics-roadmap-not-on-main)). Only join runs
that share a dataset — the stable join key is `cases.external_id`.

**12. Inspect a single generation's raw payload & retry history.**

```sql
SELECT
  g.id, c.external_id, g.outcome, g.finish_reason,
  g.ttft_ms, g.total_ms, g.attempts,
  jsonb_pretty(g.attempt_log)                          AS attempt_log,
  jsonb_pretty(g.raw_response #> '{chunks,0}')          AS first_raw_chunk,
  left(g.output, 300)                                   AS output_preview
FROM generations g JOIN cases c ON c.id = g.case_id
WHERE g.id = :generation_id;
```

*Reading it:* `attempt_log` is your retry forensics — each entry has `error_class`,
`duration_ms`, and `at`. For Ollama, `raw_response` is `{"chunks":[…]}` (the streamed
NDJSON); the final chunk carries `done`, `done_reason`, `prompt_eval_count`,
`eval_count`. This is where you go when a number looks wrong.

**13. Embedding lookups (forward‑compatible table).**

```sql
SELECT e.id, e.content_sha256, mv.model, vector_dims(e.vec) AS dims
FROM embeddings e
JOIN model_versions mv ON mv.id = e.embedding_model_version_id
LIMIT 20;
-- nearest neighbors (pgvector) once embeddings are populated:
-- SELECT content_sha256, vec <=> :query_vec AS cosine_distance
-- FROM embeddings ORDER BY vec <=> :query_vec LIMIT 10;
```

*Reading it:* empty on `main` (no path writes embeddings yet). `<=>` is pgvector's
cosine‑distance operator; smaller = more similar.

---

## 6. Metric catalog & statistics

### 6.1 The metric contract

Every metric implements the `Metric` protocol (`core/protocols.py`): a `name`,
`version`, the `task_types` it applies to, the `requires` set (data prerequisites like
`REFERENCE`/`QRELS`/`EMBEDDINGS`/`JUDGE`/`LOGPROBS`), a `score()` that returns
`ScoreValue`s for one generation, and an `aggregate()` that rolls many `ScoreValue`s
into an `AggregateValue`. **Aggregation is metric‑specific — never assume `mean()`.**

```35:43:src/evalharness/core/protocols.py
class Metric(Protocol):
    name: str
    version: str
    task_types: frozenset[TaskType]
    requires: frozenset[Requirement]

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]: ...

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue: ...
```

### 6.2 The one metric that ships: `exact_match` v1.0.0

| Property | Value |
|----------|-------|
| Measures | Does the normalized model output equal the normalized reference? |
| Applies to (`task_types`) | `generation`, `qa_short`, `summarization`, `rag` |
| Requires | `REFERENCE` (a `reference_answer`, or first of `references`) |
| Per‑case output | `value ∈ {0.0, 1.0}`, `passed ∈ {true,false}`; `value/passed = NULL` if output or reference is missing (unscoreable, excluded from n) |
| Aggregation | pass rate `= successes / n`; interval = **Wilson** 95% CI; `method="wilson"` |
| Detail stored | `{normalized_prediction, normalized_reference}` (or `{reason: missing_reference_or_output}`) |

```42:54:src/evalharness/scoring/exact_match.py
        pred = self.normalizer.normalize(gen.output)
        gold = self.normalizer.normalize(reference)
        passed = pred == gold
        return [
            ScoreValue(
                metric_name=self.name,
                metric_version=self.version,
                metric_config_sha256=self.normalizer.config_id,
                value=1.0 if passed else 0.0,
                passed=passed,
                detail={"normalized_prediction": pred, "normalized_reference": gold},
            )
        ]
```

**When to use it.** Short, closed‑form answers: arithmetic, single entities, exact
labels, canonical strings. It is precise and cheap and has zero dependence on a judge
model.

**When *not* to use it (pitfalls).** Anything with legitimate surface variation —
long‑form generation, summaries, paraphrase‑valid QA. Exact match is brutally
sensitive to formatting, so **the normalizer is doing most of the real work**; a
scoring "bug" is almost always a normalizer mismatch. Inspect `scores.detail` to see
exactly what was compared. `value=NULL` means "couldn't score" (missing reference or
empty output) and is *excluded* from the denominator — not counted as a fail.

#### The normalizer (the metric's real behavior)

`scoring/normalizer.py` applies a versioned, hashed ruleset. The default
`NormalizerConfig`:

| Step | Default | Effect |
|------|---------|--------|
| `unicode_nfkc` | true | Unicode NFKC normalization (canonicalize width/compatibility forms) |
| `lowercase` | true | case‑insensitive compare |
| `strip_punctuation` | true | replace non `[\w\s.-]` with a space |
| `numeric_tol` | `None` | if set, round numbers to a tolerance (off by default) |
| `strip_articles` | true | drop `a`/`an`/`the` |
| `collapse_whitespace` | true | squeeze runs of whitespace, trim |
| `version` | `1.0.0` | part of the config identity |

The config's SHA‑256 (`config_id`) is stored on every score as
`metric_config_sha256`. **This is why re‑scoring is safe and honest:** change any knob
and you get a new config hash ⇒ new score rows ⇒ old numbers preserved. Two "exact
match" numbers with different config hashes are *not* comparable, and the schema stops
you from pretending otherwise.

### 6.3 Statistics: what shipped

Two functions, in `scoring/stats.py`. Both are deliberately dependency‑free.

**Wilson score interval** — the 95% CI on the pass rate. Given `s` successes out of
`n`, with `z = 1.96`:

\[
\text{center} = \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad
\text{margin} = \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}
\]

where \(\hat p = s/n\); the interval is `[center − margin, center + margin]` clamped to
`[0,1]`.

*Plain language.* It's the honest error bar on a proportion. **Why Wilson and not the
textbook `p ± 1.96·√(p(1−p)/n)` (normal approximation)?** The normal approximation is
embarrassingly wrong exactly where eval lives: small n and rates near 0 or 1. At 5/5 it
would give `[100%, 100%]` — a lie. Wilson gives `[56.6%, 100%]` (the committed PoC
number), which correctly says "we saw 5/5 but we don't actually know it's not ~57%."
It never produces bounds outside `[0,1]` and behaves well at the extremes. This is
principle #4 made concrete.

```8:16:src/evalharness/scoring/stats.py
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)
```

**Percentiles** — linear interpolation between closest ranks (the same convention as
NumPy's default / `percentile_cont`). Used for latency p50/p90/p95/p99. The reporter
reports `max` and `mean` too, but labels `mean` separately because for right‑skewed
latency the mean is misleading.

**Coverage** is the third "statistic" and the most important governance number:
`1 − harness_failures/total`, gating publishability at 0.98.

### 6.4 Statistics & the full metric catalog: roadmap (NOT on `main`)

The original design calls for a Part‑6.1–6.6 catalog and a `statistics` package with
BCa/paired bootstrap, McNemar, Benjamini–Hochberg, Cohen's h, pass@k, and power. **None
of these exist on `main`** (there is no `scoring/catalog.py`, `registry.py`, `engine.py`,
`calibration.py`, `embeddings.py`, `ml.py`, and no `statistics/` package). They live on
the unmerged `feat/abstraction-and-catalog` branch. Here is the plain‑language +
formula reference so the concepts are useful when that work merges — treat this whole
subsection as *forward‑looking*.

- **BCa bootstrap** (bias‑corrected & accelerated). For a CI on a *continuous* metric
  mean (e.g. average ROUGE), resample the per‑case scores with replacement B times,
  and correct the naive percentile interval for bias (`z0`) and skew (acceleration
  `a`). *Use when* the sampling distribution isn't binomial and may be skewed. Wilson
  is for proportions; BCa is for means/continuous scores.

- **Paired bootstrap** for a difference between two models on the **same** cases:
  resample cases (not models) with replacement, recompute the per‑resample difference,
  take the 2.5/97.5 percentiles. *Use when* comparing runs on a shared dataset —
  pairing cancels case difficulty and tightens the interval dramatically vs treating
  the runs as independent.

- **McNemar test** for paired binary outcomes (pass/fail on the same cases). Build the
  2×2 of discordant pairs `b` (A pass, B fail) and `c` (A fail, B pass); the statistic
  is \(\chi^2 = (|b-c|-1)^2/(b+c)\) (with continuity correction), 1 dof. *Use when*
  asking "is model B significantly better than A on these cases?" — this is the correct
  test for the disagreement rows from query #11.

- **Benjamini–Hochberg (FDR)** to control false discoveries across **many** slices/
  metrics. Sort the m p‑values ascending; the largest k with \(p_{(k)} \le \frac{k}{m}q\)
  is the cutoff; reject all below it. *Use when* you slice a run 30 ways — without
  multiplicity control some slice *will* look "significant" by chance.

- **Cohen's h** — effect size between two proportions:
  \(h = 2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_2}\). *Use to* answer "is a
  statistically‑significant difference actually *big*?" (~0.2 small, 0.5 medium,
  0.8 large). Significance ≠ importance.

- **pass@k** — probability at least one of k samples passes. Unbiased estimator (Chen
  et al.): with n samples per case and c correct,
  \(\text{pass@}k = 1 - \binom{n-c}{k} / \binom{n}{k}\). *Use for* code/agent tasks
  evaluated with multiple attempts; needs `--repeats > 1`.

- **Power / sample size.** Before a run, how many cases do you need to detect a
  δ‑point difference at significance α with power 1−β? For proportions the two‑sided
  normal approximation gives
  \(n \approx \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,\bar p(1-\bar p)}{\delta^2}\) per
  group. *Use to* avoid running an eval that never could have shown the effect you care
  about — and to right‑size dataset builds.

---

## 7. Reading the logs (harness + Ollama/llama.cpp)

There are two log streams. **Harness logs** are structured JSON on `stderr` from the
Python process. **Ollama logs** are the container's stdout (`docker compose logs
ollama`) and contain both llama.cpp engine lines and Gin HTTP access lines.

### 7.1 Harness (structlog) logs

`observability.setup_logging` configures structlog to emit JSON to `stderr` with ISO
timestamps and a level, merging any bound context vars. Log level comes from
`LOG_LEVEL` (default `INFO`). Example event you'll see: `shutdown_requested` when a
SIGTERM/SIGINT arrives. When OTel is enabled (`OTEL_ENABLED=true`) the executor opens
`case` and `provider.call` spans and persists the `trace_id` on each generation row —
so you can join a log line, a DB row, and a trace. On `main` the exporter is an
in‑memory one (for tests); wire a real OTLP exporter before relying on it in an
operational setting.

Key fields to expect when bound: `run_id`, `case_id`, and per‑attempt fields. The
richest per‑request forensics, though, live in `generations.attempt_log` (query #12),
not in the log stream.

### 7.2 Decoding the Ollama/llama.cpp logs

The harness calls Ollama's `POST /api/chat` with `stream: true` (chat) and
`POST /api/embeddings` (embeddings). Ollama runs a llama.cpp server per model with a
fixed number of **slots** (here `n_slots = 1`), and prints slot/KV‑cache/timing lines.
Below are **real lines captured on this box** from a single `llama3.2:1b` generation
plus one embedding call, annotated. (These were produced by calling the provider
directly; the harness's `total_ms` for that call was ≈1.94 s, which matches the Gin
`/api/chat` line.)

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
[GIN] 2026/08/05 - 09:52:09 | 200 |   356.027125ms | 127.0.0.1     | POST "/api/embeddings"
```

Line‑by‑line:

- **`srv update_slots: all slots are idle`** — the server has no active work. Between
  requests you'll see this constantly; it means *healthy and waiting*, not stuck.

- **`slot get_availabl: ... selected slot by LRU, t_last = -1`** — the scheduler picked
  a slot to serve the request. On this Ollama build it selects by **LRU**
  (`t_last = -1` = never used). Newer builds instead log **`selected slot by LCP
  similarity, sim_best=… f_keep=…`**: LCP = *longest common prefix* between the new
  prompt's tokens and each slot's cached tokens. Picking the highest‑prefix‑overlap
  slot lets the server **reuse the KV cache** for the shared prefix and skip
  re‑computing it. `sim_best` is the best overlap fraction; `f_keep` is the fraction of
  that slot's cache it would keep. Either way the intent is the same: reuse work.

- **`slot launch_slot_: ... processing task, is_child = 0`** — the slot started the
  task. `is_child = 0` means it's a normal top‑level request (not a speculative/
  draft sub‑task).

- **`slot update_slots: new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 41`**
  — the prompt is admitted. `n_ctx_slot` is this slot's context budget (tokens);
  `task.n_tokens = 41` is the prompt length (matches the harness's `prompt_tokens = 41`);
  `n_keep = 4` is how many leading tokens are pinned (never evicted) when the context
  fills — usually the system/BOS preamble, so the model keeps its "instructions" even
  after truncation.

- **`cached n_tokens = 0, memory_seq_rm [0, end)`** — the KV‑cache reuse decision.
  `cached n_tokens = 0` means **nothing** was reused (cold slot), so `memory_seq_rm
  [0, end)` clears the whole sequence and the model does a full **prompt‑eval** over all
  41 tokens. If a later request shared a prefix you'd see `cached n_tokens = k` (k>0)
  and `memory_seq_rm [k, end)` — i.e. "keep the first k, recompute from k onward."
  `n_past` (in some builds) is that same k: the number of already‑computed tokens.

- **`print_timing`** — the two phases that make up latency:
  - **prompt eval** (a.k.a. prefill): processing the input tokens (here 41 tokens in
    57.59 ms → 711 tok/s). This dominates **TTFT**.
  - **eval time** (decode): generating output tokens one at a time (8 tokens in
    157.41 ms → 50.8 tok/s). Decode is much slower per token than prefill.
  - **total time** and **graphs reused = 7** — how many precompiled compute graphs were
    reused (a warm‑path optimization; higher is better).

  This split is exactly the harness's TTFT vs total: **TTFT ≈ queue + load +
  prompt‑eval + first decode step**; **total − TTFT ≈ the rest of decode**. Note the
  Gin line for this request shows **1.93 s** total while `print_timing` shows only
  215 ms of compute — the difference is one‑time **model load/warmup** on the first
  request (the harness measured ttft ≈ 1.78 s for the same reason). Subsequent requests
  are far faster.

- **`slot release: stop processing: n_tokens = 48, truncated = 0`** — the slot finished.
  `n_tokens = 48` is the total sequence length it processed; **`truncated = 0`** means
  the context was *not* exceeded. `truncated = 1` would mean the prompt+generation
  outgrew `n_ctx_slot` and llama.cpp dropped tokens (keeping the `n_keep` prefix) — a
  red flag that usually surfaces on the harness side as a `truncated` outcome
  (`finish_reason == length`) and should make you raise `--max-tokens`/context or
  shorten the prompt.

- **`[GIN] … 200 … POST "/api/chat"`** / **`POST "/api/embeddings"`** — Ollama's HTTP
  access log (Gin framework): timestamp, **status**, **duration**, client IP, method,
  path. This is your ground truth for per‑request latency and success. The harness's
  chat path is `/api/chat`; embeddings on `main` go to **`/api/embeddings`** (as
  captured). Newer Ollama also exposes **`/api/embed`** — if you see that path in the
  Gin log it's a newer client/endpoint, not `main`'s code. The frequent
  `GET /api/tags` / `HEAD /` lines you'll also see are health/inventory polls, not eval
  traffic.

### 7.3 Mapping Ollama signals → harness metrics/outcomes

| Ollama signal | Harness effect |
|---------------|----------------|
| `POST /api/chat` 200 + streamed chunks | `generations.raw_response = {"chunks":[…]}`; `finish_reason=stop` |
| final chunk `done_reason == "length"` | `finish_reason=length` → outcome `truncated` |
| `truncated = 1` in slot release | context overflow → likely `truncated` outcome; raise ctx/max‑tokens |
| `prompt eval time` | drives `ttft_ms` |
| `eval time` | drives `total_ms − ttft_ms` |
| HTTP `429` | `classify_error` → `RETRYABLE_RATE_LIMIT` → retried w/ full jitter |
| HTTP `5xx`, timeouts, conn/read errors | `RETRYABLE_TRANSIENT` → retried; if exhausted → `harness_error` (coverage loss) |
| HTTP `401/403` | `NON_RETRYABLE_AUTH` → immediate `harness_error` |
| other `4xx` | `NON_RETRYABLE_REQUEST` → `harness_error` |

```138:152:src/evalharness/providers/ollama.py
    def classify_error(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, httpx.TimeoutException):
            return ErrorClass.RETRYABLE_TRANSIENT
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return ErrorClass.RETRYABLE_RATE_LIMIT
            if code in (401, 403):
                return ErrorClass.NON_RETRYABLE_AUTH
            if code >= 500:
                return ErrorClass.RETRYABLE_TRANSIENT
            return ErrorClass.NON_RETRYABLE_REQUEST
```

### 7.4 Troubleshooting the log noise

- **"CPU is pegged during a run."** Expected. A quantized model on CPU will saturate
  cores during prompt‑eval and decode. High CPU *while* you see active `slot`
  lines is the system working. Concurrency is capped (default 2) precisely so a small
  box stays responsive.
- **"All slots are idle forever."** That's the healthy resting state. It is only a
  problem if the harness *thinks* it sent work but Ollama shows no `launch_slot_` — then
  check the Gin log for a non‑200, or a connection error in `attempt_log`.
- **"`truncated = 1` warnings."** Prompt+output exceeded the slot context. Raise
  `--max-tokens`, shorten the template, or use a larger‑context model. Expect a
  `truncated` outcome in the report.
- **"`GET /api/tags` every 10s."** Health/inventory polling, harmless.
- **429 / 5xx storms.** These map to retryable classes and will show as multiple
  `attempt_log` entries; if they exhaust retries they become `harness_error` and *lower
  coverage* — the run may drop below the floor and refuse to publish. Fix the backend,
  then `--resume` to fill only the gaps.

---

## 8. Operational runbook, troubleshooting & FAQ

### 8.1 Failure modes (fast reference)

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Tests skip `db_ready` | Postgres not up | `docker compose up -d postgres` |
| `evalctl run` exits `1` | Dataset invalid or config error | Read the red `ERROR` lines; fix the dataset/flags |
| `evalctl run` exits `2` | Coverage below floor | Inspect `harness_*` outcomes (query #4); fix infra; `--resume` |
| Ollama digest error at start | `/api/show` returned no digest | See §8.2 — known incompatibility on this box |
| Insert error on `metric_aggregates` | DB migrated past `main` (`0003`) | Use a fresh DB for `main` (see §5.1) |
| Duplicate key on resume | Two executors on one `run_id` | Run a single executor per `run_id` |
| Embedding insert fails (dim) | `vec vector(1024)` vs 768‑dim model | Not exercised on `main`; needs per‑model dim work |

### 8.2 The two `main`‑vs‑environment incompatibilities (important)

Both were confirmed live on this machine and are the most likely things to trip you up:

1. **Ollama version resolution.** `main`'s `OllamaProvider.resolve_version` reads a
   `digest` from `POST /api/show`. The running `ollama/ollama:latest` image returns
   *no* `digest` there (keys are `capabilities, details, license, model_info,
   modelfile, modified_at, template, tensors`); the digest is available via
   `GET /api/tags`. So a live `--provider ollama` run raises
   *"Ollama model '…' has no digest; cannot pin version."* This is fixed on the
   unmerged branch (which resolves via `/api/tags`). Workarounds today: use
   `--provider mock` for the full‑plane demo, or pin an older Ollama whose `/api/show`
   returns a digest, or run the branch.

2. **Database is ahead of `main`.** The local DB is at Alembic
   `0003_foundation_correctness`. `main`'s reporter can't insert into the migrated
   `metric_aggregates` (extra `NOT NULL metric_config_sha256`). Use a **fresh**
   database/volume to exercise `main` end‑to‑end.

### 8.3 FAQ

- **Q: I changed the normalizer. Do I have to re‑generate?** No — that's the whole
  point. Generations are immutable; a new normalizer produces a new
  `metric_config_sha256` and new `scores` rows over the *same* outputs. (On `main` you
  drive this from a script; a `runs rescore` verb is roadmap.)

- **Q: Why is my pass rate CI so wide?** Small n. Wilson is telling the truth. Add
  cases or repeats. See the 5/5 → `[56.6%, 100%]` example.

- **Q: Are `temperature=0` runs bit‑reproducible?** Not guaranteed for local models —
  that's why `Capabilities.supports_seed` and the decode params are recorded. Use the
  flaky‑case query to detect nondeterminism.

- **Q: Where's the raw model output?** `generations.output` (text) and
  `generations.raw_response` (full provider payload / streamed chunks). See query #12.

- **Q: Can I compare a run from today with one from last month?** Only if they share a
  `config_sha256` lineage that makes sense (same dataset hash, at minimum). Join on
  `cases.external_id`, and never compare across different dataset content hashes —
  that's principle #2.

- **Q: How do I add a new provider?** Implement the `Provider` protocol in a new file
  under `providers/`, then add one line under
  `[project.entry-points."evalharness.providers"]` in `pyproject.toml`. No executor,
  scorer, store, or CLI edits. That's principle #9.

---

## 9. Roadmap & known gaps (not on `main`)

For full transparency, here is what the broader design describes that is **absent from
`main`** (present on the unmerged `feat/abstraction-and-catalog` branch / `v0.2.0`
evidence bundle). Do not assume any of this exists when working on `main`:

| Area | On `main` | Roadmap (unmerged) |
|------|-----------|--------------------|
| CLI verbs | `dataset-validate`, `run` | `score`, `runs rescore`, `runs compare`, `power`, `calibrate-threshold`, `report` |
| Metrics | `exact_match` only | full catalog (embeddings/ML/calibration, judge, etc.) via `scoring/{catalog,registry,engine,calibration,embeddings,ml}.py` |
| Statistics | Wilson + percentiles (`scoring/stats.py`) | `statistics/` package: BCa/paired bootstrap, McNemar, BH, Cohen's h, pass@k, power |
| Providers | `ollama`, `mock` | `openai_compatible.py`, provider `config.py`, `runtime.py` (rate limiter, circuit breaker, capability negotiation) |
| Provider resilience | retries + full‑jitter backoff + error taxonomy in the executor | dedicated rate limiter + circuit breaker in a provider runtime |
| Timeouts | per‑request, per‑case | + enforced per‑run wall budget (with the HTTP service `evald`) |
| Reports | one JSON + one HTML (`reporting/report.py`) | JUnit + leadership/research/engineering views + Jinja template files |
| HTTP service | none | `evald` API |
| Object storage | JSONB in Postgres (deferred) | `BlobStore` (S3/filesystem) per `DEFERRED.md` triggers |
| Schema (live DB) | `main` models + migrations `0001`/`0002` | `0003_foundation_correctness` (extra indexes/columns) — *already applied locally* |

---

### Appendix: file map (where to look)

| Concern | File |
|---------|------|
| CLI | `src/evalharness/cli.py` |
| Config/settings | `src/evalharness/config.py` |
| Core types & enums | `src/evalharness/core/{models,enums,protocols}.py` |
| Executor (plan/retry/cache/resume/outcome) | `src/evalharness/execution/executor.py` |
| Providers | `src/evalharness/providers/{ollama,mock,registry}.py` |
| Scoring | `src/evalharness/scoring/{exact_match,normalizer,stats}.py` |
| Store (ORM + repo + session) | `src/evalharness/store/{models,repository,db}.py` |
| Reporting | `src/evalharness/reporting/report.py` |
| Hashing / observability | `src/evalharness/{hashing,observability}.py` |
| Datasets | `src/evalharness/datasets/{loader,validator}.py` |
| Migrations | `alembic/versions/000{1,2}_*.py` |
| Fixtures / PoC | `fixtures/sample_dataset/*`, `fixtures/templates/qa.jinja`, `fixtures/poc/*`, `scripts/run_poc.py` |
| Design docs | `docs/{architecture,dataplane,schema,operations,principles}.md` |
