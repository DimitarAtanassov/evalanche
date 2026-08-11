# Operations

## Purpose

The practical guide to **running evalanche locally**: bring up the stack, run the CLI,
observe what happens, and recover from failure. For the deeper on‑call material
(decoding Ollama/llama.cpp logs, a full troubleshooting matrix, and the FAQ), see
[guide.md §7–8](guide.md#7-reading-the-logs-harness--ollamallamacpp).

## Local stack

```bash
docker compose up -d postgres   # required for all runs, DB tests, and the PoC
docker compose up -d ollama     # only for live Ollama inference/embeddings
cp .env.example .env
uv sync --all-extras
uv run alembic upgrade head      # Alembic owns the schema (head: 0003)
```

Services ([`compose.yaml`](../compose.yaml)):

| Service | Image | Port | Required for |
|---------|-------|------|--------------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | All runs, DB tests, PoC |
| `ollama` | `ollama/ollama:latest` | 11434 | Live inference / embeddings |

Compose credentials default to user/password/db = `evalharness`. Configuration is
env‑backed via `evalharness/app/settings.py`; the keys live in `.env.example`
(`DATABASE_URL`, `OLLAMA_BASE_URL`, `HARNESS_VERSION`, `GIT_SHA`, `LOG_LEVEL`,
`LOG_FORMAT`, `LOG_PAYLOADS`, `LOG_PAYLOAD_HASHES`, `OTEL_ENABLED`, optional
`OTEL_EXPORTER_OTLP_ENDPOINT`, and the `OPENAI_COMPATIBLE_*` trio for that provider).

## CLI recipes

`evalctl` is the evalanche command‑line interface. The full reference with every flag
lives in [guide.md §4](guide.md#4-cli-command-reference--end-to-end-workflows); the
common recipes are below.

### Validate a dataset

```bash
uv run evalctl dataset-validate fixtures/sample_dataset
# Holdout sets are blocked unless you really mean it:
uv run evalctl dataset-validate path/to/holdout --i-am-doing-a-final-eval
```

### Run an evaluation

```bash
# Live Ollama
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model llama3.2:1b \
  --provider ollama

# Offline mock (CI / PoC) — deterministic, no GPU
uv run evalctl run \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa \
  --provider mock \
  --seed 42
```

### Rescore & compare (zero inference)

```bash
uv run evalctl runs rescore <run_id> --metrics exact_match,squad_f1,rouge_l
uv run evalctl runs compare <baseline_run_id> <candidate_run_id> \
  --metric exact_match --allow-compatible --output reports/comparison.json
```

### Score a JSONL file directly

```bash
uv run evalctl score outputs.jsonl --metrics exact_match,assertions
```

### Plan sample size, calibrate a threshold

```bash
uv run evalctl power --baseline-rate 0.70 --mde 0.05 --power 0.8 --alpha 0.05
uv run evalctl calibrate dev-similarities.jsonl   # dev only; reports ROC-AUC/PR-AUC/threshold
```

### Resume

```bash
uv run evalctl run --resume <run_uuid> \
  --dataset fixtures/sample_dataset \
  --template fixtures/templates/qa.jinja \
  --model mock-qa --provider mock
```

Resume re‑plans only missing keys and verifies dataset/template/model FKs and
`config_sha256` against the stored run.

**Exit codes for `run`:** `0` success + publishable, `1` validation/config error,
`2` coverage below the publish floor / not publishable.

## Proof of concept

Committed artifacts live under [`fixtures/poc/`](../fixtures/poc/). They prove
generate → score → report without GPU/Ollama, and CI runs the same path.

```bash
# Regenerate PoC artifacts (requires Postgres)
uv run python scripts/run_poc.py

# Verify against the committed golden report
uv run pytest tests/test_poc.py -q
```

For the full `v0.2.0` release evidence bundle (500 cases × 5 repeats, semantic
similarity, baseline‑vs‑candidate comparison), see `scripts/run_release_e2e.py`.

## Observability

Observability has three separate adapters over the same pipeline lifecycle:

- **Terminal progress** — interactive `evalctl run` and `runs rescore` commands show the
  current stage, completed/total, elapsed time, ETA, valid/other output counts, retries, and
  cache hits. Generation progress says `valid_outputs` rather than “passed”: this stage
  only knows whether output was generated successfully, not whether a later metric will
  accept it. Core execution emits transport‑neutral `ProgressEvent` values; Rich is
  confined to `cli_progress.py`, so a web/socket adapter can be added without changing
  the executor.
- **Structured logs** — structlog emits readable console logs on a TTY and deterministic
  JSON elsewhere (`LOG_FORMAT=auto`; force `json` or `console`). Context is bound for
  `run_id`, provider/model, case, repeat, attempt, metric, and slice. Stable lifecycle
  events cover generate → score → aggregate → report.
- **Tracing** — OpenTelemetry spans `run.generate → case → provider.call` and
  `run.score` carry model, attempt, token usage, run identity, and counts. With
  `OTEL_ENABLED=true`, tests/local runs default to an in‑memory exporter. Set
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318/v1/traces` to use the batched
  OTLP/HTTP production exporter. `trace_id` is persisted on generation rows.

### Event vocabulary

| Stage | Stable events |
|-------|---------------|
| Setup | `dataset_validation_started`, `dataset_validated`, `provider_resolution_started`, `provider_resolved`, `pipeline_planning_started`, `pipeline_planning_finished`, `run_created` |
| Generate | `generation_started`, `case_input_ready`, `cache_hit`, `provider_attempt_started`, `provider_attempt_finished`, `provider_retry_scheduled`, `case_finished`, `generation_finished` |
| Capacity | `provider_capacity_acquired`, `provider_circuit_state_changed`, `provider_circuit_rejected` |
| Score | `scoring_started`, `generation_scored`, `scoring_batch_finished`, `aggregate_written`, `scoring_finished` |
| Report | `report_build_started`, `report_build_finished`, `report_artifact_written`, `reporting_finished` |

`pipeline_finished` provides the final publishability, coverage, pass rate, and elapsed
time. Any uncaught stage error emits `pipeline_failed` with a bounded/redacted exception
summary before it is re-raised and the CLI exits non-zero.

INFO logs describe stage lifecycle plus bounded `generation_progress` checkpoints every
`LOG_PROGRESS_EVERY` cases (default 100). DEBUG adds successful per-case input/output
summaries and per-generation score/aggregate detail. Failed cases stay at WARNING so
they are never hidden by normal production verbosity. Errors are bounded and
credential‑redacted.

### Payload safety

Prompts, references, outputs, provider responses, API keys, and headers are sensitive.
They are **not emitted by default**. Payload fields contain only character count:

```json
{"event":"case_input_ready","prompt":{"chars":83}}
```

Set `LOG_PAYLOAD_HASHES=true` only when cross-event correlation is necessary. Plain
SHA‑256 is intentionally opt-in: hashes of low-entropy outputs such as `"1"` can be
reversed by enumeration and are not anonymization.

For a PII‑scrubbed development dataset only, set `LOG_PAYLOADS=true` to add a one-line,
redacted preview bounded by `LOG_PAYLOAD_MAX_CHARS` (default 240; maximum 4096). This
does not change database persistence. Full forensic records remain in
`generations.attempt_log`, `output`, and `raw_response`; join via `run_id`, `case_id`,
and persisted `trace_id` rather than copying raw content into the log platform.

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Tests skip `db_ready` | Postgres not up | `docker compose up -d postgres` |
| `evalctl run` exits `1` | Invalid dataset / config | Read the red `ERROR` lines; fix the dataset/flags |
| `evalctl run` exits `2` | Not publishable / coverage < floor | Inspect `harness_*` outcomes; fix infra; `--resume` |
| "has no digest" | Model not pulled / no digest in tags | `ollama pull <model>`; check `/api/tags` |
| Resume rejected | Dataset/template/model/config mismatch | Reuse the exact original inputs |
| Duplicate key on resume | Two executors on one `run_id` | Run a single executor per `run_id` |
| OpenAI‑compatible `ValueError` | Missing base URL / model revision env | Set the `OPENAI_COMPATIBLE_*` variables |
| `meteor` scores are NULL | NLTK wordnet resource missing | Install NLTK data or drop the metric |

The deep troubleshooting matrix (including decoding llama.cpp slot logs and the 429/5xx
→ taxonomy mapping) is in [guide.md §7–8](guide.md#74-troubleshooting-log-noise).

## Quality gates (dev)

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run lint-imports
uv run pytest -q
```

`lint-imports` checks the layering contracts declared under `[tool.importlinter]` in
`pyproject.toml`; see [architecture.md](architecture.md#seams-do-not-violate).

Performance gates (memory‑bounded planning/scoring) are described in
[benchmarks.md](benchmarks.md).

## Related

- [guide.md](guide.md) — the deep operations + onboarding guide
- [providers.md](providers.md) — provider setup and the managed runtime
- [dataplane.md](dataplane.md) — what the executor does under the hood
