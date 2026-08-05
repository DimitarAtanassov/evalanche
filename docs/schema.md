# Database schema

PostgreSQL 16 + `pgvector`. Tables are created via SQLAlchemy metadata (`init_db`) with Alembic revisions for evolution.

Raw provider payloads live in **`generations.raw_response` (JSONB)** — not object storage. Revisit triggers are local-only (`DEFERRED.md`).

## Entity relationships

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
```

## Tables

### Definitions (mostly immutable after insert)

| Table | Purpose | Key uniqueness |
|-------|---------|----------------|
| `datasets` | Named dataset version + content hash + manifest | `(name, version)` |
| `cases` | One eval case | `(dataset_id, external_id)` |
| `prompt_templates` | Template body + hash | `(name, version)` |
| `model_versions` | Provider + model + **resolved** version + quant | `(provider, model, resolved_version, quantization)` |

### Runs

| Table | Purpose |
|-------|---------|
| `runs` | Job metadata: decode params, `config_sha256`, status, tenant, harness/git versions |

Statuses: `queued` → `running` → `completed` | `failed` | `cancelled`.

### Immutable outputs

| Table | Purpose |
|-------|---------|
| `generations` | One row per `(run_id, case_id, repeat_idx)` — **never update** except a narrow interim outcome flip to `failed_score` after scoring (tracked debt: prefer score-only interpretation) |
| `scores` | Metric results referencing `generation_id` |
| `metric_aggregates` | Overall (and later slice) rollups with CI method |

### Forward-compatible (schema present, unused in current scoring paths)

`judgments`, `annotations`, `embeddings` — reserved for judge, human labels, and embedding metrics.

### Cache

`response_cache` — `cache_key` PK → JSON response payload.

## Generation columns of note

- `outcome` — failure taxonomy
- `raw_response` — provider payload JSONB
- `attempt_log` — retry history
- `trace_id` — OTel linkage
- `ttft_ms` / `total_ms` — latency

## Migrations

| Revision | Change |
|----------|--------|
| `0001_initial` | Ensures `vector` extension; metadata create is primary bootstrap |
| `0002_raw_response_jsonb` | `raw_response` JSONB; drop legacy `raw_uri` |

Apply locally after compose:

```bash
uv run python -c "import asyncio; from evalharness.store.db import init_db; asyncio.run(init_db())"
# or alembic upgrade head when using migration workflow
```

## Indexing

- GIN on `cases.slices` (`jsonb_path_ops`) for slice filters
- Uniqueness constraints above act as resume/idempotency guards
