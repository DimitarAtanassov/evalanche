# Data plane

## Purpose

This document traces the **end‑to‑end path of one evaluation** — from a dataset case to
a published report — and explains the reliability machinery around it: the response
cache, retries, the three‑layer timeout model, the outcome taxonomy, and how coverage
and publishability are computed. For the components involved, see
[architecture.md](architecture.md); for runnable commands, see
[operations.md](operations.md) and [guide.md §4](guide.md#4-cli-command-reference--end-to-end-workflows).

## The two stages

Everything here is a consequence of one rule (see [principles.md](principles.md)):

> **Generate once, score many times.** Generation writes immutable rows; scoring is a
> separate, pure stage over stored rows and never calls a provider.

`evalctl run` performs generation, then invokes `ScoringEngine.rescore_run` (default
`exact_match`), then writes reports. `evalctl runs rescore` and `evalctl score` are the
scoring stage in isolation.

## Pipeline

```mermaid
sequenceDiagram
  participant Case
  participant Executor
  participant Cache as response_cache
  participant Runtime as ManagedProvider
  participant Provider
  participant Store as generations
  participant Engine as ScoringEngine
  participant Report as Reporter

  Case->>Executor: render_prompt(template, inputs)
  Executor->>Cache: get(cache_key)
  alt cache hit (temperature == 0 only)
    Cache-->>Executor: GenerationResponse (cached=true)
  else miss
    Executor->>Runtime: generate(request)
    Runtime->>Runtime: acquire RPM/TPM + semaphore; breaker.before_call
    Runtime->>Provider: generate
    Provider-->>Runtime: text + timings + raw
    Runtime-->>Executor: response (+ runtime.queue_wait_ms)
    Executor->>Cache: put(cache_key) when temperature == 0
  end
  Executor->>Store: INSERT generation (immutable)
  Note over Engine: Separate stage — zero inference
  Engine->>Store: INSERT scores + metric_aggregates
  Report->>Store: read-only
  Report-->>Report: JSON + HTML + JUnit
```

## Cache key

The response cache is content‑addressed. The key is a SHA‑256 over canonical JSON of:

```text
{provider, model_version, rendered_prompt, decode_params, adapter_version}
```

- The cache is consulted **only when `temperature == 0.0`** (sampling runs are not
  cacheable by construction).
- Hits set `generations.cached = true` and skip inference.
- Puts are race‑safe (`ON CONFLICT DO NOTHING`) — the first result for a key wins.

If you *expected* hits and got none, something in the key changed: a different resolved
digest, template bytes, or decode params.

## Retries (executor‑owned)

Adapters must **not** retry; the executor owns retry policy so behavior is uniform
across providers.

- Retry only `RETRYABLE_TRANSIENT` and `RETRYABLE_RATE_LIMIT` error classes.
- Full‑jitter exponential backoff: base `0.5s`, cap `30s`, up to 5 retries (settings).
- Honor HTTP `Retry-After` when present (sleep is the max of jitter and server hint).
- Every attempt is appended to `generations.attempt_log` — **retries are data**, not
  something to hide.

## Timeouts (three layers)

1. **Per‑request** — `asyncio.wait_for` around the provider `generate` call
   (`default_request_timeout_s`, 60s). `httpx.TimeoutException` maps to the harness
   timeout taxonomy.
2. **Per‑case** — a wall‑clock budget including retries (`default_case_timeout_s`,
   120s); on expiry the executor writes an idempotent `harness_timeout` generation row.
3. **Per‑run** — an executor wall budget (`default_run_timeout_s`, 4h); triggers a
   drain, then a `failed` / `cancelled` status.

## Concurrency, backpressure, and shutdown

- A **bounded worker‑queue pool** (queue bounded to twice worker concurrency), not one
  task per case, so large runs stay memory‑bounded (see [benchmarks.md](benchmarks.md)).
- The [ManagedProvider](providers.md) adds RPM/TPM token buckets, a concurrency
  semaphore, and a circuit breaker; queue wait is recorded on
  `generations.queue_wait_ms`.
- SIGTERM/SIGINT sets a shutdown flag; workers drain for
  `default_shutdown_drain_timeout_s` (30s), then cancel. Final status is `completed`
  only when all planned keys are written; otherwise `failed` or `cancelled`.

## Outcome taxonomy

Every generation terminates in exactly one **mutually exclusive** outcome
(`FailureOutcome`), assigned at generation time — never mutated by scoring:

| Outcome | Meaning | In coverage denominator? |
|---------|---------|--------------------------|
| `passed` | Provider returned usable output | Yes |
| `refused` / `truncated` / `empty_output` / `content_filtered` / `model_error` | Model‑side terminal states | Yes |
| `harness_timeout` / `harness_error` | Infrastructure failure | **No** — counts as coverage loss |
| `skipped` | Explicitly skipped | No |

`failed_score` exists in the enum for historical/forward compatibility, but current
`main` derives pass/fail **quality** from the `scores` table (e.g. `exact_match.passed`)
rather than mutating `generations.outcome`. This is principle #1 and #3 in action:
harness failures never make a model look worse, and quality lives in versioned scores.

## Coverage and publishability

- **Planned cardinality** = `cases × repeats`
- **Coverage** = `(written_generations − harness_failures) / planned`
- A run is **publishable** only when *all three* hold:
  1. `run.status == completed`
  2. `written == planned` (every planned `(case, repeat)` produced a row)
  3. `coverage ≥ floor` (CLI `--coverage-floor`, default `0.98`)

A partial or cancelled run is withheld. Crucially, a run **cannot become publishable
simply because missing rows are absent** — the denominator is planned cardinality, not
whatever happened to be written. `evalctl run` exits `2` when a run is not publishable.

## Resume

`UNIQUE (run_id, case_id, repeat_idx)` plus `ON CONFLICT DO NOTHING` makes checkpoints
idempotent. `--resume RUN_ID` re‑plans only the missing keys and verifies that the
dataset, template, model version, and `config_sha256` match the stored run — you cannot
accidentally resume with different inputs.

## Reporting

After generation and scoring, the reporter (read‑only) loads generations, scores, and
aggregates; computes coverage from planned cardinality; computes the pass rate with a
Wilson 95% CI; and writes `{run_id}.json`, `{run_id}.html`, and `{run_id}.xml`. It
refuses to mark a run publishable unless the conditions above hold. See
[reports.md](reports.md).

## Offline PoC path

The **mock** provider returns deterministic text and timings, so CI and the committed
`fixtures/poc/` artifacts prove the full plane without Ollama or a GPU. See
[operations.md](operations.md#proof-of-concept).

## Related

- [architecture.md](architecture.md) — the components referenced here
- [providers.md](providers.md) — the managed runtime, limiter, and breaker
- [schema.md](schema.md) — the tables written along this path
- [guide.md §2](guide.md#2-architecture--data-plane) — the deep version with SQL and log traces
