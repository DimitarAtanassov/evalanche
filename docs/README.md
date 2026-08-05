# Documentation

System design and operations for **evalanche** (the `evalctl` CLI). These docs are one
coherent narrative: start with the mental model, follow the data as it flows from a
dataset case to a published report, then go deep on schema, metrics, providers, and
operations. Every concept has **one home**; other docs cross‑link to it rather than
repeat it.

## Where to start (by role)

| You are… | Read in this order |
|----------|--------------------|
| **A new engineer** | [guide.md](guide.md) (full onboarding) → [architecture.md](architecture.md) → [dataplane.md](dataplane.md) → [schema.md](schema.md) → [principles.md](principles.md) |
| **A researcher / eval author** | [metrics.md](metrics.md) → [metrics-catalog/](metrics-catalog/README.md) (per‑family deep dive) → [guide.md §6](guide.md#6-metric-catalog--statistics) → [reports.md](reports.md) → [dataplane.md](dataplane.md#coverage-and-publishability) |
| **An operator / on‑call** | [operations.md](operations.md) → [guide.md §7–8](guide.md#7-reading-the-logs-harness--ollamallamacpp) → [providers.md](providers.md) → [benchmarks.md](benchmarks.md) |
| **A stakeholder / reviewer** | [reports.md](reports.md) → [principles.md](principles.md) → capability matrix in the [root README](../README.md) |

## Document index

| Doc | Purpose (one line) |
|-----|--------------------|
| [guide.md](guide.md) | The deep onboarding & operations guide: mental model, CLI reference, database deep‑dive with a copy‑paste SQL library, full metric + statistics detail, log decoding, and a runbook. **The single source of truth for formulas and queries.** |
| [architecture.md](architecture.md) | What the system is made of, the seams you must not violate, and a where‑to‑find‑what module map. |
| [dataplane.md](dataplane.md) | The Case → Generate → Score → Report pipeline, including the cache, retries, three‑layer timeouts, the outcome taxonomy, and how coverage/publishability are computed. |
| [schema.md](schema.md) | The PostgreSQL model (tables, constraints, indexes) aligned with `store/models.py` and Alembic `0003`. |
| [metrics.md](metrics.md) | The metric catalog as a narrative: what each metric is *for*, when to reach for it, and how lexical / structured / classification / calibration / retrieval / overlap / semantic metrics compose with statistics for honest comparison. |
| [metrics-catalog/](metrics-catalog/README.md) | The in‑depth per‑family, per‑metric drill‑down: beginner intuition, formulas, edge cases, registered names/versions, and code links. One subdirectory per family. |
| [providers.md](providers.md) | The `Provider` protocol, the Ollama / OpenAI‑compatible / Mock adapters, the managed runtime (rate limiter + circuit breaker), and how to add a backend. |
| [reports.md](reports.md) | The JSON / HTML / JUnit artifacts and the leadership / research / engineering views. |
| [operations.md](operations.md) | Local stack, CLI recipes, observability, failure modes, and quality gates. |
| [principles.md](principles.md) | The non‑negotiables — with a one‑line justification each — that any PR must respect. |
| [benchmarks.md](benchmarks.md) | Performance gates and how to run the 100k‑case benchmark. |

## Conventions used across these docs

- **Structure.** Each doc follows *Purpose → Concepts → Details → Related* so you can
  skim or go deep predictably.
- **Grounding.** Everything is grounded in the code on `main`. Metric names, CLI flags,
  and table columns match the source; when in doubt the source wins.
- **Deferred work** (object storage, LLM‑as‑judge, the `evald` HTTP API) is labeled as
  such and kept brief — see [`DEFERRED.md`](../DEFERRED.md) and [guide.md §8.4](guide.md#84-known-gaps--deferred).

**Committed proof of concept:** [`fixtures/poc/`](../fixtures/poc/) — mock‑provider E2E
reports generated without Ollama/GPU. Private working notes (if any) live under
`docs/private/` and are gitignored.
