# Production-Grade LLM Evaluation Harness — Build Specification

> This document is a self-contained build prompt. It assumes no prior codebase, no prior
> conversation, and no organizational context. Hand it to an engineer or a coding agent as-is.

---

## Part 0 — What was wrong with the original prompt

Read this section first; it explains why the spec below is shaped the way it is.

| Gap in the original ask | Why it breaks in production |
|---|---|
| "Record model name" | Model *names* are not versions. `gpt-4o` and `llama3.1:8b` silently change under you. Without a resolved version + weight digest + quantization level, run-to-run comparison is meaningless. |
| "Compute exact match, pass rate, avg latency, error rate" | No confidence intervals, no variance across repeats, no significance test. A pass rate of 87% on n=50 is indistinguishable from 79% or 93%. Point estimates without CIs are the single most common failure of homegrown harnesses. |
| "Average latency" | Averages hide tail behavior. You need p50/p90/p95/p99, TTFT vs. total, and tokens/sec — separately for local (GPU-bound) vs. hosted (network-bound) backends. |
| "Error rate" | Conflates *harness* failures with *model* failures. A 429 from OpenAI is your problem; a refusal, a truncation, or a schema violation is the model's. Counting them together makes the metric uninterpretable. |
| No LLM-as-judge | Exact match covers maybe 20% of real evals. Everything open-ended needs a judge — with rubrics, position-bias control, and calibration against human labels. |
| No gold-label provenance | Metrics computed against unvalidated references are theater. You need annotation, inter-annotator agreement, and adjudication. |
| No slicing | Aggregate scores hide subpopulation regressions. "Overall 91%, but 62% on non-English inputs" is the finding that matters. |
| No cost accounting | Tokens in/out, $/run, $/correct-answer, and budget guards. Evals are the second-largest LLM spend line at most companies. |
| No caching / resumability | A 20k-case run that dies at case 19,000 must resume, not restart. |
| Retrieval metrics named but no retrieval task | P@k / NDCG / MRR / MAP require qrels with graded relevance, ranked candidate lists, and a retrieval task type. None of that exists in the original design. |
| BLEU/ROUGE/METEOR named without tokenizer spec | BLEU is not a number, it's a *family* of numbers. Unspecified tokenization makes scores non-comparable across tools. Use SacreBLEU with a recorded signature. |
| Cosine similarity as a metric | Raw cosine has no absolute meaning. It requires a pinned embedding model, L2 normalization, and a threshold calibrated on a dev split — report ROC-AUC, not a magic 0.8. |
| No structured-output / tool-use eval | Modern workloads are JSON and tool calls. Need schema-validity rate, field-level accuracy, tool-selection accuracy, argument accuracy. |
| No safety eval | Jailbreak resistance, over-refusal, under-refusal, PII leakage, prompt-injection resistance. |
| "Expose an endpoint" | Needs async job semantics (202 + job id), idempotency keys, backpressure, quotas, multi-tenancy, auth, and a versioned OpenAPI contract. A synchronous POST that blocks for 40 minutes is not an API. |
| "Setup a database" | Needs an explicit immutable-generations schema, separation of generation from scoring, and the ability to re-score historical outputs with a new metric *without re-running inference*. This one design decision saves more money than everything else combined. |

**The load-bearing architectural idea the original prompt is missing:**
*Generation and scoring are separate, independently versioned stages joined by a durable store.*
You generate once. You score many times. Re-scoring a six-month-old run with a new rubric must cost
zero inference dollars.

---

## Part 1 — Role and objective

You are building `evalctl` + `evald`: a reproducible, resumable, statistically honest evaluation
harness for language models, plus a service that exposes it to other teams.

Primary backend is **Ollama** (local). Frontier backends (OpenAI, Anthropic, Google, Bedrock,
vLLM, any OpenAI-compatible endpoint) must be addable by implementing **one interface** and
registering it — no changes to the runner, scorer, store, or API.

Target scale: 10^5 cases per run, 10^2 concurrent in-flight requests, runs resumable across
process restarts, results queryable for two years.

---

## Part 2 — Non-negotiable principles

1. **Immutability.** A `generation` row is written once and never updated. Scores are separate rows
   referencing it. Corrections are new rows, never mutations.
2. **Content addressing.** Datasets, prompt templates, and configs are hashed (SHA-256 of canonical
   JSON). A run records the hashes. If a hash changes, it is a different run — the tool refuses to
   compare them silently.
3. **Harness errors are never model errors.** Two distinct counters, always reported separately.
   A case with a harness error is `excluded` from denominators and reported as coverage loss.
4. **No point estimate without an interval.** Every reported rate ships with a 95% CI. Every
   model-vs-model claim ships with a paired significance test.
5. **Generation ≠ scoring.** Enforced by module boundaries and by the schema.
6. **Everything is versioned:** dataset, prompt template, model, decoding params, metric
   implementation, judge model, judge rubric, embedding model, normalizer ruleset.
7. **Deterministic where possible, honest where not.** Record `seed`, `temperature`, `top_p`,
   `top_k`, and whether the provider actually honors seeding (most don't guarantee it).

---

## Part 3 — System architecture

```
                        ┌──────────────────────────────────────────────┐
                        │  evald (FastAPI)                             │
  consumers ───HTTP───▶ │  POST /v1/evaluations   (async, 202 + job)   │
                        │  POST /v1/score          (sync, no inference)│
                        │  GET  /v1/runs/{id}/report                   │
                        └───────────────┬──────────────────────────────┘
                                        │ enqueue
                                        ▼
   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐
   │  Dataset   │──▶│   Planner    │──▶│  Executor  │──▶│  Store   │──▶│  Scorer    │
   │  Loader    │   │ (matrix +    │   │ (async,    │   │(Postgres │   │ (metric    │
   │ +validator │   │  sampling +  │   │  bounded,  │   │ +pgvector│   │  registry) │
   └────────────┘   │  resume plan)│   │  resilient)│   │ +S3 blobs│   └─────┬──────┘
                    └──────────────┘   └─────┬──────┘   └──────────┘         │
                                             │                                ▼
                                       ┌─────▼──────┐                  ┌────────────┐
                                       │  Provider  │                  │ Aggregator │
                                       │  Registry  │                  │ (slices +  │
                                       │ ollama/oai/│                  │  CIs +     │
                                       │ anthropic/ │                  │  sig tests)│
                                       │ vertex/... │                  └─────┬──────┘
                                       └────────────┘                        ▼
                                                                      ┌────────────┐
                                                                      │  Reporter  │
                                                                      │ JSON/HTML/ │
                                                                      │ JUnit/diff │
                                                                      └────────────┘
```

**Data flow, one case:**

```
case ──▶ render(prompt_template, case.inputs) ──▶ cache lookup (hash of
   {provider, model_version, rendered_prompt, decode_params, adapter_version})
     │ hit ──────────────────────────────────────────────────▶ generation row (cached=true)
     │ miss ─▶ rate limiter ─▶ circuit breaker ─▶ provider.generate()
     │            │ retryable error ─▶ backoff+jitter ─▶ retry (≤N)
     │            │ non-retryable ─▶ harness_error row
     │            ▼
     └──────────▶ generation row {output, finish_reason, tokens, timings, attempt_log}
                          │
                          ▼
                  scorer.score(generation, case) ──▶ score rows (one per metric)
                          │
                          ▼
                  aggregator ──▶ metric_aggregate rows (overall + per slice)
```

---

## Part 4 — Core abstractions

Implement exactly these seams. Python 3.12, `typing.Protocol`, no inheritance-based frameworks.

```python
# ---------- 4.1 Provider seam: the ONLY thing you implement to add a backend ----------


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[Message]  # normalized chat form
    max_tokens: int | None
    temperature: float
    top_p: float | None
    top_k: int | None
    seed: int | None
    stop: list[str]
    response_format: JsonSchema | None  # structured output
    tools: list[ToolSpec] | None
    timeout_s: float


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"]
    prompt_tokens: int | None
    completion_tokens: int | None
    logprobs: list[TokenLogprob] | None
    ttft_ms: float | None
    total_ms: float
    raw: dict  # provider payload, stored as blob


class Capabilities(TypedDict):
    supports_seed: bool
    supports_logprobs: bool
    supports_tools: bool
    supports_json_schema: bool
    supports_streaming: bool
    supports_system_role: bool
    max_context_tokens: int


class Provider(Protocol):
    name: str

    async def resolve_version(self, model: str) -> ModelVersion:
        ...
        # Ollama: digest from /api/show. OpenAI/Anthropic: dated snapshot id.
        # MUST fail loudly if only a floating alias is available.

    def capabilities(self, model: str) -> Capabilities: ...
    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse: ...
    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
    def classify_error(self, exc: Exception) -> ErrorClass:
        ...
        # RETRYABLE_TRANSIENT | RETRYABLE_RATE_LIMIT | NON_RETRYABLE_REQUEST |
        # NON_RETRYABLE_AUTH | MODEL_REFUSAL | CONTENT_FILTER
```

Registration is via entry points (`evalctl.providers`) so third parties add backends without
touching the repo.

```python
# ---------- 4.2 Metric seam ----------


class Metric(Protocol):
    name: str
    version: str  # bump on ANY behavior change; stored per score row
    task_types: frozenset[TaskType]
    requires: frozenset[Requirement]  # REFERENCE | QRELS | EMBEDDINGS | JUDGE | LOGPROBS

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]: ...
    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        ...
        # NB: aggregation is metric-specific. Corpus BLEU ≠ mean of sentence BLEUs.
        # Micro-F1 ≠ mean of per-example F1. Do not assume mean().
```

```python
# ---------- 4.3 Task types ----------
class TaskType(StrEnum):
    GENERATION = "generation"  # open-ended text
    CLASSIFICATION = "classification"  # label from fixed set
    EXTRACTION = "extraction"  # structured JSON out
    SUMMARIZATION = "summarization"
    QA_SHORT = "qa_short"  # normalized EM / token-F1
    RETRIEVAL = "retrieval"  # ranked list vs qrels
    RAG = "rag"  # retrieval + grounded generation
    TOOL_USE = "tool_use"
    AGENT_TRAJECTORY = "agent_trajectory"  # multi-step
    SAFETY = "safety"
    PAIRWISE = "pairwise"  # A/B preference
```

---

## Part 5 — Dataset contract

Datasets are JSONL, one case per line, plus a sidecar `manifest.yaml`.

```jsonc
{
  "id": "case-00417",                       // stable, unique, never reused
  "task_type": "rag",
  "inputs": {"question": "...", "context_docs": ["..."]},
  "reference_answer": "...",                // optional
  "references": ["...", "..."],             // multi-reference for BLEU/METEOR
  "expected_label": "REFUND",               // classification
  "expected_json": {...},                   // extraction; validated against schema
  "qrels": {"doc_17": 3, "doc_2": 1},       // graded relevance, 0..3
  "slices": {"lang": "es", "difficulty": "hard", "tenant": "acme", "source": "prod_2026q2"},
  "must_contain": ["policy 4.1"],           // cheap deterministic assertions
  "must_not_contain": ["as an AI"],
  "canary": "EVALCANARY:8f3a...",           // contamination probe
  "weight": 1.0,
  "provenance": {"annotator_ids": ["a3","a9"], "adjudicated": true, "iaa_kappa": 0.81}
}
```

**Manifest requirements:** name, semver, content SHA-256, split (`dev`|`test`|`holdout`), license,
PII-scrub status, creation date, and a `slices` declaration listing which slice keys are required
on every case (validator enforces).

**Loader must enforce:**
- No duplicate ids; no duplicate normalized prompts (dedupe report).
- Every case has the fields its `task_type` requires (fail fast, list all violations).
- Holdout split is refused unless `--i-am-doing-a-final-eval` is passed. Repeated holdout use is
  how you overfit to your own benchmark.
- Contamination check: report cases whose normalized prompt appears in a configured
  known-public-benchmark bloom filter, and check whether the model reproduces `canary` strings.

---

## Part 6 — Metric catalog

Implement all of the following. Each is a `Metric`. Each has a version string.

### 6.1 Deterministic / lexical
- **Exact match** — with an explicit, versioned `Normalizer` (lowercase, strip articles, strip
  punctuation, collapse whitespace, unicode NFKC, number canonicalization with `--numeric-tol`).
  The normalizer ruleset id is stored on every score row. Never hardcode.
- **Token-level F1 / precision / recall** (SQuAD-style), same normalizer.
- **Fuzzy match** — normalized Levenshtein ratio with a calibrated threshold.
- **Regex / assertion metrics** — `must_contain`, `must_not_contain`, numeric-within-tolerance.
- **Schema validity rate** — for `extraction`/`tool_use`: parses-as-JSON rate, validates-against-schema
  rate, and **field-level accuracy** (per-key precision/recall over the flattened object).

### 6.2 Classification
Per-class and aggregate: precision, recall, F1 (**macro / micro / weighted — report all three**),
accuracy, balanced accuracy, **Matthews correlation coefficient** (the honest metric under class
imbalance), confusion matrix, Cohen's kappa vs. gold. If scores/logprobs available: ROC-AUC, PR-AUC.

### 6.3 Calibration (requires logprobs or elicited confidence)
- **Expected Calibration Error (ECE)**, 15 equal-mass bins; also adaptive-bin ECE.
- **Brier score**, **negative log-likelihood**, reliability diagram data.
- **Selective prediction**: risk–coverage curve, AURC, accuracy at 80% coverage.
  This is what tells you whether the model knows when it doesn't know.

### 6.4 Ranking / retrieval (requires `qrels`)
For cutoffs k ∈ {1, 3, 5, 10, 20}, per-query then averaged:

- **Precision@k** = |relevant ∩ top-k| / k
- **Recall@k** = |relevant ∩ top-k| / |relevant|
- **Hit rate@k** (a.k.a. success@k) = 1 if any relevant in top-k
- **MRR** = mean over queries of 1/rank of first relevant item
- **MAP** = mean over queries of AP, where AP = Σ_k P@k · rel(k) / |relevant|
- **NDCG@k** = DCG@k / IDCG@k, with **DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log₂(i+1)**
  Use the exponential gain form and state it in the report; the linear-gain variant gives different
  numbers and is a classic source of cross-team disagreement.
- **Recall ceiling** — recall@k of the retriever alone, which upper-bounds the whole RAG system.

Edge cases you must handle explicitly and document: queries with zero relevant docs (exclude from
recall, note count), ties in the ranking (break deterministically by original order and say so),
truncated lists shorter than k.

### 6.5 Summarization / generation overlap
- **ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum** — report precision, recall, *and* F; stemming on/off
  recorded; multi-reference = max over references.
- **BLEU** — **SacreBLEU only**, and store the full signature
  (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0`). Report corpus-level, not the mean
  of sentence BLEUs. Provide sentence-BLEU separately for per-case inspection only.
- **chrF++** — better than BLEU for morphologically rich languages; cheap; include it.
- **METEOR** — with the language's stemmer/synonym resources declared.
- **BERTScore** — pinned model (`microsoft/deberta-xlarge-mnli` for English), with baseline
  rescaling on, layer index recorded.

State plainly in the report: *these metrics measure surface overlap and correlate weakly with human
judgment on abstractive tasks.* They are regression tripwires, not quality measures.

### 6.6 Semantic similarity (cosine)
- Pinned embedding model + revision (e.g. `bge-m3`, `nomic-embed-text` via Ollama, or
  `text-embedding-3-large`). Record model + dimension + normalization.
- **L2-normalize before dot product.** Store embeddings in `pgvector` with an HNSW index.
- **Do not ship a hardcoded threshold.** Fit the threshold on the dev split by maximizing F1 against
  gold binary labels; report **ROC-AUC and PR-AUC** as the threshold-free quality measure, and the
  chosen operating point with its dev-set F1.
- Also implement **max-over-references** and **asymmetric** (query-doc) variants.

### 6.7 RAG-specific (requires an NLI model and/or judge)
- **Context precision@k / context recall** — are the retrieved chunks the ones needed?
- **Faithfulness / groundedness** — decompose the answer into atomic claims (small LLM), then run an
  **NLI entailment model** (`deberta-v3-large-mnli` or similar) of each claim against the retrieved
  context. Score = fraction entailed. Report the unsupported claims verbatim — that list is the most
  actionable artifact the harness produces.
- **Answer relevance** — embedding similarity between the question and questions back-generated from
  the answer.
- **Citation attribution** — precision/recall of cited span ids vs. the spans that actually support
  each claim.

### 6.8 LLM-as-judge (see Part 7)
- **Pointwise rubric score** (integer scale, anchored).
- **Pairwise preference** → aggregated via **Bradley–Terry** (report ability estimates with CIs) or
  Elo; report win/loss/tie rates and the swap-consistency rate.
- **Reference-guided grading** for tasks with a gold answer but flexible surface form.

### 6.9 Safety
- Attack success rate against a jailbreak suite (report per attack family).
- **Over-refusal rate** on a benign-but-sensitive control set — a model that refuses everything
  scores perfectly on the attack suite and is useless. Both numbers or neither.
- Toxicity (Detoxify or an equivalent classifier, threshold recorded).
- PII leakage (Presidio + regex for the entity types you care about).
- Prompt-injection resistance: fraction of cases where an injected instruction in the *context*
  overrides the system instruction.

### 6.10 Operational metrics (per case and aggregated)
- `ttft_ms`, `total_ms`, `queue_wait_ms`, `tokens_per_second`
- Latency reported as **p50 / p90 / p95 / p99 / max**, never mean alone. Also report mean for
  capacity math, labeled as such.
- `prompt_tokens`, `completion_tokens`, `cost_usd` (from a versioned price table).
- **Cost per correct answer** = total cost / count(passed). This is the metric procurement actually
  needs and no one computes.
- Retry count, retry reason histogram, cache hit rate.
- `finish_reason` distribution — a rising `length` rate is a silent quality regression.

### 6.11 Failure taxonomy (mandatory, mutually exclusive)
Every case terminates in exactly one outcome:

```
PASSED | FAILED_SCORE | REFUSED | TRUNCATED | SCHEMA_INVALID | EMPTY_OUTPUT |
CONTENT_FILTERED | MODEL_ERROR | HARNESS_TIMEOUT | HARNESS_ERROR | SKIPPED
```

Model-quality denominators exclude `HARNESS_*`. Report coverage = 1 − (harness failures / total)
and refuse to publish a run below a configurable coverage floor (default 0.98).

---

## Part 7 — LLM-as-judge subsystem

This is the highest-risk component. Treat it as a model you are deploying, not a utility function.

**Requirements:**

1. **Rubrics are versioned artifacts.** A rubric is a file with: the criterion, an integer scale
   (use 1–5 or 1–7 with *anchored* descriptions per point — never an unanchored 1–10), 2–3 few-shot
   graded examples, and a required-output JSON schema `{reasoning: str, score: int, evidence: [str]}`.
   Force reasoning *before* score in the schema field order.
2. **Judge model is pinned and separate** from every candidate model in the run. Never let a model
   judge its own family without flagging it — self-preference bias is well documented.
3. **Position-bias control (pairwise):** run every comparison twice with A/B swapped. Report
   **swap-consistency rate**; treat inconsistent pairs as ties. A judge below ~0.75 consistency is
   not usable for that task.
4. **Calibration against humans is mandatory before a judge is trusted.** Hold out ≥150 human-labeled
   cases. Report **Cohen's κ** (categorical) or **Spearman ρ** (ordinal) and **Krippendorff's α**
   for multi-annotator sets. Gate: a judge with κ < 0.6 against humans may not be used for
   release decisions; it may only be used for exploratory triage, and the report must say so.
5. **Judge ensembles** for high-stakes runs: 3 judges, majority vote or median score; report
   inter-judge agreement.
6. **Cost and concurrency controls** separate from the candidate model's — a judge pass over 50k
   cases is its own budget line.
7. **Judge outputs are stored as first-class rows** (`judgments`), including full reasoning text, so
   you can audit and re-aggregate without re-judging.
8. **Meta-eval:** ship a `evalctl judge validate` command that runs the judge against the human
   holdout and prints κ/ρ, per-slice agreement, and the confusion matrix.

---

## Part 8 — Data model (PostgreSQL 16 + pgvector)

Blobs (raw provider payloads, full prompts >8KB) go to object storage; the DB stores URIs.

```sql
-- immutable definitions ------------------------------------------------
CREATE TABLE datasets (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL, version TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  split TEXT NOT NULL CHECK (split IN ('dev','test','holdout')),
  manifest JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE cases (
  id BIGSERIAL PRIMARY KEY,
  dataset_id BIGINT NOT NULL REFERENCES datasets(id),
  external_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  inputs JSONB NOT NULL, reference JSONB, qrels JSONB,
  slices JSONB NOT NULL DEFAULT '{}',
  weight REAL NOT NULL DEFAULT 1.0,
  UNIQUE (dataset_id, external_id)
);
CREATE INDEX ON cases USING GIN (slices jsonb_path_ops);

CREATE TABLE prompt_templates (
  id BIGSERIAL PRIMARY KEY, name TEXT, version TEXT,
  body TEXT NOT NULL, content_sha256 TEXT NOT NULL, UNIQUE (name, version)
);

CREATE TABLE model_versions (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,            -- 'ollama' | 'openai' | 'anthropic' | ...
  model TEXT NOT NULL,               -- as requested
  resolved_version TEXT NOT NULL,    -- dated snapshot / ollama digest
  quantization TEXT,                 -- 'Q4_K_M' — CHANGES RESULTS. Always record.
  params_b REAL, context_window INT,
  capabilities JSONB NOT NULL,
  UNIQUE (provider, model, resolved_version, quantization)
);

-- runs -----------------------------------------------------------------
CREATE TABLE runs (
  id UUID PRIMARY KEY,
  dataset_id BIGINT REFERENCES datasets(id),
  prompt_template_id BIGINT REFERENCES prompt_templates(id),
  model_version_id BIGINT REFERENCES model_versions(id),
  decode_params JSONB NOT NULL,       -- temperature/top_p/top_k/seed/max_tokens
  config_sha256 TEXT NOT NULL,
  harness_version TEXT NOT NULL, git_sha TEXT NOT NULL,
  repeats INT NOT NULL DEFAULT 1,     -- >1 enables variance / pass@k
  status TEXT NOT NULL,               -- queued|running|completed|failed|cancelled
  tenant_id TEXT NOT NULL,
  started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
  baseline_run_id UUID REFERENCES runs(id)
);

-- immutable outputs ----------------------------------------------------
CREATE TABLE generations (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID NOT NULL REFERENCES runs(id),
  case_id BIGINT NOT NULL REFERENCES cases(id),
  repeat_idx INT NOT NULL DEFAULT 0,
  output TEXT, tool_calls JSONB,
  finish_reason TEXT, outcome TEXT NOT NULL,     -- failure taxonomy
  prompt_tokens INT, completion_tokens INT, cost_usd NUMERIC(12,6),
  ttft_ms REAL, total_ms REAL, queue_wait_ms REAL,
  attempts INT NOT NULL DEFAULT 1, attempt_log JSONB,
  cached BOOLEAN NOT NULL DEFAULT false,
  raw_uri TEXT, trace_id TEXT,                   -- links to OTel
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (run_id, case_id, repeat_idx)
) PARTITION BY RANGE (created_at);

-- scoring, decoupled ---------------------------------------------------
CREATE TABLE scores (
  id BIGSERIAL PRIMARY KEY,
  generation_id BIGINT NOT NULL REFERENCES generations(id),
  metric_name TEXT NOT NULL, metric_version TEXT NOT NULL,
  metric_config_sha256 TEXT NOT NULL,   -- normalizer ruleset, k, thresholds...
  value DOUBLE PRECISION, passed BOOLEAN, detail JSONB,
  scored_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (generation_id, metric_name, metric_version, metric_config_sha256)
);

CREATE TABLE judgments (
  id BIGSERIAL PRIMARY KEY,
  generation_id BIGINT REFERENCES generations(id),
  compared_generation_id BIGINT REFERENCES generations(id),  -- pairwise
  judge_model_version_id BIGINT REFERENCES model_versions(id),
  rubric_name TEXT, rubric_version TEXT,
  score INT, preference TEXT,          -- 'A'|'B'|'tie'
  swap_position INT,                    -- 0 = original order, 1 = swapped
  reasoning TEXT, evidence JSONB, cost_usd NUMERIC(12,6)
);

CREATE TABLE annotations (          -- human gold labels
  id BIGSERIAL PRIMARY KEY,
  case_id BIGINT REFERENCES cases(id),
  generation_id BIGINT REFERENCES generations(id),
  annotator_id TEXT NOT NULL, label JSONB NOT NULL,
  adjudicated BOOLEAN DEFAULT false, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE embeddings (
  id BIGSERIAL PRIMARY KEY,
  content_sha256 TEXT NOT NULL,
  embedding_model_version_id BIGINT REFERENCES model_versions(id),
  vec vector(1024) NOT NULL,
  UNIQUE (content_sha256, embedding_model_version_id)
);
CREATE INDEX ON embeddings USING hnsw (vec vector_cosine_ops);

CREATE TABLE metric_aggregates (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID REFERENCES runs(id),
  metric_name TEXT, metric_version TEXT,
  slice_key TEXT NOT NULL DEFAULT '__overall__',   -- e.g. 'lang=es'
  n INT NOT NULL, value DOUBLE PRECISION,
  ci_low DOUBLE PRECISION, ci_high DOUBLE PRECISION,
  stddev DOUBLE PRECISION, method TEXT              -- 'wilson'|'bootstrap-bca'
);

CREATE TABLE response_cache (
  cache_key TEXT PRIMARY KEY,      -- sha256(provider|model_version|prompt|params|adapter_ver)
  response JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now()
);
```

**Why this shape:** `generations` is append-only and partitioned; `scores` references it with a
metric version + config hash in the uniqueness constraint. Adding a metric = a new INSERT pass over
existing `generations`. Zero inference cost. This is the whole point.

---

## Part 9 — Statistical requirements

1. **Binomial rates** (pass rate, EM, accuracy): **Wilson score interval**, not normal
   approximation. At n < 30 or p near 0/1 the normal interval is wrong.
2. **Continuous / composite metrics** (NDCG, ROUGE, judge scores): **BCa bootstrap**, 10,000
   resamples, resampling *cases* (not sentences), seed recorded.
3. **Model A vs. model B on the same dataset**: this is paired data. Use **paired bootstrap** on the
   per-case difference, or **McNemar's test** for binary outcomes. Never an unpaired t-test.
4. **Multiple comparisons**: when reporting k slices or k models, apply
   **Benjamini–Hochberg FDR control** at q=0.05 and print adjusted p-values. Twenty slices means one
   spurious "significant" regression per run by chance.
5. **Variance from sampling**: at temperature > 0, run `repeats ≥ 5` and report the mean plus the
   between-repeat standard deviation. If run-to-run σ exceeds the A/B delta, the delta is noise —
   the reporter must say this in plain language.
6. **pass@k** for code/agent tasks, unbiased estimator:
   `pass@k = 1 − C(n−c, k) / C(n, k)` for n samples with c correct. Compute in log space.
7. **Power/sample size**: `evalctl power --baseline 0.85 --mde 0.03 --alpha 0.05 --power 0.8` prints
   the required n. Run it before building the dataset, not after the argument.
8. **Effect size, always**: absolute delta, relative delta, and Cohen's h for proportions. Statistical
   significance without effect size is how teams ship a 0.4% "win" for 3× the cost.

---

## Part 10 — Reliability engineering

- **Retries**: exponential backoff with full jitter, `base=0.5s`, `cap=30s`, max 5. Retry only
  `RETRYABLE_*` classes. Honor `Retry-After`. Every attempt appended to `attempt_log` with its error
  class and duration — retries are data, not something to hide.
- **Rate limiting**: token-bucket per (provider, model), enforcing **both RPM and TPM** — TPM is what
  actually throttles you on frontier APIs. Estimate request tokens before admission.
- **Circuit breaker** per provider: open after N consecutive non-retryable or 5xx failures, half-open
  probe after cooldown. Prevents burning a 20k-case run against a dead endpoint.
- **Concurrency**: bounded `asyncio.Semaphore` per provider, configured independently. For Ollama,
  concurrency is GPU-bound: default to 1–4 and expose `num_parallel`/`keep_alive`; oversubscribing a
  local GPU inflates latency without raising throughput and silently corrupts your latency metrics.
- **Timeouts at three layers**: per-request (provider), per-case (including retries), per-run (wall
  clock budget). All configurable; all enforced.
- **Checkpointing / resume**: the executor writes each generation before moving on. `evalctl run
  --resume <run_id>` re-plans only the cases with no terminal row. Idempotent by
  `UNIQUE(run_id, case_id, repeat_idx)`.
- **Dead-letter queue** for cases that exhaust retries, with a `--retry-dlq` command.
- **Graceful shutdown**: SIGTERM drains in-flight requests up to a deadline, then checkpoints.
- **Budget guard**: `--max-cost-usd` aborts the run and reports partial results rather than
  surprising someone with a five-figure bill.
- **Determinism harness self-test**: a nightly job runs a fixed 50-case suite twice against Ollama
  with `seed` fixed and asserts byte-identical output. If it drifts, your reproducibility claims are
  false and you need to know before a stakeholder finds out.

---

## Part 11 — Service API (`evald`)

FastAPI, OpenAPI 3.1, versioned under `/v1`. Auth via OIDC bearer tokens; tenant derived from claims.

```
POST   /v1/evaluations              # async batch. Body: dataset ref OR inline cases,
                                    # model config, metric set. Header: Idempotency-Key.
                                    # → 202 {run_id, status_url}
GET    /v1/evaluations/{run_id}     # status + progress {completed, total, failed, cost_so_far}
GET    /v1/evaluations/{run_id}/results?slice=&metric=&outcome=&cursor=   # paginated
GET    /v1/evaluations/{run_id}/report?format=json|html|junit
POST   /v1/evaluations/{run_id}/cancel
POST   /v1/evaluations/{run_id}/rescore   # apply new metrics to stored generations, NO inference

POST   /v1/score                    # SYNCHRONOUS, no generation: caller supplies
                                    # {output, reference?, qrels?, metrics[]} → scores.
                                    # This is the endpoint most consumers actually want.
POST   /v1/judge                    # synchronous rubric judgment, ≤ N items
GET    /v1/runs?model=&dataset=&since=      # comparison / leaderboard queries
POST   /v1/runs/compare             # {run_a, run_b} → deltas + paired significance tests
GET    /v1/metrics                  # registry introspection: names, versions, requirements
GET    /v1/providers                # registered backends + capabilities
GET    /healthz /readyz /metrics    # liveness, readiness (DB+provider probes), Prometheus
```

**Requirements:** idempotency keys stored 24h; per-tenant quotas (concurrent runs, monthly token
budget) with 429 + `Retry-After`; request-size caps; optional webhook callback on completion with
HMAC signature; cursor-based pagination (never OFFSET at this scale); every response carries
`trace_id`.

---

## Part 12 — Observability

- **OpenTelemetry** throughout. Span tree: `run` → `case` → `attempt` → `provider.call`, plus
  `score` and `judge` spans. Follow the OTel **GenAI semantic conventions**
  (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.) so this data joins
  with production traces from the same models.
- `trace_id` persisted on the `generations` row — from a bad score in the report you can jump to the
  exact span.
- Prometheus metrics: in-flight requests, queue depth, provider error rate by class, retry rate,
  cache hit rate, cost rate ($/min), cases/sec.
- Structured JSON logs with `run_id`, `case_id`, `attempt`, `trace_id` on every line.
- **Alerting** on: coverage below floor, cost rate anomaly, provider circuit open, run stalled.

---

## Part 13 — CI / regression gating

- `evalctl run --baseline <run_id> --gate gates.yaml` exits non-zero on regression.
- `gates.yaml` expresses thresholds as **statistically qualified** conditions:
  ```yaml
  gates:
    - metric: exact_match
      slice: __overall__
      condition: not_worse_than_baseline
      tolerance: 0.02
      require_significance: true     # only fail if the drop is significant, not noise
    - metric: refusal_rate
      condition: absolute_max
      value: 0.05
    - metric: p95_latency_ms
      condition: absolute_max
      value: 4000
    - metric: cost_per_correct_usd
      condition: not_worse_than_baseline
      tolerance_pct: 20
  ```
- JUnit XML output so any CI system renders per-metric results natively.
- **Flakiness detection**: cases whose outcome varies across repeats are tagged `flaky` and excluded
  from gating (but reported) — otherwise your gate is a random number generator.
- Golden-file test for the harness itself: a recorded-response fixture set that asserts identical
  metric outputs across harness versions. This is how you know a metric refactor was behavior-preserving.

---

## Part 14 — ML models the harness itself must run

| Purpose | Model class | Concrete default |
|---|---|---|
| Semantic similarity, retrieval eval | Embedding | `bge-m3` or `nomic-embed-text` (Ollama), `text-embedding-3-large` (hosted) |
| Reranking / relevance judgment | Cross-encoder | `bge-reranker-v2-m3` |
| Faithfulness / groundedness | NLI entailment | `deberta-v3-large-mnli` |
| BERTScore | Contextual encoder | `microsoft/deberta-xlarge-mnli` |
| Toxicity | Classifier | Detoxify (`unbiased`) |
| PII detection | NER + rules | Presidio |
| Rubric grading, pairwise preference | Judge LLM | pinned frontier snapshot, ≠ candidate family |
| Claim decomposition for faithfulness | Small LLM | a fast local model is fine here |
| Pairwise aggregation | Statistical model | Bradley–Terry MLE with bootstrap CIs |
| Language ID for slicing | Classifier | fastText `lid.176` |

All of these are pinned by revision hash and recorded in `model_versions` exactly like candidate
models. An embedding model upgrade invalidates every cosine score computed before it — the schema
must make that visible, not silent.

---

## Part 15 — Phased build plan

### Phase 1 — Core loop (target: 1 week)
**Do:** Get a reproducible single-model run end to end.
**Read:**
- SacreBLEU paper, "A Call for Clarity in Reporting BLEU Scores" (Post 2018) — *why unspecified
  tokenization makes scores non-comparable; this shapes your metric interface.*
- OpenTelemetry GenAI semantic conventions — *adopt the attribute names now, not after you have
  10^6 rows.*
- Ollama API docs, `/api/generate`, `/api/show`, `keep_alive`, `num_ctx` — *for version pinning via
  digest and for correct concurrency behavior.*

**Implement:** `Provider` protocol + Ollama adapter with `resolve_version` returning the model
digest; dataset loader + validator; async executor with semaphore, timeout, retry-with-jitter,
checkpointed writes; Postgres schema (Parts 8, all tables); exact-match with a versioned normalizer;
latency percentiles; failure taxonomy; JSON + HTML report.

**Done when:** A 500-case run against Ollama completes; killing the process at case 250 and running
`--resume` produces a run identical to the uninterrupted one (verified by comparing the set of
`(case_id, output)` pairs); the report shows p50/p95/p99, a full outcome histogram, and a Wilson CI
on the pass rate; `config_sha256` and the model digest are recorded.

---

### Phase 2 — Provider abstraction + statistics (1 week)
**Do:** Make backends swappable and make numbers defensible.
**Read:**
- Wilson vs. Wald interval — *the default `mean ± 1.96·sd/√n` is wrong at your sample sizes.*
- Efron & Tibshirani on BCa bootstrap — *the correct interval for NDCG/ROUGE-type metrics.*
- McNemar's test — *the right paired test for A/B on binary outcomes.*
- HumanEval paper, §pass@k estimator — *the unbiased formula and why the naive one is biased.*

**Implement:** OpenAI, Anthropic, Google adapters + a generic OpenAI-compatible adapter behind the
same protocol; capability negotiation with graceful degradation (if `supports_json_schema` is false,
fall back to prompt-based JSON + repair, and *record that it happened*); response cache; rate limiter
(RPM+TPM); circuit breaker; Wilson + BCa + paired bootstrap + McNemar + BH correction; `repeats`
with between-repeat σ; pass@k; `evalctl power`; `runs/compare`.

**Done when:** The same dataset runs unchanged against Ollama and two hosted providers; a
comparison report shows deltas with paired p-values, BH-adjusted, plus effect sizes; adding a fourth
provider required touching exactly one new file plus an entry-point registration (prove it with a
diff).

---

### Phase 3 — Full metric catalog (1.5 weeks)
**Do:** Cover retrieval, summarization, semantic, and classification metrics.
**Read:**
- Järvelin & Kekäläinen on NDCG — *gain formulations and why teams disagree about the number.*
- TREC evaluation conventions for qrels and unjudged documents — *how to handle the holes.*
- BERTScore paper, esp. baseline rescaling — *raw BERTScore values are misleadingly compressed.*
- pgvector HNSW tuning docs — *`m`, `ef_construction`, `ef_search` trade-offs at your corpus size.*

**Implement:** all of Part 6.1–6.6; embedding service with `pgvector` storage and dedupe by content
hash; threshold calibration command producing ROC-AUC and the chosen operating point; per-slice
aggregation; `POST /v1/score` and `rescore`.

**Done when:** `rescore` adds a brand-new metric to a completed 5,000-case run with zero provider
calls (assert `cost_usd == 0` for the operation); NDCG@10 matches a `trec_eval` reference output on
a public qrels file to within 1e-6; the cosine threshold is fit on dev and the report prints ROC-AUC
rather than a hardcoded cutoff.

---

### Phase 4 — Judge subsystem + RAG metrics (1.5 weeks)
**Do:** Make open-ended evaluation trustworthy.
**Read:**
- "Judging LLM-as-a-Judge" (MT-Bench / Chatbot Arena) — *position bias, verbosity bias,
  self-enhancement bias, and the agreement rates you should expect.*
- Krippendorff's α and Cohen's κ — *the right agreement statistics for your annotation setup.*
- RAGAS / ARES methodology — *faithfulness decomposition and context precision/recall definitions.*
- Bradley–Terry model fitting — *turning pairwise preferences into ranked abilities with CIs.*

**Implement:** rubric artifacts with anchored scales and forced reasoning-before-score; pointwise +
pairwise judges with A/B swap and swap-consistency reporting; judge ensembles; `annotations` table
and a minimal labeling CLI; `evalctl judge validate` reporting κ/ρ vs. the human holdout; the κ<0.6
gate; claim decomposition + NLI faithfulness with unsupported-claim output; context precision/recall;
citation attribution; Bradley–Terry aggregation.

**Done when:** A judge is blocked from gating a release because its κ against the 150-case human
holdout is below threshold, and the report states this in plain language; pairwise runs report swap
consistency; the faithfulness metric emits the list of unsupported claims per answer.

---

### Phase 5 — Service, CI gating, hardening (1 week)
**Do:** Make it consumable by other teams and by CI.
**Read:**
- OpenAPI 3.1 + RFC 9457 (problem details) — *machine-readable error contracts.*
- Idempotency-key patterns for async job APIs.
- OTel trace-to-metric exemplars — *linking an aggregate regression back to individual spans.*

**Implement:** all endpoints in Part 11; idempotency; per-tenant quotas and budgets; webhooks with
HMAC; cursor pagination; `gates.yaml` + JUnit output; flakiness detection; nightly determinism
self-test; harness golden-file tests; Grafana dashboard; runbook.

**Done when:** A CI pipeline fails on a statistically significant regression and passes on an
equal-sized non-significant one (both demonstrated with a synthetic fixture); a consumer team
integrates against the OpenAPI spec without reading the source; killing the DB mid-run leaves the
run resumable.

---

## Part 16 — Explicit non-goals

State these in the README so scope stays honest:
- Not a training or fine-tuning pipeline.
- Not a prompt optimizer (though it is the substrate one would sit on).
- Not a real-time production guardrail — evaluation is offline/near-line; guardrails are a separate
  latency budget and a separate system.
- Not a replacement for human review on high-stakes decisions. The harness *routes* work to humans
  and measures agreement; it does not eliminate them.

---

## Part 17 — Acceptance criteria for "production grade"

The harness is done when all of these are demonstrably true:

1. Two runs of the same config against the same pinned local model produce identical outputs at
   `temperature=0, seed=fixed` — verified by an automated nightly test.
2. Any metric can be added and applied to historical runs with zero inference cost.
3. Every reported number carries an interval; every comparison carries a paired, multiplicity-
   corrected test and an effect size.
4. Harness failures and model failures are never summed.
5. Adding a provider is one file plus one entry-point line.
6. A run that dies at any point resumes without duplicating or losing work.
7. Cost is known before (estimate), during (live), and after (actual, per correct answer).
8. Every judge in use has a published agreement statistic against human labels, and judges below the
   threshold cannot gate releases.
9. Aggregate metrics are always accompanied by slice breakdowns, and the report surfaces the worst
   slice unprompted.
10. From any number in the report you can reach: the exact prompt sent, the exact raw response, the
    OTel trace, the metric version, and the model digest.