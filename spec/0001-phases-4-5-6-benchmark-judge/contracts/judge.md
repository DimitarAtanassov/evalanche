# Contract: judge and calibration (Phase 6)

Status: accepted · Version: **0.1** · Consumers: `evalctl judge *`, optional suite panels

## Purpose

Versioned rubrics, pointwise/pairwise judgment artifacts, and human agreement
validation. Informational unless **holdout** calibration clears the threshold.
Dev labels never set the gate bit.

## Rubric artifact (`rubric.yaml`)

```yaml
schema_version: "0.1"
name: helpfulness
version: "1.0.0"
mode: pointwise              # pointwise | pairwise
scale:
  min: 1
  max: 5
  anchors:
    1: "Useless or wrong"
    5: "Fully helpful and correct"
instructions: |
  Reason step by step. Score only after reasoning.
require_reasoning_before_score: true
calibration:
  agreement_threshold: 0.60
  agreement_metric: cohen_kappa   # cohen_kappa | spearman | krippendorff_alpha
  min_holdout_n: 150
  min_dev_n: 50                  # recorded for tuning; never used for gating_allowed
```

`forbidden_candidate_families` is not a bypass. Family separation is **mandatory**
whenever `gating_allowed` may become true (see below).

## `evalctl judge run` (CLI + input contract)

Same depth and file-primary discipline as `judge validate`. No Postgres, no HF,
no network on the default CI path. Candidate text and ids come from an **input
file**, never from a live DB read for CI.

### Command (pointwise)

```text
uv run evalctl judge run \
  --mode pointwise \
  --rubric fixtures/judge/rubric.yaml \
  --candidates fixtures/judge/candidates-pointwise.jsonl \
  --provider mock \
  --model mock-judge \
  --judge-family qwen \
  --candidate-family llama \
  --responses fixtures/judge/mock-judge-responses.jsonl \
  --seed 42 \
  --output /tmp/judgment.json
```

### Command (pairwise)

```text
uv run evalctl judge run \
  --mode pairwise \
  --rubric fixtures/judge/rubric.yaml \
  --pairs fixtures/judge/pairs.jsonl \
  --provider mock --model mock-judge \
  --judge-family qwen --candidate-family llama \
  --responses fixtures/judge/mock-judge-responses.jsonl \
  --seed 42 \
  --output /tmp/judgment-pairwise.json
```

### Required flags

| Flag | Required | Meaning |
|------|----------|---------|
| `--mode` | yes | `pointwise` \| `pairwise`; must match `rubric.mode` |
| `--rubric` | yes | Path to `rubric.yaml` (rubric name/version copied into output) |
| `--candidates` | pointwise | JSONL of pointwise items (below) |
| `--pairs` | pairwise | JSONL of pairwise items (below) |
| `--provider` | yes | Judge provider id: `mock` \| `ollama` \| `openai_compatible` |
| `--model` | yes | Judge model name; resolved to `resolved_version` digest |
| `--judge-family` | yes | Recorded as `judge_model_family` (fail closed if empty) |
| `--candidate-family` | yes | Recorded as `candidate_model_family` (fail closed if empty) |
| `--responses` | for `mock` | Deterministic judge-response fixture (below); required when `--provider mock` so CI is hermetic |
| `--seed` | yes | Seed for any ordering the runner controls |
| `--output` | yes | Destination `judgment.json` |

`--provider mock` + `--responses` is the deterministic local test double: no DB,
no network, byte-stable output. Real providers reuse the existing
`ManagedProvider` seam with a **separate** judge RPM/TPM/timeout budget.

### Input: pointwise candidates file (`candidates-pointwise.jsonl`)

One JSON object per line. This file is the source of candidate text and ids;
the runner does not invent them.

```json
{
  "case_id": "case-00001",
  "generation_id": "gen-abc123",
  "prompt": "What is 1 plus one?",
  "candidate_text": "2",
  "reference": "2"
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `case_id` | yes | Stable case identity; copied to the judgment item |
| `generation_id` | yes | Ties the item back to the immutable generation row |
| `candidate_text` | yes | Text to judge; truncated per published-text bounds |
| `prompt` | optional | Included in the judge prompt when present |
| `reference` | optional | Reference answer when the rubric uses one |

### Input: pairwise pairs file (`pairs.jsonl`)

Declares the A/B pairing explicitly; the runner does not derive pairs. Each line
is one pair and produces **both** orderings.

```json
{
  "case_id": "case-00001",
  "a_generation_id": "gen-A-1",
  "b_generation_id": "gen-B-1",
  "a_model_label": "llama3.2:3b",
  "b_model_label": "qwen2.5:3b",
  "prompt": "Summarize the passage.",
  "a_text": "…candidate A…",
  "b_text": "…candidate B…"
}
```

`a_model_label` / `b_model_label` are the node identities used to build the
Bradley–Terry graph. A pair with the same label on both sides is a hard error
(`SELF_PAIR`).

### Input: mock judge responses (`mock-judge-responses.jsonl`)

Deterministic double keyed so CI needs no live model. Pointwise keys by
`generation_id`; pairwise keys by `(case_id, swap_position)`.

```json
{"generation_id": "gen-abc123", "score": 4, "reasoning": "Direct and correct."}
{"case_id": "case-00001", "swap_position": 0, "preference": "A", "reasoning": "…"}
{"case_id": "case-00001", "swap_position": 1, "preference": "A", "reasoning": "…"}
```

A missing key for a requested item is a hard error (`MOCK_RESPONSE_MISSING`), so
fixtures stay complete and swap-flip fixtures are explicit.

### Pinned judge identity written into `judgment.json`

Every run records, from the flags and provider resolution:
`judge_model.provider`, `judge_model.model`, `judge_model.resolved_version`
(digest), `judge_model_family`, `candidate_model_family`, `rubric_name`,
`rubric_version`. These are the fields `judge validate` later checks for family
separation and rubric match. `gating_allowed` is always written `false` here.

### Done-when → fixture (judge run)

| Done-when | Deterministic fixture |
|-----------|-----------------------|
| Pointwise artifact schema + reasoning truncation | `candidates-pointwise.jsonl` + mock responses with an over-long `reasoning` |
| Pairwise both orderings + swap consistency | `pairs.jsonl` + mock responses for `swap_position` 0 and 1 |
| Swap flip ⇒ tie | mock responses that disagree across `swap_position` |
| BT refuse on disconnected graph | `pairs.jsonl` whose `*_model_label`s form two components |
| Cost/latency/digest/rubric recorded | assert those fields present and non-null in output |

## Judgment run artifact (`judgment.json`)

Produced by `evalctl judge run`. Always ships with `gating_allowed: false`.
It does not compute agreement.

```json
{
  "schema_version": "0.1",
  "mode": "pairwise",
  "rubric_name": "helpfulness",
  "rubric_version": "1.0.0",
  "judge_model": {
    "provider": "ollama",
    "model": "llama3.2:3b",
    "resolved_version": "sha256:..."
  },
  "candidate_model_family": "llama",
  "judge_model_family": "qwen",
  "gating_allowed": false,
  "gating_block_reason": "Judgment artifacts are informational until a passing calibration digest is merged",
  "calibration_digest": null,
  "cost_usd_total": 0.0,
  "latency_ms": {"p50": 0, "p95": 0},
  "items": [],
  "pairwise_summary": {
    "n_pairs": 0,
    "swap_consistency": null,
    "position_bias": null,
    "bradley_terry": null
  }
}
```

### Published text bounds (judgment items)

Aligned with Phase 3 payload policy (`EXAMPLE_TEXT_LIMIT = 280` for galleries;
reasoning may be slightly longer but still bounded):

| Field | Max chars in published JSON/HTML |
|-------|----------------------------------|
| `reasoning` | 1_000 (truncate with ellipsis) |
| Evidence / quoted spans | 280 each |
| Suite/judge galleries | 280; never include raw provider payloads |

Redact credential-like tokens using the same sanitizer spirit as
`observability.sanitize_text`. Adapter/judge tests assert caps.

### Pointwise item

`generation_id` / case id, `score` (int), truncated `reasoning`, `evidence`
object, optional harness error outcome (no fake score).

### Pairwise item

Both orderings required:

```json
{
  "case_id": "...",
  "a_generation_id": "...",
  "b_generation_id": "...",
  "orderings": [
    {"swap_position": 0, "preference": "A", "reasoning": "..."},
    {"swap_position": 1, "preference": "B", "reasoning": "..."}
  ],
  "consistent": false,
  "final_preference": "tie"
}
```

Inconsistent orderings ⇒ `final_preference: "tie"` for aggregation.

### Bradley–Terry (concrete refuse rule)

Build an **undirected** graph whose nodes are model identities under comparison
(e.g. candidate system labels A/B/…). Add an edge `{u,v}` for every pairwise
item that produced a non-tie `final_preference` after swap resolution (ties do
not add edges).

**Accept BT** only when all of the following hold:

1. Let `V` = every model that appears in at least one pairwise item.
2. `|V| >= 2`.
3. The subgraph induced by `V` has **exactly one** connected component
   (every node in `V` is reachable from every other).
4. Every node in `V` has **degree >= 1**.

Otherwise refuse and set:

```json
"bradley_terry": {
  "status": "refused",
  "reason": "DISCONNECTED_PAIRWISE_GRAPH",
  "n_models": 0,
  "n_edges": 0,
  "component_sizes": [],
  "isolated_models": []
}
```

No stronger sparsity rule in Phase 6. Do not invent alternate connectivity
heuristics in code without a contract bump.

## Human label schema (files primary)

Primary path is JSONL label files (not DB). Optional later: load into
`annotations`. Each line:

```json
{
  "schema_version": "0.1",
  "rubric_name": "helpfulness",
  "rubric_version": "1.0.0",
  "case_id": "...",
  "label_shape": "ordinal_score",
  "value": 4,
  "split": "holdout",
  "label_set_id": "helpfulness-holdout-v1"
}
```

`split` is required: `dev` | `holdout`.  
`label_set_id` identifies the label artifact; holdout and dev must use
**distinct** ids.  
`label_shape`: `ordinal_score` | `preference` | `nominal`.

## Calibration artifact (`calibration.json`) — single source of truth for the gate bit

Produced **only** by:

```text
uv run evalctl judge validate \
  --judgments <judgment.json> \
  --labels-dev <dev-labels.jsonl> \
  --labels-holdout <holdout-labels.jsonl> \
  --rubric <rubric.yaml> \
  --output <calibration.json>
```

`--rubric` is **required**: thresholds (`agreement_threshold`, `min_holdout_n`,
`min_dev_n`, `agreement_metric`) come from the rubric that produced the
judgment, and its name/version must match the judgment. There is no default
threshold fallback, so a stricter rubric can never be cleared at 0.60 by
omitting the flag.

File paths are the primary and CI path. DB reads are optional later and must not
be required for C4 verify.

### Mechanical separation

| Split | Purpose | May set `gating_allowed`? |
|-------|---------|---------------------------|
| `dev` | Tune prompts/rubrics; record agreement for diagnosis | **Never** |
| `holdout` | Gate eligibility only | Yes, if all rules pass |

### `gating_allowed` is true only when all hold

1. Agreement for the gate bit is computed **exclusively** on labels with
   `split: holdout` (and matching `label_set_id` for that holdout artifact).
   Using `dev` labels for this number is a hard error (`DEV_USED_FOR_GATE`).
2. `n_holdout >= min_holdout_n` (default 150), where `n_holdout` counts
   **distinct labeled `case_id`s paired once each**, never judgment item
   volume. A `case_id` repeated in the judgment or in a label file is a hard
   error (`DUPLICATE_CASE_ID`).
3. A separate **dev** calibration section **must be recorded** in the same
   `calibration.json` (`label_set_id` distinct from holdout, `n_dev >=
   min_dev_n`, `agreement_dev` present for diagnosis). That `agreement_dev`
   value is **never** used in the gate predicate. Missing `--labels-dev` ⇒
   `gating_allowed: false`.
4. **Family separation is mandatory** whenever the bit can become true:
   `judge_model_family != candidate_model_family` (case-insensitive string
   compare on recorded fields). Empty `forbidden_candidate_families` does
   **not** bypass this. If either family field is empty → fail closed.
5. `agreement_holdout >= agreement_threshold` (default 0.60).
6. Rubric name/version matches judgment artifact.

Phase 6 CI uses a **synthetic holdout** fixture with `split: holdout`. Never use
production `dev` labels to clear the gate bit.

### Calibration payload (required fields)

```json
{
  "schema_version": "0.1",
  "calibration_digest": "sha256:...",
  "judgment_digest": "sha256:...",
  "rubric_name": "helpfulness",
  "rubric_version": "1.0.0",
  "holdout": {
    "label_set_id": "helpfulness-holdout-v1",
    "n": 150,
    "agreement_metric": "cohen_kappa",
    "agreement": 0.62,
    "agreement_ci": null
  },
  "dev": {
    "label_set_id": "helpfulness-dev-v1",
    "n": 50,
    "agreement_metric": "cohen_kappa",
    "agreement": 0.71,
    "agreement_ci": null
  },
  "threshold": 0.60,
  "min_holdout_n": 150,
  "family_separation_ok": true,
  "gating_allowed": false,
  "plain_language": "..."
}
```

`calibration_digest` hashes the canonical calibration body excluding itself.

## Linking judgment ↔ calibration

1. `evalctl judge validate` writes **`calibration.json` only** (never flips
   `judgment.json` in place by default).
2. `judgment.json.gating_allowed` remains **false** unless an explicit merge
   step runs:

   ```text
   uv run evalctl judge attach-calibration \
     --judgment <judgment.json> \
     --calibration <calibration.json> \
     --output <judgment-calibrated.json>
   ```

   Attach copies `gating_allowed`, `plain_language`, and `calibration_digest`
   from a calibration artifact only when `calibration.gating_allowed` is true
   and `judgment_digest` matches the judgment file. Otherwise refuse.
3. Suite panels that need the gate bit must reference **`calibration.json`**
   paths (or a judgment file only after attach). Suite must not infer
   `gating_allowed` from judgment alone when `calibration_digest` is null.

## Persistence

Optional later: map into `judgments` / `annotations`. Phase 6 C1–C4 verify on
files. Zero-inference re-eval may read stored judgments when Present; validate
CI stays file-based.

## Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `UNCALIBRATED_JUDGE` | No passing calibration digest | No for gating |
| `HOLDOUT_REQUIRED` | Gate computed on non-holdout labels | No |
| `DEV_USED_FOR_GATE` | Implementer attempted gate on `dev` | No |
| `DUPLICATE_CASE_ID` | A `case_id` repeats in judge input, judgment items, or a label file | No |
| `JUDGE_FAMILY_CONFLICT` | Families equal or missing when gating | No |
| `DISCONNECTED_PAIRWISE_GRAPH` | BT refused | No |
| `SWAP_INCOMPLETE` | Missing second ordering | No |
| `SELF_PAIR` | Pairwise labels identical on both sides | No |
| `MOCK_RESPONSE_MISSING` | `--responses` fixture lacks a requested key | No |
| `RUBRIC_VERSION_MISMATCH` | Labels vs judgments | No |
| `CALIBRATION_JUDGMENT_MISMATCH` | attach digest mismatch | No |

## Non-goals

- Blocking CI/release gates (Phase 7)
- Judge ensembles
- Treating judge as ground truth
- Embedding full sensitive source documents in published suite artifacts
- Clearing `gating_allowed` from `dev` agreement
