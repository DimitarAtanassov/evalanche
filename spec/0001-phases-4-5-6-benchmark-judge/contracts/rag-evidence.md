# Contract: RAG evidence (Phase 6)

Status: accepted · Version: **0.1** · Consumers: `evalctl rag evidence`,
optional suite panels

## Purpose

Separate **retrieval quality**, **answer faithfulness**, **context
precision/recall**, and **citation attribution** so retrieval defects cannot hide
inside a single answer score. Deterministic run metrics remain visible and are
never overwritten.

## NLI / model policy

Faithfulness NLI runs **only** through the existing `Provider` seam
(digest-pinned Ollama or OpenAI-compatible), or an optional extra that is
**never imported** by suite or dataset-adapter paths.

- Core and default `pytest` must pass with faithfulness
  `status: "unavailable"` / `NLI_UNAVAILABLE` when no NLI provider is
  configured. No Hugging Face model or dataset download on the default test
  path.
- Do not add HF dataset loaders for RAG corpora (ADR-001 still applies).

## `evalctl rag evidence` (CLI + input contract)

File-primary, same discipline as `judge validate`: no Postgres, no HF, no
network on the default CI path. Retrieval numbers come from a published **run
report** (Phase 3 schema 2.1); per-case answer text, retrieved contexts, qrels,
and citations come from a local **evidence file** that acts as the deterministic
test double for what the store would otherwise supply.

### Command

```text
uv run evalctl rag evidence \
  --report fixtures/rag/report.json \
  --evidence fixtures/rag/evidence.jsonl \
  --nli-provider mock \
  --nli-model mock-nli \
  --nli-responses fixtures/rag/mock-nli-responses.jsonl \
  --output /tmp/rag_evidence.json
```

Omit `--nli-provider` (and `--nli-responses`) to exercise the honest
`NLI_UNAVAILABLE` path; retrieval, context, and citation sections still populate
from deterministic inputs.

### Required flags

| Flag | Required | Meaning |
|------|----------|---------|
| `--report` | yes | Run report 2.1; source of `run_id`, `model_digest`, `dataset_sha256`, and the retrieval aggregate (`retrieval_ndcg_10`) |
| `--evidence` | yes | Per-case evidence JSONL (below); the local double for answers, contexts, qrels, citations |
| `--nli-provider` | optional | `mock` \| `ollama` \| `openai_compatible`; absent ⇒ faithfulness `unavailable` |
| `--nli-model` | with provider | NLI model name; resolved to `resolved_version` digest |
| `--nli-responses` | for `mock` | Deterministic NLI label fixture (below); required when `--nli-provider mock` |
| `--output` | yes | Destination `rag_evidence.json` |

### Input: run report (`--report`)

Read-only. Supplies identity and the **retrieval** section only; `rag evidence`
never recomputes or overwrites deterministic retrieval scores (invariant 1). If
the report has no `retrieval_ndcg_10` aggregate, the retrieval section is written
with `status: "missing"` and `QRELS_MISSING` is reported (retrieval remains a
deterministic-metric responsibility, not this artifact's to fabricate).

### Input: evidence file (`--evidence`, JSONL)

One JSON object per case. This is the deterministic double; no DB needed.

```json
{
  "case_id": "case-00001",
  "generation_id": "gen-abc123",
  "answer_text": "The drug reduced mortality. [d2]",
  "retrieved_contexts": [
    {"doc_id": "d1", "text": "…", "rank": 1},
    {"doc_id": "d2", "text": "The trial reported lower mortality.", "rank": 2}
  ],
  "qrels": {"d2": 1, "d5": 1},
  "citations": [{"claim_index": 0, "doc_id": "d2"}]
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `case_id` | yes | Case identity |
| `generation_id` | yes | Ties to the immutable generation row |
| `answer_text` | yes | Candidate answer; claims are decomposed from this |
| `retrieved_contexts` | yes | Ordered contexts actually supplied to the model; `doc_id` + bounded `text` + `rank` |
| `qrels` | optional | Gold relevance for context precision/recall; absent ⇒ those fields `status: "unavailable"` |
| `citations` | optional | Explicit claim→doc references; absent ⇒ citation attribution `status: "unavailable"` |

Claims may instead be pre-declared with a `claims` array per case; when absent,
the runner decomposes `answer_text` (see Methods). Cap contexts and claims per
case to keep artifacts bounded.

### Input: mock NLI responses (`--nli-responses`, JSONL)

Keyed by `(case_id, claim_index, doc_id)` so CI needs no live model:

```json
{"case_id": "case-00001", "claim_index": 0, "doc_id": "d2", "label": "entailment"}
```

`label` ∈ `entailment` | `neutral` | `contradiction`. A missing key for a
requested `(claim, context)` is a hard error (`MOCK_RESPONSE_MISSING`) so
fixtures stay complete.

## Artifact: `rag_evidence.json`

```json
{
  "schema_version": "0.1",
  "run_id": "uuid",
  "model_digest": "...",
  "dataset_sha256": "...",
  "config": {
    "nli_model": {"provider": "...", "model": "...", "resolved_version": "..."},
    "nli_config_sha256": "..."
  },
  "retrieval": {
    "metric": "retrieval_ndcg_10",
    "metric_version": "...",
    "aggregate": {"value": 0.0, "n": 0, "ci_low": null, "ci_high": null},
    "notes": "From scores table; independent of faithfulness"
  },
  "faithfulness": {
    "method": "claim_nli_v1",
    "status": "ok",
    "aggregate": {"unsupported_claim_rate": 0.0, "n": 0},
    "examples": [
      {
        "case_id": "...",
        "claims": [
          {"text": "...", "supported": false, "evidence_spans": []}
        ]
      }
    ]
  },
  "context": {
    "method": "qrels_context_v1",
    "precision": {"status": "ok", "value": 0.0, "n": 0},
    "recall": {"status": "ok", "value": 0.0, "n": 0}
  },
  "citations": {
    "method": "cited_present_relevant_v1",
    "attribution": {"status": "ok", "value": 0.0, "n": 0},
    "missing_support_examples": []
  },
  "deferred": {
    "answer_grounded_context": {"status": "deferred", "reason": "ADR_004_RAG_METHODS"},
    "nli_verified_citations": {"status": "deferred", "reason": "ADR_004_RAG_METHODS"}
  },
  "bounded_examples": [],
  "cost_usd_total": 0.0,
  "gating_allowed": false
}
```

Every optional signal carries an explicit `status`, never a bare `null`:

| `status` | When | Numeric field |
|----------|------|---------------|
| `ok` | Inputs present; value computed | populated |
| `unavailable` | Required input absent (no `--nli-provider`, no `qrels`, no `citations`) | `value: null` + `reason` |
| `missing` | Retrieval aggregate absent from report | `value: null` + `reason` |
| `deferred` | NLI-grounded refinement out of Phase 6 scope | `value: null` + `reason: "ADR_004_RAG_METHODS"` |

When NLI is missing: `"faithfulness": {"status": "unavailable", "reason": "NLI_UNAVAILABLE", "aggregate": {"unsupported_claim_rate": null, "n": 0}, "examples": []}`;
retrieval, context, and citation sections still populate from deterministic inputs.

## Published text bounds

| Field | Max chars |
|-------|-----------|
| Claim `text` | 280 |
| Evidence spans | 280 each |
| `bounded_examples` / missing citation snippets | 280 |
| Full source documents | **Never** in this artifact or suite HTML |

Cap example count (default 8), same spirit as report `EXAMPLE_LIMIT`.

## Methods (Phase 6 minimal)

Each method is deterministic given its inputs and the mock doubles, so every
Done-when is meetable in CI without HF, network, or a live model. Method ids are
versioned; changing an algorithm bumps the id and the `schema_version`.

### Claim decomposition — `claim_split_v1`

Deterministic, no model. If a case supplies an explicit `claims` array, use it
verbatim. Otherwise split `answer_text` into atomic claims by:

1. Normalize whitespace; split on line breaks first.
2. Within each line, segment on sentence terminators `.`, `?`, `!` followed by
   whitespace, guarding a small fixed abbreviation set (`e.g.`, `i.e.`, `vs.`,
   `etc.`, `Dr.`, `Mr.`, `Ms.`, `Fig.`, `No.`) so they do not split.
3. Trim, drop empty fragments, cap at `max_claims_per_case` (default 20),
   truncate each claim to 280 chars, and index claims `0..k-1` in order.

This is intentionally lexical. Model-based or coreference-aware decomposition is
deferred to ADR-004. `CLAIM_PARSE_FAILED` is recorded per case if segmentation
yields zero claims from non-empty text; the case is skipped in the faithfulness
denominator but retained for reporting.

### Faithfulness NLI mapping — `claim_nli_v1`

For each claim, pair it with the case's `retrieved_contexts` (premise) and ask
the NLI provider whether the context entails the claim. Provider output maps to a
fixed label set:

| Provider label | Meaning | Effect |
|----------------|---------|--------|
| `entailment` | Context supports claim | claim `supported: true` |
| `neutral` | No support | claim `supported: false` |
| `contradiction` | Context refutes claim | claim `supported: false` (flagged) |

A claim is `supported` if **any** retrieved context yields `entailment`.
`evidence_spans` records the supporting `doc_id`(s) and a bounded quoted span.
`unsupported_claim_rate = unsupported_claims / total_claims` over cases with at
least one claim. With no NLI provider, faithfulness is `unavailable`
(`NLI_UNAVAILABLE`); claims are still listed with `supported: null` for
inspection but excluded from the rate.

### Context precision / recall — `qrels_context_v1`

Deterministic from `retrieved_contexts` + `qrels`; no model. This is the
retrieval-context view, distinct from the ranked `retrieval_ndcg_10` metric and
from the answer-grounded RAGAS-style definition (deferred, ADR-004).

- **precision** = relevant retrieved contexts / retrieved contexts, where a
  context is relevant when `qrels[doc_id] > 0`.
- **recall** = relevant retrieved contexts / total relevant contexts in `qrels`.
- Aggregate by mean over cases that supply `qrels`; `n` is that case count.
- No `qrels` on a case ⇒ that case is excluded; if no case has `qrels`, both
  fields are `status: "unavailable"`.

### Citation attribution — `cited_present_relevant_v1`

Deterministic when `citations` are present. For each `citation` (`claim_index` →
`doc_id`), the citation is **attributed** when both hold:

1. `doc_id` is in the case's `retrieved_contexts` (present), and
2. `qrels[doc_id] > 0` (relevant) **or**, when NLI is available, the cited
   context yields `entailment` for that claim.

`attribution.value = attributed_citations / total_citations`;
`missing_support_examples` lists bounded, capped cases where a claim cites a
`doc_id` that is absent or non-relevant. No `citations` ⇒ `status:
"unavailable"`. NLI-verified citation support (requiring entailment, not just
relevance) is the ADR-004 refinement and is reported under `deferred`.

## Phase 6 exit (RAG evidence, narrowed)

Phase 6 ships the deterministic methods above. It does **not** ship
answer-grounded context precision/recall or NLI-verified citation attribution;
those need methodology research and are a named follow-on ([ADR-004](../adr/004-rag-methods-minimal.md)).
Their fields are present with `status: "deferred"` and `reason:
"ADR_004_RAG_METHODS"`, never bare `null`. Faithfulness is available only through
the Provider seam and reports `unavailable` otherwise.

### Done-when → fixture (rag evidence)

| Done-when | Deterministic fixture |
|-----------|-----------------------|
| Retrieval failure separated from faithfulness failure | evidence with low `qrels` overlap but a supported claim, and vice versa |
| Unsupported claims inspectable per bounded example | mock NLI returns `neutral` for a claim; assert `supported: false` + example |
| Missing citation support inspectable | citation to a `doc_id` absent from `retrieved_contexts`; assert `missing_support_examples` |
| Deterministic metrics never overwritten | assert `retrieval` equals the report aggregate byte-for-byte |
| Honest unavailability | run without `--nli-provider`; assert `NLI_UNAVAILABLE`, retrieval/context/citation still populated |
| No unexplained nulls | assert every optional section has a `status` field |

## Invariants

1. `retrieval` section is sourced from existing `retrieval_ndcg_10` (or stated
   sibling) scores; regenerating faithfulness must not delete those scores.
2. Missing NLI ⇒ unavailable, not fabricated labels.
3. Published suite may link this file; must not inline full corpus documents.
4. `gating_allowed` stays false in Phase 6 (informational).

## Relation to metrics catalog

| Signal | Where it lives |
|--------|----------------|
| NDCG / P@k / R@k | Existing metric rows + report aggregates |
| Unsupported claims | This artifact |
| Context precision/recall | This artifact |
| Citation attribution | This artifact |

## Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `NLI_UNAVAILABLE` | Model/provider missing | Yes after config fix |
| `QRELS_MISSING` | Retrieval aggregate absent from report | No |
| `CLAIM_PARSE_FAILED` | Decomposition yielded no claims for a case | Partial OK if recorded |
| `MOCK_RESPONSE_MISSING` | `--nli-responses` fixture lacks a requested `(case, claim, doc)` key | No |

## Non-goals

- Replacing retrieval metrics with an LLM-only score
- Storing full source documents in suite HTML
- Blocking releases on faithfulness in Phase 6
- HF downloads on default pytest
