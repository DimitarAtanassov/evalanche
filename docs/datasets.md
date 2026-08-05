# Dataset policy and catalog

Phase 4 separates repository-authored harness fixtures from externally sourced
benchmarks. A committed smoke fixture proves task-shape and metric wiring. It is
not evidence of model quality on the named public benchmark.

## Materialization contract

`evalctl dataset materialize` reads a local snapshot and writes
`manifest.yaml` plus `cases.jsonl`. It never downloads data and does not import
Hugging Face packages.

External snapshots require an adjacent `<snapshot>.pin.yaml`:

```yaml
revision: dev-v1.1
revision_digest: sha256:<64 lowercase hex characters>
canonical_url: https://example.invalid/canonical-snapshot
```

The digest is checked before any output is written. The pin file is an
operator-reviewed record, not a license grant. A rerun with the same source
bytes, adapter version, seed, size, and tier emits identical case bytes and
content hash.

Source records that exceed the published field bounds leave the sampling pool
before sampling, so a corpus of long documents yields its publishable records
instead of aborting the pack. When fewer in-bound records remain than the
requested size, materialization fails with `SOURCE_TOO_SMALL` and reports both
counts. Validation of the written pack remains the fail-closed gate.

## Manifest schema versions

Packs carrying `schema_version: 0.1` are held to the full Phase 4 contract:
tier, source pin, adapter identity, task metrics, contamination risk, and
per-case provenance. Manifests with no `schema_version` are legacy harness
fixtures (`fixtures/sample_dataset`, `fixtures/large_dataset`). They are still
loaded and validated against the legacy key set, and unknown keys are reported
as warnings rather than errors. This compatibility is intentional so Phase 1 to
3 fixtures keep working; new packs must be materialized, never hand-written.

## Redistribution policy

Committed source text must set `source.redistributable_smoke: true` and use one
of: `CC0-1.0`, `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`,
`CC-BY-4.0`, or `CC-BY-SA-4.0`. BY and BY-SA require attribution. BY-SA content
retains its share-alike obligations and is not relicensed as MIT.

The policy keys off the output location: any path with a `fixtures` segment is
treated as committed repository content, whether or not a `.git` directory
exists above it. A bare checkout, an unpacked tarball, and a vendored copy are
all gated. Validation applies the same rule, so a pack that lives under
`fixtures/` must carry an allow-listed license even when it declares
`redistributable_smoke: false`.

Unknown licenses, non-commercial terms, data-use agreements, gated clinical
records, and news full text are cache-only. Financial PhraseBank,
CNN/DailyMail, and XSum text must never be committed. AG News full text remains
cache-only until a reviewed license and redistribution right are recorded.
MIMIC is excluded.

Synthetic fixtures in this repository are authored for harness testing and
dedicated to `CC0-1.0`. They do not contain copied public-dataset text and do
not claim the license of any public source.

## Tiers, splits, and holdouts

- `smoke`: 5 to 20 development cases, mock provider, committed only when safe.
- `ci`: 50 to 200 development cases, normally cached or a CI artifact.
- `nightly`: 500 to 2,000 development cases from a digest-pinned snapshot.
- `release`: a frozen holdout or declared capped subset. Validation and runs
  require `--i-am-doing-a-final-eval`.

Only `release` may use `split: holdout`. Development data must not be used as a
final-evaluation holdout, and holdout data must not be used for calibration.

## Privacy and PII

Adapters publish only fields required by the task. Input strings are limited to
2,000 characters and references or labels to 500 characters. Full document
dumps and unredacted PII are rejected. Synthetic fixtures use fictional
organizations, products, people, and events. No fixture contains contact
details, patient records, account identifiers, or provider payloads.

`pii_scrubbed: true` means the adapter applied this procedure:

1. Select only task-required fields.
2. Remove direct identifiers and free-text contact or account details.
3. Drop records exceeding the published bounds instead of silently truncating
   them, and reject any that reach the written pack.
4. Review committed smoke text as repository content.

This is a publication control, not a claim that an upstream corpus is
de-identified for every use.

External cache-only adapters set `pii_scrubbed: false`. Field selection and
length enforcement alone do not prove that upstream identifiers were removed.
Repository-authored synthetic fixtures set it to true after the review above.

## Contamination policy

Public, long-lived benchmarks default to medium or high contamination risk.
Repository-authored synthetic fixtures are low risk but are public once
committed. Dataset cards record risk rather than claiming automated
decontamination. Model-quality reports must identify the exact source revision,
sample seed, and content hash.

## Task and metric policy

| Task shape | Constrained output | Primary metrics |
|---|---|---|
| Short QA | concise answer only | `squad_f1`, `exact_match` |
| Classification | one listed label only | `classification` |
| Extraction | one JSON object only | `json_validity`, `json_field_f1` |
| Summarization | concise factual summary | `rouge_l`, `chrf_pp` |
| Numeric QA | final number only | `numeric_assertion` |
| Retrieval | JSON array of document ids | `retrieval_ndcg_10` |

`meteor` is optional because its NLTK resources are not guaranteed in the
default offline test environment.

## Committed synthetic smoke cards

All cards below use revision `synthetic-v1`, split `dev`, tier `smoke`,
`CC0-1.0`, and the privacy procedure above. Their canonical source is the
corresponding file under `tools/datasets/sources/`; exact source and content
digests are recorded in generated manifests.

### synthetic_qa

Five original arithmetic and factual questions. Task: short QA. Metrics:
`squad_f1`, `exact_match`. Contamination risk: low.

### synthetic_news

Five fictional news headlines and briefs. Task: label-only classification.
Metric: `classification`. Contamination risk: low.

### synthetic_healthcare

Five fictional, non-clinical health-information questions with no patient data.
Task: label-only classification. Metric: `classification`. Contamination risk:
low.

### synthetic_finance

Five fictional business calculations with no account or market data. Task:
numeric QA. Metric: `numeric_assertion`. Contamination risk: low.

### synthetic_summarization

Five original short notices from fictional organizations. Task: concise
summarization. Metrics: `rouge_l`, `chrf_pp`. Contamination risk: low.

### synthetic_extraction

Five original fictional purchase notices. Task: JSON extraction. Metrics:
`json_validity`, `json_field_f1`. Contamination risk: low.

### synthetic_retrieval

Five original queries over fictional document titles and snippets. Task:
retrieval. Metric: `retrieval_ndcg_10`. Contamination risk: low.

### synthetic_math

Five original arithmetic word problems. Task: numeric QA. Metric:
`numeric_assertion`. Contamination risk: low.

## External adapter cards

These adapters are cache-only. `license: unknown` means this repository has not
verified an allow-listed SPDX license and redistribution right. Operators must
review upstream terms before use and must provide a digest pin.

### squad_v1.1

- Source/version: SQuAD v1.1 development snapshot, revision `dev-v1.1`.
- Canonical URL from the accepted spec:
  `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`.
- SHA-256: not verified from repository evidence. The adapter requires an
  operator pin and no SQuAD text is committed.
- License/redistribution: not asserted by this repository; cache-only.
- PII: upstream encyclopedic text; publish only reviewed bounded fields.
- Split/task/metrics: source dev, harness dev; short QA; `squad_f1`,
  `exact_match`.
- Contamination risk: high.

### ag_news

- Source/version: operator-pinned AG News CSV snapshot.
- License/redistribution: not verified here; news full text is cache-only.
- PII: upstream news text may name people; no committed text.
- Split/task/metric: operator-declared source split; classification;
  `classification`.
- Contamination risk: high.

### pubmedqa

- Source/version: operator-pinned PubMedQA labeled JSON snapshot.
- License/redistribution: not verified here; cache-only.
- PII: biomedical literature, not a clinical-record source, but reviewed
  publication controls still apply.
- Split/task/metric: operator-declared source split; classification;
  `classification`.
- Contamination risk: medium.

### financial_phrasebank

- Source/version: operator-pinned Financial PhraseBank snapshot.
- License/redistribution: non-commercial restrictions, cache-only, never git.
- PII: financial-news sentences may identify people or companies.
- Split/task/metric: operator-declared source split; classification;
  `classification`.
- Contamination risk: high.

### finqa

- Source/version: operator-pinned FinQA JSON snapshot.
- License/redistribution: not verified here; cache-only.
- PII: financial reports can identify companies and officers.
- Split/task/metric: operator-declared source split; numeric QA;
  `numeric_assertion`.
- Contamination risk: high.

### cnn_dailymail and xsum

- Source/version: operator-pinned local JSONL snapshots.
- License/redistribution: news full text is cache-only, never git.
- PII: articles may identify people; no committed text.
- Split/task/metrics: operator-declared source split; summarization;
  `rouge_l`, `chrf_pp`.
- Contamination risk: high.

### scifact

- Source/version: operator-pinned, prejoined local JSONL cases.
- License/redistribution: not verified here; cache-only.
- PII: scientific publication metadata; bounded publication controls apply.
- Split/task/metric: operator-declared source split; retrieval;
  `retrieval_ndcg_10`.
- Contamination risk: medium.

### docred

- Source/version: operator-pinned, pretransformed local JSONL cases.
- License/redistribution: not verified here; cache-only.
- PII: encyclopedic entity text may identify people; no committed text.
- Split/task/metrics: operator-declared source split; extraction;
  `json_validity`, `json_field_f1`.
- Contamination risk: high.
