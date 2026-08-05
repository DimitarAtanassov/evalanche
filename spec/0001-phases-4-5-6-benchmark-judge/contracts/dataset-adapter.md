# Contract: dataset adapters and policy (Phase 4)

Status: accepted · Version: **0.1** · Consumers: `tools/datasets/*`,
`evalctl dataset materialize`, `evalctl dataset-validate`, `load_dataset`

## Purpose

Define how external sources become the existing harness dataset bundle without
runtime Hugging Face and without committing illegal text into this MIT-licensed
repository.

## Stable output (unchanged primary contract)

Directory:

```text
<dataset_dir>/
  manifest.yaml
  cases.jsonl
```

`cases.jsonl`: one JSON object per line matching `Case` fields understood by
`datasets/loader.py` (`id`, `task_type`, `inputs`, task-required fields, optional
`slices`, `provenance`, …).

Published case text fields (`inputs` string values, `reference_answer`,
`references`) must respect the field length caps in
[Published text bounds](#published-text-bounds). Adapters must not dump full
source documents into cases when a span or question field suffices.

## Manifest (additive, typed)

Required today (keep): `name`, `version`, `split`, `license`, `pii_scrubbed`,
`created_at`, `slices`, `content_sha256` (recommended; validated when present).

**Additive Phase 4 fields** (optional for pre-existing synthetic fixtures without
`schema_version`; **required** when `schema_version` is present, including every
adapter-emitted pack):

```yaml
schema_version: "0.1"          # dataset manifest contract version
tier: smoke                    # smoke | ci | nightly | release
source:
  id: squad_v1.1
  revision: "dev-v1.1"
  revision_digest: "sha256:..."   # of pinned archive bytes
  canonical_url: "https://..."    # required for adapter pins
  redistributable_smoke: true
  attribution: "docs/datasets.md#squad-v11"
adapter:
  name: squad_v1_1
  version: "1.0.0"
  sample_seed: 42
  sample_size: 16
task_metrics:
  - squad_f1
  - exact_match
contamination_risk: medium     # low | medium | high
pii_scrub_procedure: "docs/datasets.md#scrub-squad"  # required if pii_scrubbed
```

**Loader / validator behavior (decided):**

1. Extend `DatasetManifest` with typed optional fields for the additive keys
   above (`source`, `adapter`, `tier`, `task_metrics`, `contamination_risk`,
   `schema_version`, `pii_scrub_procedure`). No untyped catch-all bag.
2. If `schema_version` is absent (legacy synthetic fixtures): validate only
   today's required fields; ignore unknown top-level keys with a warning.
3. If `schema_version` is present: unknown top-level keys are **errors**; all
   additive fields listed above are required; `source.revision_digest` and
   `source.canonical_url` are required for adapter packs.
4. Existing rules unchanged: `content_sha256` mismatch fails; `split: holdout`
   without `--i-am-doing-a-final-eval` fails.

## Adapter CLI (single name)

Materialization is invoked only as:

```text
uv run evalctl dataset materialize \
  --adapter <adapter_name> \
  --source <path-to-pinned-snapshot> \
  --out <dataset_dir> \
  --seed <int> \
  --size <int> \
  --tier smoke|ci|nightly|release
```

The CLI is a thin wrapper over `tools/datasets/<adapter>`. Do not document a
second public entrypoint. Exit codes: `0` success; `1` validation/license/digest
failure; `2` I/O.

Determinism: two runs with identical args and source bytes produce identical
`cases.jsonl` bytes and `content_sha256`.

Pin integrity: before emit, hash source bytes and compare to
`source.revision_digest`. Mismatch → `SOURCE_DIGEST_MISMATCH`, no write.

### SQuAD v1.1 pin (A2)

Dataset card and adapter pin must record both URL and digest before any commit:

| Field | Value |
|-------|--------|
| `canonical_url` | `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json` |
| `revision` | `dev-v1.1` |
| `revision_digest` | `sha256:<hex>` of those exact bytes, computed and written into `docs/datasets.md` in A1/A2 before smoke commit |

Train split is out of smoke scope unless a separate card + pin is added later.

## License machine checks

This package is distributed under **MIT**. Committed fixture text must be
compatible with that distribution model (including downstream commercial use).

### Ban list (never commit source text; cache-only + operator rights)

- Any **NC** (non-commercial) license, including CC BY-NC-*
- Any **DUA** / gated clinical or restricted corpus (including MIMIC)
- News **fulltext** corpora when redistribution of article body is unclear or
  forbidden (treat AG News fulltext as cache-only unless the card proves an
  allow-listed SPDX and redistribution right)
- Explicitly: **Financial PhraseBank**, **CNN/DailyMail**, **XSum**, and
  similar summarization fulltext packs → **cache-only**, never git

### SPDX allow list for git-committed smoke

Only these SPDX ids may set `redistributable_smoke: true` and write under
`fixtures/`:

| SPDX | Notes |
|------|--------|
| `CC0-1.0` | Preferred |
| `MIT` | OK |
| `Apache-2.0` | OK |
| `BSD-2-Clause` / `BSD-3-Clause` | OK |
| `CC-BY-4.0` | Attribution path required |
| `CC-BY-SA-4.0` | Attribution path required; fixture remains under SA; card must state SA obligations for redistributors |

Anything else → `LICENSE_BLOCK` and cache-only output.

### Pre-commit checks (all required)

Before writing into a git-tracked fixtures path:

1. Dataset card exists in `docs/datasets.md` (source, version, license, SPDX,
   redistribution, contamination, PII scrub procedure, task/metric pairing,
   canonical URL + sha256 for the pin).
2. `redistributable_smoke` is true.
3. SPDX `license` is on the allow list above and is **not** NC/DUA.
4. Attribution path is non-empty for BY/BY-SA.
5. Adapter tests assert published field length caps and absence of raw document
   dumps (see bounds).

Otherwise output only to cache path (default `.cache/datasets/`) and print
`LICENSE_BLOCK`.

## Published text bounds

Aligned with Phase 3 report truncation (`EXAMPLE_TEXT_LIMIT = 280` for gallery
text; stricter where noted):

| Field class | Max chars in committed / published artifacts |
|-------------|-----------------------------------------------|
| Case `inputs` string values | 2_000 per key (smoke); reject oversize |
| `reference_answer` / label | 500 |
| Retrieval doc text in cases | Prefer id + title; body ≤ 2_000 if required |
| Suite failure gallery (from `case_examples`) | Inherit report truncation (280); no `raw_response` |
| Adapter must not embed | Full article/doc dumps, unredacted PII |

Dataset card defines the scrub procedure; `pii_scrubbed: true` is a claim that
procedure was applied.

## Size tiers

| Tier | n (dev) | Provider default | Git |
|------|---------|------------------|-----|
| smoke | 5–20 | mock | commit only if allow-list + card pass |
| ci | 50–200 | mock | cache or CI artifact |
| nightly | 500–2,000 | digest-pinned real | cache |
| release | frozen holdout / capped | digest-pinned | holdout text not casually committed; final-eval flag required |

Dev calibration sets and holdout evaluation sets are **mechanically separated**
(different `split` values and dataset dirs). Holdout never used for threshold
fitting or judge gate agreement on `dev`.

## Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `SOURCE_DIGEST_MISMATCH` | Pin does not match bytes | No |
| `LICENSE_BLOCK` | Cannot write to committed path | No |
| `CASE_VALIDATION_FAILED` | Emitted bundle fails `validate_dataset` | No |
| `NONDETERMINISTIC_ADAPTER` | Self-check hash drift | No |
| `FIELD_LENGTH_EXCEEDED` | Published text over cap | No |
| `UNKNOWN_MANIFEST_KEY` | Extra key when `schema_version` set | No |

## Non-goals

- Runtime download inside `evalctl run`
- MIMIC / DUA corpora in this phase
- Committing NC or news-fulltext packs
- Replacing `cases.jsonl` with Arrow/Parquet
- A second materialize entrypoint beside `evalctl dataset materialize`
