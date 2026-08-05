# Decomposition: Phases 4–6

Contract-first, dependency-ordered, independently green units. Each unit leaves
`main` deployable. Line counts are targets; one concern per PR matters more.

```text
A1 ──► A2 ──► A3 ──┬──► A4 ──► A5
                   ├──► A6a (parallel after A3; license-gated)
                   ├──► A6b (parallel after A3; license-gated)
                   └──► A7  (parallel after A3; license-gated / cache-only)
A5 + ≥2 families ──► B1 ──► B2 ──► B3
B3 (Phase 5 done) ──► C1 ──► C2 ──┬──► C3
                                  ├──► C4
                                  └──► C5
C2–C5 stable ──► C6 (optional suite panels)
```

Strict phase sequence for this delivery: **Phase 4 → Phase 5 → Phase 6**. Phase 6
(`C*`) does not start until Phase 5 **B3** is green. Phase 4 and Phase 6 are not
run in parallel here even though `C1–C5` touch no Phase 5 code, because the
delivery requirement and the phase-6 template pin Phase 6 behind the suite
contract.

New committed families after A5 land with **A5′** (matrix extension PR) rather
than blocking A5 on future adapters.

## Phase 4 units

### A1 — Dataset policy + adapter contract (S)

- Add `docs/datasets.md` (policy, SPDX allow list vs MIT, ban list, tiers, card
  template, PII scrub, contamination).
- Implement typed manifest extension + validator rules from
  [contracts/dataset-adapter.md](contracts/dataset-adapter.md).
- Stub `tools/datasets/` + `evalctl dataset materialize` wrapper.
- Record SQuAD pin placeholders (URL required; sha256 filled when bytes verified).
- **Verify:** `dataset-validate` on `fixtures/sample_dataset`; ruff; mypy;
  pytest. No HF in core deps.

### A2 — SQuAD smoke adapter + QA template (M)

- Pin `dev-v1.1.json` URL + sha256 in card; materialize smoke only after
  allow-list + attribution + card checks.
- Constrained short-answer template; field-length tests.
- **Verify:**
  ```bash
  uv run evalctl dataset-validate fixtures/datasets/squad-v1.1-smoke
  uv run evalctl dataset materialize --adapter squad_v1_1 ... --check-deterministic
  uv run pytest tests/datasets/test_squad_smoke.py -q
  ```

### A3 — Classification smoke adapter (M)

- Allow-listed source only (e.g. PubMedQA if SPDX clears; AG News fulltext is
  cache-only by default). Second **committed** task family unlocks Phase 5.
- **Verify:** validate + mock run + `classification` rescore.

### A4 — Extraction smoke adapter (M)

- License-safe extraction smoke + JSON template; parallel with A3 after A2.

### A5 — CI harness-correctness matrix over **present** committed smokes (M)

- Parametrize mock E2E across task shapes that have **committed** fixtures at
  merge time (not a wishlist of future families).
- Wire CI only for those smokes.
- **Verify:** full CI; PoC goldens clean.

### A5′ — Matrix extension (S, as needed)

- When A6a/A6b/A4/A7 commit a new smoke, extend the matrix in a small follow-up
  PR. Does not reopen A5.

### A6a — Summarization adapter (M, license-gated)

- CNN-DM / XSum are **cache-only** (never commit). Smoke may use a synthetic or
  allow-listed substitute if one exists; otherwise cache-path tests only.
- Primary metrics for summarization smoke: **`rouge_l`**, **`chrf_pp`**.
- **`meteor` is optional**: enable only when NLTK resources are Present
  (vendored/pinned in the enabling PR) or skip with an explicit skip reason.
- **Verify:** validate + overlap rescore without requiring meteor on default CI.

### A6b — Retrieval adapter (M, license-gated)

- SciFact / NFCorpus per license card; commit only if allow-listed.
- Metric: `retrieval_ndcg_10`.
- **Verify:** validate + mock/rescore path. Parallel with A6a after A3.

### A7 — Finance / healthcare adapters (M, cache-first)

- Financial PhraseBank: **cache-only**, never commit (NC).
- FiQA/FinQA/PubMedQA: commit only if SPDX allow-list + card pass; else cache.
- No MIMIC.

### A8 — Digest-pinned Ollama model-quality smoke (S, ops)

- `workflow_dispatch` pattern; not default PR gate.

**Phase 4 exit (unlocks Phase 5 only):** A1–A3 green (policy + ≥2 committed task
families) and A5 covering those families. Phase 6 stays blocked until Phase 5 B3.

## Phase 5 units

### B1 — Suite contracts + validator (S)

- [contracts/suite.md](contracts/suite.md); `evalctl suite validate`.
- No `store` / `providers` imports.
- **Verify:** valid/invalid manifest unit tests.

### B2 — `suite.json` assembler (M)

- Reads report 2.1 + compare JSON (+ optional calibration paths later).
- **Verify:** golden `suite.json`; byte-stable serialize.

### B3 — `suite.html` + CLI build (M)

- Altair offline embed; golden under `fixtures/suite/`.
- **Verify:** build + diff golden; no CDN; PoC fixtures untouched.

## Phase 6 units

Phase 6 implementation units **C1–C5 start only after Phase 5 B3 is green**.
`C1–C5` are standalone (no Phase 5 imports), so `C6` (suite panels) only needs
their schemas stable, not a further wait. `C6` also assumes B3 exists, which the
strict sequence already guarantees.

### C1 — Human-label schema + file fixtures (S)

- Label JSONL schema with `split` + `label_set_id`.
- Commit synthetic **dev** and synthetic **holdout** fixtures for CI (distinct
  ids). No production human `dev` labels for gate tests.
- **Verify:** schema tests; files only.

### C2 — Rubric + pointwise judge artifact (M)

- `evalctl judge run --mode pointwise` reading a file-primary
  `candidates-pointwise.jsonl` + `--provider mock --responses …`
  (see [contracts/judge.md](contracts/judge.md)); judgment JSON always
  `gating_allowed: false`; truncation tests for `reasoning`; pinned judge
  identity + families recorded.
- **Verify:** mock provider (no DB); artifact schema + truncation tests.

### C3 — Pairwise swap + Bradley–Terry refuse rule (M)

- Both orderings; ties on inconsistency.
- BT only under the concrete connected-component rule in
  [contracts/judge.md](contracts/judge.md); assert `DISCONNECTED_PAIRWISE_GRAPH`
  payload shape.
- **Verify:** swap-flip fixture; disconnected graph fixture.

### C4 — Offline `judge validate` + attach (S)

- File-only:

  ```bash
  uv run evalctl judge validate \
    --judgments fixtures/judge/judgment.json \
    --labels-dev fixtures/judge/labels-dev.jsonl \
    --labels-holdout fixtures/judge/labels-holdout.jsonl \
    --rubric fixtures/judge/rubric-pointwise.yaml \
    --output /tmp/calibration.json
  ```

- Assert gate false on `dev`-only; true only on synthetic holdout meeting
  threshold + family separation; attach-calibration merge path.
- **Verify:** no DB required.

### C5 — RAG evidence (M)

- `evalctl rag evidence` file-primary: `--report` (retrieval, read-only) +
  `--evidence` JSONL double; deterministic methods `claim_split_v1`,
  `qrels_context_v1`, `cited_present_relevant_v1`; NLI via Provider seam or
  `NLI_UNAVAILABLE`; NLI-grounded refinements deferred to
  [ADR-004](adr/004-rag-methods-minimal.md); truncation; no HF on default pytest.
- **Verify:** unsupported-claim fixture; retrieval independent of faithfulness;
  every optional section carries an explicit `status` (no bare nulls).

### C6 — Optional suite panels (S)

- Depends on stable C2–C5 schemas (B3 already present under the strict sequence).
- Gate badges from calibration paths.

**Phase 6 exit:** informational artifacts; `gating_allowed` false without
holdout calibration; deterministic metrics untouched; Phase 7 not started.

## Parallelization map

| Parallel set | Notes |
|--------------|-------|
| {A3, A4} | After A2 |
| {A6a, A6b, A7} | After A3; license/cache gated |
| {B1…B3} | Phase 5; after Phase 4 exit. **Not** parallel with Phase 6 |
| {C3, C4, C5} | After C2 (which follows B3) |

Phase 6 does not run in parallel with Phase 5 for this delivery; the phases are
strictly sequential.

Critical path to suite UI: **A1 → A2 → A3 → A5 → B1 → B2 → B3**.  
Critical path to calibrated judge bit:
**A1 → … → Phase 4 exit → B1 → B2 → B3 → C1 → C2 → C4**.

## Global verify (every merge)

```bash
uv run ruff check .
uv run mypy src/evalharness
uv run pytest -q
git diff --exit-code -- fixtures/poc/report.json fixtures/poc/report.html fixtures/poc/meta.json
```

Never weaken PoC goldens to pass suite/judge work. Default pytest must not
download HF models or datasets.
